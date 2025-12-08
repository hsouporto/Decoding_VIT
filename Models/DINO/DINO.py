#!..\..\.VENVS\torch\Scripts\python.exe

import sys
#sys.path.append("..")
from LayerUtils import BasicClassificationHead, L2NormalizationLayer, CosineScheduler, BaseSetup

import os
import torch
import torch.nn as nn
from torchvision import datasets,transforms
import torch.distributed as dist
import numpy as np





class DINOClassificationHead(BasicClassificationHead):
	def __init__(self, Arch, outDim, embedSize=768, bottleneck=256, hiddenLayers=[2048]):
		super().__init__(Arch,embedSize,numClasses=bottleneck,hiddenLayers=hiddenLayers,
			activationFunctions=[nn.GELU() for _ in hiddenLayers],applySoftMax=False)
		
		self.l2Norm = L2NormalizationLayer(dim=-1)
		self.lastLayer = nn.utils.parametrizations.weight_norm(nn.Linear(bottleneck, outDim, bias=False))
		
		#This two forces the layer to be always normalized (e.g. between 0 and 1) - it locks the magnitude (original_0)
		#from the vector as 1, 
		self.lastLayer.parametrizations.weight.original0.data.fill_(1)
		self.lastLayer.parametrizations.weight.original0.requires_grad = False

	def forward(self, x):
		grouped_inputs = {}
		for input_tensor in x:
			resolution = tuple(input_tensor.shape[-2:])
			if resolution not in grouped_inputs:
				grouped_inputs[resolution] = torch.empty(0).to(x[0].device)
			
			grouped_inputs[resolution] = torch.cat([grouped_inputs[resolution],input_tensor])
		
		output = torch.empty(0).to(x[0].device)
		for resolutions in grouped_inputs:
			out = self.Arch(grouped_inputs[resolutions])
			output = torch.cat([output,out])
		
		x = output
		x = self.layers(x)
		x = self.l2Norm(x)
		x = self.lastLayer(x)
		return x




class DINOLoss(nn.Module):
	def __init__(self, out_dim, ncrops, teacher_temp=0.04, student_temp=0.1,center_momentum=0.9, distributed=True, world_size=1):
		# Class equal to https://github.com/facebookresearch/dino/blob/main/main_dino.py
		super().__init__()
		self.distributed = distributed
		self.student_temp = student_temp
		self.teacher_temp = teacher_temp
		self.center_momentum = center_momentum
		self.ncrops = ncrops
		self.world_size = world_size
		self.register_buffer("center", torch.zeros(1, out_dim))

	def forward(self, student_output, teacher_output):
		student_out = student_output / self.student_temp
		student_out = student_out.chunk(self.ncrops)
		teacher_out = nn.functional.softmax((teacher_output - self.center) / self.teacher_temp, dim=-1)
		teacher_out = teacher_out.detach().chunk(2)

		total_loss = 0
		n_loss_terms = 0
		for iq, q in enumerate(teacher_out):
			for v in range(len(student_out)):
				if v == iq:
					#skip cases where student and teacher operate on the same view
					continue
				loss = torch.sum(-q * nn.functional.log_softmax(student_out[v], dim=-1), dim=-1)
				total_loss += loss.mean()
				n_loss_terms += 1
		total_loss /= n_loss_terms
		self.update_center(teacher_output)
		return total_loss

	@torch.no_grad()
	def update_center(self, teacher_output):
		batch_center = torch.sum(teacher_output, dim=0, keepdim=True)
		if self.distributed:
			dist.all_reduce(batch_center) # sync thew batch across multiple nodes/GPUS
		
		batch_center = batch_center / (len(teacher_output) * self.world_size)
		self.center = self.center * self.center_momentum + batch_center * (1 - self.center_momentum)





class DINOAugmentation(object):
	def __init__(self, local_crops_number, global_crops_scale=(0.4, 1.), local_crops_scale=(0.05, 0.4), 
		normalizeMean=(0.485, 0.456, 0.406),normalizeStd=(0.229, 0.224, 0.225)):
		self.local_crops_number = local_crops_number
		flip_and_color_jitter = transforms.Compose([
			transforms.RandomHorizontalFlip(p=0.5),
			transforms.RandomApply(
				[transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.2, hue=0.1)],
				p=0.8
			),
			transforms.RandomGrayscale(p=0.2),
		])
		normalize = transforms.Compose([
			transforms.ToTensor(),
			transforms.Normalize(normalizeMean, normalizeStd),
		])

		def gaussianBlur(p):
			return transforms.Compose([
				transforms.RandomApply(
					[transforms.GaussianBlur(5,(0.1,2.))],p #match pillow gaussian blur
				)
			])
		# first global crop
		self.global_crop=[
			transforms.Compose([
				transforms.RandomResizedCrop(224, scale=global_crops_scale, interpolation=transforms.InterpolationMode.BICUBIC ),
				flip_and_color_jitter,
				gaussianBlur(1.0),#Gaussian
				normalize,
			]),
			transforms.Compose([
				transforms.RandomResizedCrop(224, scale=global_crops_scale, interpolation=transforms.InterpolationMode.BICUBIC ),
				flip_and_color_jitter,
				gaussianBlur(0.1),#Gaussian
				transforms.RandomSolarize(128, p=0.2), #to match pillow solarize equal to original dino
				normalize,
			])
		]
		# transformation for the local small crops
		self.local_view = transforms.Compose([
			transforms.RandomResizedCrop(96, scale=local_crops_scale, interpolation=transforms.InterpolationMode.BICUBIC ),
			flip_and_color_jitter,
			gaussianBlur(p=0.5),#gaussian
			normalize,
		])

	def __call__(self, image):
		teacherViews = [gl(image) for gl in self.global_crop]
		studentViews = []
		for _ in range(self.local_crops_number):
		    studentViews.append(self.local_view(image))
		return [teacherViews,studentViews]



class DINOOptmizer(object):
	def __init__(self, student, learningRateScheduler, weightDecayScheduler, optimizer="Adamw", distributed=False):
		student = student if not distributed else student.module

		regularizedParams = {
			"params": [param for name,param in student.named_parameters() if not(name.endswith(".bias") or len(param.shape) == 1)],
			"name": "regularized"
		}
		non_regularizedParams = {
			"params": [param for name,param in student.named_parameters() if name.endswith(".bias") or len(param.shape) == 1],
			"weight_decay": 0.,
			"name": "non_regularized"
		}
		self.lr_schedule = learningRateScheduler
		self.wd_schedule = weightDecayScheduler

		if optimizer.lower() == "adamw":
			self.optimizer = torch.optim.AdamW([regularizedParams,non_regularizedParams])  # to use with ViTs
		else:
			self.optimizer = torch.optim.SGD(params_groups, lr=0, momentum=0.9)  # lr is set by scheduler

	def updateSchedule(self, globalTrainingIteration=0):
		for param_group in self.optimizer.param_groups:
			param_group["lr"] = self.lr_schedule[globalTrainingIteration]
			if param_group["name"] == "regularized":
				param_group["weight_decay"] = self.wd_schedule[globalTrainingIteration]

	def zero_grad(self):
		self.optimizer.zero_grad()

	def step(self):
		self.optimizer.step()

	def state_dict(self):
		return self.optimizer.state_dict() 

	def load_state_dict(self, state_dict):
		if state_dict is not None:
	 		self.optimizer.load_state_dict(state_dict) 




class DINOSetup(BaseSetup):
	def __init__(self, outDim, teacher, student,  # main args
			embedSize=768, bottleneck=256, hiddenLayers=[], #Head Kwargs
			teacher_temp=0.04, student_temp=0.1,center_momentum=0.9, #Loss Kwargs
			global_crops_scale=(0.4, 1.), local_crops_scale=(0.05, 0.4),local_crops_number=8,  #Augmentation Kwargs
			normalizeMean=(0.485, 0.456, 0.406),normalizeStd=(0.229, 0.224, 0.225), #Augmentation Kwargs (mean and std from imagenet - input your own from dataset)
			clipGradient=1.,
			nameLastLayer="lastLayer", epochsLastLayerFreezed=1,
			distributed = True,
			baseArchName="DINO",runName="run-1", checkpointsFolder =r"..\CheckPoints"
		):

		

		self.outDim = outDim
		self.base_teacher = teacher
		self.base_student = student

		super().__init__(self.base_student,distributed=distributed,clipGradient=clipGradient, 
			baseArchName=baseArchName, runName=runName,
			checkpointsFolder=checkpointsFolder
			)

		global_crops_number = 2
		ncrops = local_crops_number + global_crops_number
		
		self.teacher = DINOClassificationHead(teacher, outDim, 
			embedSize=embedSize, 
			bottleneck=bottleneck, 
			hiddenLayers=hiddenLayers
		)
		self.student = DINOClassificationHead(student, outDim, 
			embedSize=embedSize, 
			bottleneck=bottleneck, 
			hiddenLayers=hiddenLayers
		)
		self.augmentation = DINOAugmentation(
			local_crops_number=local_crops_number, 
			global_crops_scale=global_crops_scale, 
			local_crops_scale=local_crops_scale
		)
		self.loss = DINOLoss(outDim, ncrops, 
			teacher_temp=teacher_temp, student_temp=student_temp, 
			center_momentum=center_momentum, distributed=distributed,
		)

		self.epochsLastLayerFreezed = epochsLastLayerFreezed
		self.calculated_loss= 0

		



	def setup(self, rank, world_size, nEpochs, batchSizeGPU, dataPath,
			learningRate=0.0005, minLearningRate=0.000001, warmup_epochs=10, #LearningRate
			weight_decay=0.04, weight_decay_end=0.4, momentumTeacher=0.996, #momentum-WeightDecay Kwargs
			optimizer="Adamw", #Optimizer Kwargs
			dataLoaderWorkers=10, seed=42,
			dataset=None
		): 
		#Distribution on multiple machines and set important variables		
		#basic Setup
		self.device = rank
		torch.manual_seed(seed)
		np.random.seed(seed)

		self.world_size = world_size
		self.loss.world_size = world_size
		self.epochs = nEpochs
		self.batch_size = batchSizeGPU
		#============== setup dataset ============
		if dataset is None:
			dataset = datasets.ImageFolder(dataPath, transform=self.augmentation)
		
		sampler = None
		if self.distributed:
			sampler = torch.utils.data.DistributedSampler(dataset, shuffle=True)
		
		self.data_loader = torch.utils.data.DataLoader(
			dataset,
			sampler=sampler,
			batch_size=batchSizeGPU,
			num_workers=dataLoaderWorkers,
			pin_memory=True,
			drop_last=True,
		)
		self.epochSize = len(self.data_loader)

		#============== setup optimizer
		learningRateScheduler = CosineScheduler(
			learningRate * (batchSizeGPU * world_size) / 256.,  # linear scaling rule
			minLearningRate,
			self.epochs, self.epochSize,
			warmup_epochs=warmup_epochs,
		)

		weightDecayScheduler = CosineScheduler(
			weight_decay,
			weight_decay_end,
			self.epochs, self.epochSize
		)
		self.momentumScheduler = CosineScheduler(momentumTeacher, 1, self.epochs, self.epochSize)

		#========= setup Teacher Student ===========

		self.loadCheckPoint()
		self.student = self.student.to(self.device)
		self.teacher = self.teacher.to(self.device)
		self.teacher_without_ddp = self.teacher

		starterTeacher = self.student.state_dict()
		# synchronize batch norms (if any)
		if self.distributed:
			if self.has_batchnorms():
				self.student = nn.SyncBatchNorm.convert_sync_batchnorm(self.student)
				self.teacher = nn.SyncBatchNorm.convert_sync_batchnorm(self.teacher)

				# we need DDP wrapper to have synchro batch norms working...
				self.teacher = nn.parallel.DistributedDataParallel(teacher, find_unused_parameters=False,device_ids=[self.device])
				self.teacher_without_ddp = teacher.module
			self.student = nn.parallel.DistributedDataParallel(self.student, find_unused_parameters=False,device_ids=[self.device])
			starterTeacher = self.student.module.state_dict()
		
		if self.curentEpoch <= 0:
			self.teacher_without_ddp.load_state_dict(starterTeacher)
		
		self.optimizer = DINOOptmizer(self.student, learningRateScheduler, weightDecayScheduler, optimizer=optimizer)
		self.optimizer.load_state_dict(self.optimizer_state_dict)


		for p in self.teacher.parameters():
			p.requires_grad = False

		#============ setup loss =================
		self.loss.to(self.device)
		pass

	def startEpoch(self):
		self.curentIteration = 0
		self.curentEpoch += 1


	def endEpoch(self):
		self.saveCheckPoint()

	
	def forwardNetworks(self, teacherViews, studentViews):
		self.curentIteration += 1
		self.globalIteration = (self.epochSize * self.curentEpoch) + self.curentIteration
		self.optimizer.updateSchedule(self.globalIteration)
		
		# move images to gpu
		teacherViews = [im.to(self.device) for im in teacherViews]
		studentViews = [im.to(self.device) for im in studentViews]
		
		teacher_output = self.teacher(teacherViews)
		student_output = self.student(teacherViews+studentViews)
		self.calculated_loss = self.loss(student_output, teacher_output)


	def updateNetworks(self):
		self.optimizer.zero_grad()
		self.calculated_loss.backward()
		
		studentParams = self.student.module.parameters() if self.distributed else self.student.parameters()

		if self.clipGradient:
			nn.utils.clip_grad_norm_(studentParams,self.clipGradient)
		
		if self.curentEpoch < self.epochsLastLayerFreezed:
			for name,parameter in self.student.named_parameters():
				if nameLastLayer in name:
					parameter.grad = None
		self.optimizer.step()

		# EMA update for the teacher
		with torch.no_grad():
			m = self.momentumScheduler[self.globalIteration]  # momentum parameter
			
			for param_q, param_k in zip(studentParams, self.teacher_without_ddp.parameters()):
				param_k.data.mul_(m).add_((1 - m) * param_q.detach().data)

	
	def saveCheckPoint(self):
		if (not self.distributed) or int(self.device) == 0:
			studentState = self.student if not self.distributed else self.student.module
			studentState = studentState.state_dict()
			teacherState = self.teacher_without_ddp.state_dict()
			optimizerState = self.optimizer.state_dict()
			torch.save({
				'epoch': self.curentEpoch,
				'student_state_dict': studentState,
				'teacher_state_dict': teacherState,
				'optimizer_state_dict': optimizerState,
				'loss': self.calculated_loss,
				},self.getPath())
		
		if self.distributed:
			dist.barrier()

	
	def loadCheckPoint(self):
		pathName = self.getPath(lastIter=True)
		if pathName is not None:
			checkpoint = torch.load(pathName)
			self.student.load_state_dict(checkpoint['student_state_dict'])
			self.teacher.load_state_dict(checkpoint['teacher_state_dict'])
			self.optimizer_state_dict = checkpoint['optimizer_state_dict']

			self.curentEpoch = checkpoint['epoch']
			self.calculated_loss = checkpoint['loss']
			
			self.student.train()
			self.teacher.train()
			
			if self.distributed:
				dist.barrier()

		return self.curentEpoch


#NOTE - given the nature of DINO's multicrop augmentation this network will only work if the interpolation class is already defined in the ViT
#This regularization method will only work to train a backbone - the classifier needs to be finetuned after

