#!..\..\.VENVS\torch\Scripts\python.exe
import torch
import torch.nn as nn
import numpy as np
import os
import torch.distributed as dist

class StochasticDepth(nn.Module):
	def __init__(self, survivalProbability):
		super().__init__()
		self.survivalProbability = survivalProbability

	def forward(self,x):
		if self.survivalProbability == 1. or not self.training:
			return x
		shape = (x.shape[0],) + (1,) * (x.ndim - 1)
		randomT = torch.bernoulli(torch.ones(shape) * self.survivalProbability).type(x.dtype).to(x.device)
		return x.div(self.survivalProbability) * randomT


class BasicClassificationHead(nn.Module):
	def __init__(self, Arch, embedSize=768, numClasses=1000, hiddenLayers=[], activationFunctions=[], applySoftMax=False):
		super().__init__()
		layers = []
		lastOut = embedSize
		for out,func in zip(hiddenLayers,activationFunctions):
			layers.append(nn.Linear(lastOut, out))
			layers.append(func)
			lastOut = out

		layers.append(nn.Linear(lastOut, numClasses))
		if applySoftMax:
			layers.append(nn.Softmax(dim=1))
		self.layers = nn.Sequential(*layers)
		self.Arch = Arch

	def forward(self,x):
		x = self.Arch(x)
		return self.layers(x)


class L2NormalizationLayer(nn.Module):
    def __init__(self, dim=1, eps=1e-12):
        super(L2NormalizationLayer, self).__init__()
        self.dim = dim
        self.eps = eps

    def forward(self, x):
        return nn.functional.normalize(x, p=2, dim=self.dim, eps=self.eps)



class LayerScale(nn.Module):
	"""docstring for LayerScale"""
	def __init__(self, arg):
		super(LayerScale, self).__init__()
		self.arg = arg
		



class CosineScheduler(object):
	def __init__(self, base_value, final_value, epochs, niter_per_ep, warmup_epochs=0, start_warmup_value=0):
		warmup_schedule = np.array([])
		warmup_iters = warmup_epochs * niter_per_ep
		if warmup_epochs > 0:
			warmup_schedule = np.linspace(start_warmup_value, base_value, warmup_iters)

		iters = np.arange(epochs * niter_per_ep - warmup_iters)
		schedule = final_value + 0.5 * (base_value - final_value) * (1 + np.cos(np.pi * iters / len(iters)))

		schedule = np.concatenate((warmup_schedule, schedule))
		assert len(schedule) == epochs * niter_per_ep
		self.schedule = schedule

	def __getitem__(self,items):
		items = min(items,len(self.schedule)-1)
		return self.schedule[items]













class BaseSetup(object):
	def __init__(self,arch, distributed=True, clipGradient=0, 
		baseArchName="Network", runName="run-1",
		checkpointsFolder =r"..\CheckPoints"
		):
		self.augmentation = None
		self.loss = None
		self.optimizer = None
		self.learningRateScheduler = None
		
		self.distributed = distributed
		self.architecture = arch

		self.runName = runName
		self.baseArchName = baseArchName
		self.checkpointsFolder = checkpointsFolder

		self.clipGradient = clipGradient
		self.epochs = 0
		self.curentIteration = 0
		self.curentEpoch = 0
		self.epochSize = 0
		self.device = 0
		self.optimizer_state_dict = None
		self.optimizerKwargs = {}
	

	def initializeOptmizer(self):
		self.optimizer = self.optimizer_call(self.architecture.parameters(), **self.optimizerKwargs)
		if self.optimizer_state_dict is not None:
			self.optimizer.load_state_dict(self.optimizer_state_dict)

	
	def setup(self, rank, world_size, nEpochs, batchSizeGPU, dataPath,
			learningRate=3e-3, minLearningRate=None, warmup_epochs=10,
			dataLoaderWorkers=10, seed=42,
			dataset=None
		):

		self.device = rank
		torch.manual_seed(seed)
		np.random.seed(seed)

		self.world_size = world_size
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

		#============== setup scheduler
		if minLearningRate is None:
			minLearningRate = learningRate * (batchSizeGPU * world_size) / 256.

		self.learningRateScheduler = CosineScheduler(
			learningRate * (batchSizeGPU * world_size) / 256.,  # linear scaling rule
			minLearningRate,
			self.epochs, self.epochSize,
			warmup_epochs=warmup_epochs,
		)

		#========= setup Network ===========

		self.loadCheckPoint()
		self.architecture = self.architecture.to(self.device)
		# synchronize batch norms (if any)
		if self.distributed:
			if self.has_batchnorms():
				self.architecture = nn.SyncBatchNorm.convert_sync_batchnorm(self.architecture)
			self.architecture = nn.parallel.DistributedDataParallel(self.architecture, device_ids=[self.device])

		self.initializeOptmizer()

		#============ setup loss =================
		self.loss.to(self.device)
		pass


	def startEpoch(self):
		self.curentIteration = 0
		self.curentEpoch += 1
		self.optimizer.zero_grad()


	def forwardNetworks(self, images, labels):
		self.curentIteration += 1
		self.globalIteration = (self.epochSize * self.curentEpoch) + self.curentIteration
		self.optimizer.updateSchedule(self.learningRateScheduler[self.globalIteration])
		# move images to gpu
		images = images.to(self.device)
		labels = labels.to(self.device)
		out = self.architecture(images)
		self.calculated_loss = self.loss(out,labels)


	def updateNetworks(self):
		self.calculated_loss.backward()
		archParams = self.architecture.module.parameters() if self.distributed else self.architecture.parameters()
		
		if self.clipGradient:
			nn.utils.clip_grad_norm_(archParams,self.clipGradient)

		self.optimizer.step()

	def endEpoch(self):
		self.saveCheckPoint()
	

	def has_batchnorms(self):
		bn_types = (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d, nn.SyncBatchNorm)
		for name, module in self.architecture.named_modules():
			if isinstance(module, bn_types):
				return True
		return False
	
	def getLoader(self):
		return self.data_loader		

	def getLossValue(self):
		lossValue = self.calculated_loss
		if self.distributed:
			dist.all_reduce(lossValue) # sync the loss across multiple nodes/GPUS for logging
		return lossValue.item()/self.world_size


	def saveCheckPoint(self):
		if not self.distributed or int(self.device) == 0:
			modelState = self.architecture if not self.distributed else self.architecture.module
			modelState = modelState.state_dict()
			optimizerState = self.optimizer.state_dict()

			torch.save({
				'epoch': self.curentEpoch,
				'model_state_dict': modelState,
				'optimizer_state_dict': optimizerState,
				'loss': self.calculated_loss,
				}, self.getPath())
			
		if self.distributed:
			dist.barrier()


	
	def loadCheckPoint(self):
		pathName = self.getPath(lastIter=True)
		if pathName is not None:
			checkpoint = torch.load(pathName)
			self.architecture.load_state_dict(checkpoint['model_state_dict'])
			self.optimizer_state_dict = checkpoint['optimizer_state_dict']
			self.curentEpoch = checkpoint['epoch']
			self.calculated_loss = checkpoint['loss']
			
		if self.distributed:
			dist.barrier()
		return self.curentEpoch


	def getPath(self, lastIter=False):
		checkFolder = os.path.join(self.checkpointsFolder,self.baseArchName,self.runName)
		if lastIter and not os.path.isdir(checkFolder):
			return None

		os.makedirs(checkFolder,exist_ok=True)
		fileName = "checkpoint_"
		extension = ".tar"
		
		if lastIter:
			directoryList = os.listdir(checkFolder)
			if len(directoryList) <= 0:
				return None
			pastIters = [int(x[len(fileName):-len(extension)]) for x in directoryList]
			iteration = max(pastIters)

		else:
			iteration = self.curentEpoch
		
		fileName = f"{fileName}{iteration}{extension}"
		return os.path.join(checkFolder,fileName)


	def startEvaluation(self):
		self.architecture.eval()
		pass





if __name__ == '__main__':
	a = CosineScheduler(1,1, 5,5)
	print(a.schedule)