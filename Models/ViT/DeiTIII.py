#!..\..\.VENVS\torch\Scripts\python.exe
import torch
import torch.nn as nn
from torchvision import datasets,transforms
import torch.distributed as dist
import numpy as np
from LAMBoptmizer import Lamb
from LayerUtils import BaseSetup

'''
Features from DeiTIII

- Batchsize = 2048
- Optmizer = LAMB

- LR decay cosine (DINO like)
- Weight decay = 0.02
- warmup epochs = 5


- no dropout
- Stochastic Depth (depends on arch, 0.9 for B - 0.7 for L - 0.5 for H - 1 for the rest)
- LayerScale
- Gradiet Clip = 1


-Augmentations: 
	- horizontal flip (torchvision.transforms.RandomHorizontalFlip)
	- 1 of: 
		Grayscale (torchvision.transforms.Grayscale), 
		Solarization (EQUAL TO DINO), 
		Gaussian Blur (EQUAL TO DINO)
	- SRC (torchvision.transforms.RandomCrop)
	- ColorJitter (torchvision.transforms.ColorJitter) = 0.3 


- Cross Entropy
- label smoothing = 0.1 (inside cross entropy )
'''





class DEITAugmentation(object):
	def __init__(self, trainingResolution, normalizeMean=(0.485, 0.456, 0.406),normalizeStd=(0.229, 0.224, 0.225)):
		self.SRC = transforms.Compose([
			transforms.Resize(trainingResolution),
			transforms.RandomCrop(trainingResolution, padding=4, padding_mode='reflect')
			])

		self.flip_and_color_jitter = transforms.Compose([
			transforms.RandomHorizontalFlip(p=0.5),
			transforms.RandomApply(
				[transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.2, hue=0.1)],
				p=0.3
			),
			transforms.RandomGrayscale(p=0.2),
		])
		self.normalize = transforms.Compose([
			transforms.ToTensor(),
			transforms.Normalize(normalizeMean, normalizeStd),
			])

		
		self.options = transforms.RandomChoice([
			transforms.RandomSolarize(128),
			transforms.Grayscale(num_output_channels=3),
			transforms.GaussianBlur(5,(0.1,2.)),
			])

	def __call__(self, image):
		augmented = self.flip_and_color_jitter(image)
		augmented = self.options(augmented)
		augmented = self.SRC(augmented)
		augmented = self.normalize(augmented)
		return augmented



class DEITIIISetup(BaseSetup):
	def constructLamb(*args,**kwargs):
		return Lamb(*args,**kwargs)

	def __init__(self, arch, trainingResolution,
		normalizeMean=(0.485, 0.456, 0.406),normalizeStd=(0.229, 0.224, 0.225),
		label_smoothing=0.1, weight_decay=0.02,clipGradient=1.,
		distributed=True,
		baseArchName="DEIT",runName="run-1", checkpointsFolder =r"..\CheckPoints"
		):
		super().__init__(arch, distributed, clipGradient=clipGradient, 
			baseArchName=baseArchName, runName=runName, 
			checkpointsFolder=checkpointsFolder
		)
		self.augmentation = DEITAugmentation(trainingResolution, normalizeMean=normalizeMean,normalizeStd=normalizeStd)
		self.loss = nn.CrossEntropyLoss(label_smoothing=label_smoothing)
		self.optimizer_call = DEITIIISetup.constructLamb
		self.optimizer_kwargs = dict(weight_decay=weight_decay)

		self.learningRateScheduler = None

	
	
	def setup(self, rank, world_size, nEpochs, batchSizeGPU, dataPath,
			learningRate=3e-3, minLearningRate=3e-4, warmup_epochs=5,
			dataLoaderWorkers=10, seed=42,
			dataset=None
		):

		super().setup(rank, world_size, nEpochs, batchSizeGPU, dataPath,
			learningRate=learningRate, minLearningRate=minLearningRate, warmup_epochs=warmup_epochs,
			dataLoaderWorkers=dataLoaderWorkers, seed=seed,
			dataset=dataset
		)
		pass





if __name__ == '__main__':
	from PIL import Image
	im = Image.open(r"C:\Users\MrXester\Pictures\Sus.png").convert('RGB')
	im.show()
	augment = DEITAugmentation(224)
	for _ in range(5):
		aug = augment(im)
		im = transforms.functional.to_pil_image(aug)
		im.show()












