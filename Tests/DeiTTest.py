#!..\.VENVS\torch\Scripts\python.exe

import sys
sys.path.append("..")
sys.path.append("..\Models")
from Models.ViT.ViTBase import ViT
from Models.LayerUtils import BasicClassificationHead
from Models.ViT.DeiTIII import DEITIIISetup

import torch
from torch.utils.data import Dataset
from torchvision import datasets
import torch.distributed as dist
import torch.multiprocessing as mp
import os


def setup(rank, world_size):
	os.environ["CUDA_VISIBLE_DEVICES"] = "0" 
	os.environ['MASTER_ADDR'] = 'localhost'
	os.environ['MASTER_PORT'] = '12355'

	# initialize the process group
	dist.init_process_group("gloo", rank=rank, world_size=world_size)


def demo_basic(rank, world_size):
	print(f"Running basic DDP example on rank {rank}.")
	setup(rank, world_size)
	classes = ('plane', 'car', 'bird', 'cat',
	       'deer', 'dog', 'frog', 'horse', 'ship', 'truck')


	vitParams = dict(numHeads=2,numLayers=2,D=768,mlpSize=512, dropout=0.1, registers=3, 
		kernelPEG=None, survivalProbability=0.9, layerScale=True)
	vit = ViT(**vitParams)
	vit = BasicClassificationHead(vit, embedSize=768, numClasses=len(classes))
	NUM_EPOCHS=20
	
	deit = DEITIIISetup(vit,224, distributed=True, checkpointsFolder =r"..\CheckPoints")
	trainset = datasets.CIFAR10(root=r'..\Datasets\CIFAR10', train=True, download=True, transform=deit.augmentation)
	deit.setup(rank,world_size, nEpochs=NUM_EPOCHS, batchSizeGPU=32, dataPath="", dataset=trainset, dataLoaderWorkers=4)


	for epoch in range(deit.curentEpoch,NUM_EPOCHS):
		deit.startEpoch()
		running_loss = 0.0
		loop = deit.getLoader()
		for i, (data,labels) in enumerate(loop):
			deit.forwardNetworks(data,labels)
			deit.updateNetworks()
			print(f"Epoch [{epoch + 1}/{NUM_EPOCHS}] - Loss: {running_loss / (i+1):.3f}")
			running_loss += deit.getLossValue()

		deit.endEpoch()





if __name__ == '__main__':
	#device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
	#demo_basic(device,1)
	world_size = 1
	mp.spawn(demo_basic,
			args=(world_size,),
			nprocs=world_size,
			join=True)
