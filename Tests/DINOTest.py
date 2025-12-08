#!..\.VENVS\torch\Scripts\python.exe
import sys
sys.path.append("..")
sys.path.append("..\Models")
from Models.DINO.DINO import DINOSetup
from Models.ViT.ViTBase import ViT
from torchvision import datasets
import torch
import torch.nn as nn
from tqdm import tqdm
import torch.distributed as dist
import torch.multiprocessing as mp
import os


def setup(rank, world_size):
	os.environ["CUDA_VISIBLE_DEVICES"] = "0" 
	os.environ['MASTER_ADDR'] = 'localhost'
	os.environ['MASTER_PORT'] = '12355'

	# initialize the process group
	dist.init_process_group("gloo", rank=rank, world_size=world_size)
	import builtins as __builtin__
	builtin_print = __builtin__.print

	def print(*args, **kwargs):
		builtin_print(*args, **kwargs)

	__builtin__.print = print	

def cleanup():
	dist.destroy_process_group()


def demo_basic(rank, world_size):
	print(f"Running basic DDP example on rank {rank}.")
	setup(rank, world_size)

	vitParams = dict(numHeads=2,numLayers=2,D=768,mlpSize=512, dropout=0.1, registers=3, kernelPEG=None, survivalProbability=0.9)
	teacher = ViT(**vitParams)
	student = ViT(**vitParams)
	dino = DINOSetup(200,teacher,student, distributed=True)
	NUM_EPOCHS=20
	trainset = datasets.CIFAR10(root=r'..\Datasets\CIFAR10', train=True, download=True, transform=dino.augmentation)
	
	dino.setup(rank, world_size, nEpochs=NUM_EPOCHS, batchSizeGPU=32, dataPath="", dataset=trainset, dataLoaderWorkers=4)

	for epoch in range(dino.curentEpoch,NUM_EPOCHS):
		dino.startEpoch()
		running_loss = 0.0
		loop = dino.getLoader()
		for i, (data,_) in enumerate(loop):
			dino.forwardNetworks(*data)
			dino.updateNetworks()
			#print(f"Epoch [{epoch + 1}/{NUM_EPOCHS}] - Loss: {running_loss / (i+1):.3f}")
			# print statistics
			running_loss += dino.getLossValue()
		dino.endEpoch()
	cleanup()




if __name__ == '__main__':
	world_size = 1
	mp.spawn(demo_basic,
			args=(world_size,),
			nprocs=world_size,
			join=True)

