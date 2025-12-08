#!..\..\.VENVS\hugFace\Scripts\python.exe
from transformers import ViTImageProcessor, ViTForImageClassification
import torch
from torchvision.models import resnet50, ResNet50_Weights
from torchvision import datasets
from torchvision import transforms
from tqdm import tqdm
import timm
import numpy as np
import os



class HugTransform(object):
	def __init__(self, processor):
		self.processor = processor

	def __call__(self, image):
		return self.processor(image,return_tensors="pt")
		

if __name__ == '__main__':
	device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
	
	

	# get model specific transforms (normalization, resize)

	model = timm.create_model('deit3_base_patch16_224.fb_in22k_ft_in1k', pretrained=True)
	#model = timm.create_model('deit3_large_patch16_224.fb_in22k_ft_in1k', pretrained=True)
	
	data_config = timm.data.resolve_model_data_config(model)
	transforms = timm.data.create_transform(**data_config, is_training=False)

	model.eval()

	batch_size = 32

	testset = datasets.ImageNet(root=r'..\..\Datasets\IMNET1K', split="val", transform=transforms)
	testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=8)

	model.to(device)
	correct = 0.0
	totalSamples = 0

	fullOuts = torch.empty(0)
	fullLabels = torch.empty(0)
	with torch.no_grad():
		loop = tqdm(testloader)
		for i, data in enumerate(loop):
			images, labels = data[0].to(device), data[1].to("cpu")
			fullLabels = torch.cat((fullLabels,labels)).to("cpu")
			# calculate outputs by running images through the network
			outputs = model(images)
			# the class with the highest energy is what we choose as prediction
			loop.set_description(f"Evaluation on Test Set")
			fullOuts = torch.cat((fullOuts,outputs.to("cpu")),dim=0).to("cpu")
			loop.set_postfix(loss=f"{fullOuts.size()}")

		outs = fullOuts.to("cpu").numpy()
		labels = fullLabels.to("cpu").numpy()
		np.save(r"..\Results\DeitIIIBase_HUG_IMNET1k.npy",outs)
		#np.save(r"..\Results\IMNET1kLABELS.npy",labels)
