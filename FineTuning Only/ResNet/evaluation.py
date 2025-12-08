#!..\..\.VENVS\torch\Scripts\python.exe
import torch
from torchvision.models import resnet50, ResNet50_Weights
from torchvision.models import resnet34, ResNet34_Weights
from torchvision.models import resnet18, ResNet18_Weights
from torchinfo import summary
from torchvision import datasets
from torchvision import transforms
from tqdm import tqdm
import numpy as np
import os

def collate_fn(batch):
	return (torch.stack([x[0] for x in batch]), torch.tensor([x[1] for x in batch]))	


if __name__ == '__main__':
	device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

	#resnet = resnet18(weights=ResNet18_Weights.DEFAULT)
	#resnet = resnet34(weights=ResNet34_Weights.DEFAULT)
	resnet = resnet50(weights=ResNet50_Weights.DEFAULT)
	resnet.eval()

	#summary(resnet, input_size=(50, 3, 224, 224))

	batch_size = 64


	preprocess = transforms.Compose([
		transforms.Resize((224,224)),
	    transforms.ToTensor(),
	    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
	])

	testset = datasets.ImageNet(root=r'..\..\Datasets\IMNET1K', split="val", transform=preprocess)
	testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=8, collate_fn=collate_fn)


	resnet.to(device)
	correct = 0.0
	totalSamples = 0

	fullOuts = torch.empty(0).to(device)
	fullLabels = torch.empty(0)
	with torch.no_grad():
		loop = tqdm(testloader)
		for i, data in enumerate(loop):
			images, labels = data[0].to(device), data[1]
			fullLabels = torch.cat((fullLabels,labels))
			# calculate outputs by running images through the network
			outputs = resnet(images)
			# the class with the highest energy is what we choose as prediction
			loop.set_description(f"Evaluation on Test Set")
			fullOuts = torch.cat((fullOuts,outputs),dim=0)

		outs = fullOuts.to("cpu").numpy()
		labels = fullLabels.to("cpu").numpy()
		#np.save(r"..\Results\ResNet_Base_18_IMNET1k.npy",outs)
		#np.save(r"..\Results\ResNet_Base_34_IMNET1k.npy",outs)
		np.save(r"..\Results\ResNet_Base_50_IMNET1k.npy",outs)
		#np.save(r"..\Results\IMNET1kLABELS.npy",labels)
