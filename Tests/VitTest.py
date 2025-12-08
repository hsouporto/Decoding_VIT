#!..\.VENVS\torch\Scripts\python.exe
import sys
sys.path.append("..")
sys.path.append("..\Models")
from Models.ViT.ViTBase import ViT
from Models.LayerUtils import BasicClassificationHead
import torch
from torch.utils.data import Dataset
from torchvision import datasets
from torchvision import transforms
import torchvision
import matplotlib.pyplot as plt
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from tqdm import tqdm


if __name__ == '__main__':
	device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
	transform = transforms.Compose(
	    [
	    	transforms.ToTensor(),
	    ])
	
	transformTest = transforms.Compose(
	    [
	    	#transforms.Resize(64),
	    	transforms.ToTensor(),
	    ])
	
	classes = ('plane', 'car', 'bird', 'cat',
	       'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
	
	###############################################################
	train = True
	test = True
	batch_size = 100
	NUM_EPOCHS = 20

	############################################################### 
	vit = ViT(imageSize=(32,32),patchSize=8,numHeads=8,numLayers=7,D=512,mlpSize=512, dropout=0.1, registers=0, 
		kernelPEG=None, layerScale=False,laterClassToken=0,laterRegisterToken=0,classAttentionLayers=0,
		includeRegistersOnCA=False, survivalProbability=1.).to(device)
	transEnc = BasicClassificationHead(vit, embedSize=512, numClasses=len(classes)).to(device)
	
	#transEnc = ViT(image_size=32,patch_size=16,num_classes=len(classes),dim=64,depth=6,heads=8,mlp_dim=512,dropout=0.1).to(device)
	

	print(transEnc)
	criterion = nn.CrossEntropyLoss()
	optimizer = optim.Adam(transEnc.parameters(), lr=0.0001)
	#optimizer = optim.SGD(transEnc.parameters(), lr=0.01, momentum=0.9)
	
	###############################################################

	trainset = datasets.CIFAR10(root=r'..\Datasets\CIFAR10', train=True, download=True, transform=transform)
	trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=2)

	testset = datasets.CIFAR10(root=r'..\Datasets\CIFAR10', train=False, download=True, transform=transformTest)
	testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=2)


	# functions to show an image
	def imshow(img):
	    img = img / 2 + 0.5     # unnormalize
	    npimg = img.numpy()
	    plt.imshow(np.transpose(npimg, (1, 2, 0)))
	    plt.show()


	# get some random training images
	dataiter = iter(trainloader)
	images, labels = next(dataiter)

	# show images
	#imshow(torchvision.utils.make_grid(images))
	# print labels
	#print(' '.join(f'{classes[labels[j]]:5s}' for j in range(batch_size)))




	dataiter = iter(trainloader)
	images, labels = next(dataiter)

	if train:
		for epoch in range(NUM_EPOCHS):  # loop over the dataset multiple times
			running_loss = 0.0
			correct = 0.0
			totalSamples = 0
			loop = tqdm(trainloader)
			for i, data in enumerate(loop):
				# get the inputs; data is a list of [inputs, labels]
				inputs, labels = data[0].to(device), data[1].to(device)

				# zero the parameter gradients
				optimizer.zero_grad()

				# forward + backward + optimize
				outputs = transEnc(inputs)
				#print(outputs.size())
				#print(labels.size())
				loss = criterion(outputs, labels)
				loss.backward()
				optimizer.step()
				correct += (torch.argmax(outputs, dim=1) == labels).float().sum()
				totalSamples += batch_size

				loop.set_description(f"Epoch [{epoch + 1}/{NUM_EPOCHS}]")
				loop.set_postfix(loss=f"{running_loss / (i+1):.3f}",acc=f"{100*correct /(totalSamples) :.2f}%")
				# print statistics
				running_loss += loss.item()


		torch.save(transEnc.state_dict(), r"..\CheckPoints\testeVit.trc")
		print('Finished Training\n\n===============================================================')

	classifier = transEnc
	classifier.load_state_dict(torch.load(r"..\CheckPoints\testeVit.trc"))
	classifier.eval()
	classifier.to(device)

	if test:
		running_loss = 0.0
		correct = 0.0
		totalSamples = 0

		# since we're not training, we don't need to calculate the gradients for our outputs
		with torch.no_grad():
			loop = tqdm(testloader)
			for i, data in enumerate(loop):
				images, labels = data[0].to(device), data[1].to(device)
				# calculate outputs by running images through the network
				outputs = transEnc(images)
				# the class with the highest energy is what we choose as prediction
				correct += (torch.argmax(outputs, dim=1) == labels).float().sum()
				totalSamples += batch_size
				loop.set_description(f"Evaluation on Test Set")
				loop.set_postfix(loss=f"{running_loss / (i+1):.3f}",acc=f"{100*correct /(totalSamples) :.2f}%")


	x = classifier.forward(images.to(device))
	print(x.to("cpu").size())
	print(x)
