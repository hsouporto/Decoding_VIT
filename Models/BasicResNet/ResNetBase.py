#!..\..\.VENVS\torch\Scripts\python.exe
import torch
import torch.nn as nn
from torchviz import make_dot
from torchvision import datasets
from torchvision import transforms

class ResidualBlock(nn.Module):
	def __init__(self,firstChannelInput, kernelDims, channelsOut, blocks=3, first_stride=2):
		super().__init__()
		blks = []
		channelsIn = [channelsOut[-1]] + [x for x in channelsOut[:-1]]
		self.relu = nn.ReLU()
		
		if firstChannelInput != channelsOut[:-1]:
			self.shortcut = nn.Sequential(
			    nn.Conv2d(firstChannelInput, channelsOut[-1], kernel_size=1, bias=False, stride=first_stride),
			    nn.BatchNorm2d(channelsOut[-1])
			)

		for blk in range(blocks):
			convs = []
			for kernelSz,chnIn,chnOut in zip(kernelDims,channelsIn,channelsOut):
				isFirst = (len(convs) + len(blks)) == 0
				chnIn = firstChannelInput if isFirst else chnIn
				stride = first_stride if isFirst else 1
				conv = nn.Sequential(
					nn.Conv2d(chnIn,chnOut,kernelSz,stride,padding=1),
					nn.BatchNorm2d(chnOut)
				)
				convs.append(conv)
			blks.append(nn.ModuleList(convs))
		self.blks = nn.ModuleList(blks)

	
	def forward(self,x):
		for i,blk in enumerate(self.blks):
			residual = self.shortcut(x) if i == 0 else x
			for i,conv in enumerate(blk):
				x = conv(x)
				if i < (len(blk) - 1):
					x = self.relu(x)
			x = x + residual
			x = self.relu(x)
		return x





class ResNet(nn.Module):
	def __init__(self, blocksArgs, channelInputSize=3, numClasses=1000):
		super().__init__()
		self.kickStart = nn.Sequential(
			nn.Conv2d(channelInputSize,64,7,stride=2, padding=3, bias= False),
			nn.BatchNorm2d(64),
			nn.MaxPool2d(3,stride=2, padding=1)
		)
		
		blocks = []
		lastChannelOut = 64
		for kernelDims,channelsOut,blocks_num,first_stride in blocksArgs:
			blocks.append(ResidualBlock(lastChannelOut,kernelDims,channelsOut,blocks_num,first_stride))
			lastChannelOut = channelsOut[-1]

		
		self.blocks= nn.ModuleList(blocks)
		
		self.pool = nn.AdaptiveAvgPool2d(1)
		self.final= nn.Sequential(
            nn.Linear(lastChannelOut, numClasses),
            nn.Softmax(dim=1)
		)

	def forward(self,img):
		x = self.kickStart(img)
		for block in self.blocks:
			x = block(x)
		x = torch.flatten(self.pool(x), start_dim=1)
		return self.final(x)


		
		
RESNET50_BLOCKS = [
	[
		[1,3,1],
		[64,64,256],
		3,1
	],
	[
		[1,3,1],
		[128,128,512],
		4,2
	],
	[
		[1,3,1],
		[256,256,1024],
		6,2
	],
	[
		[1,3,1],
		[512,512,2048],
		3,2
	],
]

RESNET34_BLOCKS = [
	[
		[3,3],
		[64,64],
		3,1
	],
	[
		[3,3],
		[128,128],
		4,2
	],
	[
		[3,3],
		[256,256],
		6,2
	],
	[
		[3,3],
		[512,512],
		3,2
	],
]

RESNET18_BLOCKS = [
	[
		[3,3],
		[64,64],
		2,1
	],
	[
		[3,3],
		[128,128],
		2,2
	],
	[
		[3,3],
		[256,256],
		2,2
	],
	[
		[3,3],
		[512,512],
		2,2
	],
]

if __name__ == '__main__':
	rnt = ResNet(RESNET50_BLOCKS,numClasses=10)
	print(rnt)
	print(sum([param.numel() for param in rnt.parameters()]))
	
	rnt = ResNet(RESNET18_BLOCKS,numClasses=10)
	image = torch.rand([100,3,224,224])
	y = rnt(image)
	make_dot(y,params=dict(list(rnt.named_parameters()))).render("ResNet", format="png")