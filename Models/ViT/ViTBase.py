#!..\..\.VENVS\torch\Scripts\python.exe
import sys
sys.path.append("..")
from LayerUtils import StochasticDepth, BasicClassificationHead


import torch
import torch.nn as nn
from einops import rearrange
from einops.layers.torch import Rearrange
from torchviz import make_dot



class AttentionHead(nn.Module):
	def __init__(self, embedDim, dimH, dropout=0., bias_qkv=False):
		super().__init__()
		self.dimH = dimH
		self.scale = dimH**(-0.5)
		self.linearProjQ = nn.Linear(embedDim, dimH, bias = bias_qkv)
		self.linearProjK = nn.Linear(embedDim, dimH, bias = bias_qkv)
		self.linearProjV = nn.Linear(embedDim, dimH, bias = bias_qkv)
		self.logits = nn.Softmax(dim=1)
		self.dropout = nn.Dropout(dropout)
	
	def forward(self, q, k, v):
		q = self.linearProjQ(q)
		k = self.linearProjK(k)
		v = self.linearProjV(v)
		k = rearrange(k,"b n d -> b d n") #transpose K
		attentionScores = torch.matmul(q,k)
		scaledAttn = attentionScores * self.scale
		logitAttn = self.logits(self.dropout(attentionScores))
		return torch.matmul(logitAttn,v)



class MyMultiHeadAttention(nn.Module):
	def __init__(self, embedDim, dimH, heads=8, dropout=0., bias_qkv=False):
		super().__init__()
		self.attentionHeads = nn.ModuleList([AttentionHead(embedDim, dimH, dropout=dropout, bias_qkv=bias_qkv) for _ in range(heads)])
		self.MLP = nn.Sequential(
			nn.Linear(dimH*heads,embedDim),
			nn.Dropout(dropout)
		)

	def forward(self, x):
		attns = [head(x,x,x) for head in self.attentionHeads]
		return self.MLP(torch.concat(attns,dim=-1))


class EncoderLayer(nn.Module):
	def __init__(self,heads, D, mlpSize, dimH=64, dropout=0., bias_qkv=False, survivalProbability=1., layerScale=False, **kwargs):
		super().__init__()
		self.normA = nn.LayerNorm(D)
		self.normB = nn.LayerNorm(D)
		self.att = MyMultiHeadAttention(D, dimH, heads, dropout=dropout, bias_qkv=bias_qkv)
		self.stochastic = StochasticDepth(survivalProbability)
		self.mlp = nn.Sequential(
			nn.Linear(D, mlpSize),
			nn.GELU(),
			nn.Dropout(dropout),
			nn.Linear(mlpSize, D),
			nn.Dropout(dropout),
		)

		self.gamma_1 = torch.ones((D))
		self.gamma_2 = torch.ones((D))

		if layerScale:
			self.gamma_1 = nn.Parameter(1e-4*self.gamma_1,requires_grad=True)
			self.gamma_2 = nn.Parameter(1e-4*self.gamma_2,requires_grad=True)
		else:
			self.gamma_1 = nn.Parameter(self.gamma_1,requires_grad=False)
			self.gamma_2 = nn.Parameter(self.gamma_2,requires_grad=False)

	def forward(self, x, *args):
		x = self.normA(x)
		y = self.att(x)
		x = x + self.stochastic(self.gamma_1 * y)
		x = self.normB(x)
		y = self.mlp(x)
		x = x + self.stochastic(self.gamma_2 *y)
		return x




class ClassAttention(EncoderLayer):
	def __init__(self, *args,registers=0,**kwargs):
		self.registers = registers
		super().__init__(*args,**kwargs)

	
	def forward(self,x):
		x = self.normA(x)
		y = self.att(x)
		if self.registers > 0:
			y = torch.cat((y[:,0:,:],y[:,-self.registers:,:]),dim=1)
			z = torch.cat((x[:,0:,:],x[:,-self.registers:,:]),dim=1)
		else:
			y = y[:,0:,:]
			z = x[:,0:,:]
		
		z = z + self.stochastic(self.gamma_1 * y)
		z = self.normB(z)
		y = self.mlp(z)
		z = z + self.stochastic(self.gamma_2 * y)
		
		if self.registers > 0:
			x = torch.cat((z[:,0:1,:], x[:,1:,:],z[:,-self.registers:,:]), dim=1)
		else:
			x = torch.cat((z[:,0:1,:], x[:,1:,:]), dim=1)

		return x
		




class PatchEmbed(nn.Module):
	def __init__(self, patchSize, channels, D):
		super().__init__()
		self.D = D
		self.linearProjection = nn.Sequential(
			Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1=patchSize, p2=patchSize),
			nn.LayerNorm((patchSize**2)* channels),
			nn.Linear((patchSize**2)* channels, D),
			nn.LayerNorm(D),
		)
	
	def forward(self, x):
		return self.linearProjection(x)



class PositionalEncoding(nn.Module):
	def __init__(self, imageW, imageH, patchSize, D):
		super().__init__()
		self.patchSize = patchSize
		self.D = D
		self.positionEncoders = nn.Parameter(torch.randn(1, D, imageW//patchSize, imageH//patchSize))


	def forward(self, x, imageW, imageH, classTokens, registers):
		#interpolate
		positional = nn.functional.interpolate(self.positionEncoders,(imageW//self.patchSize, imageH//self.patchSize),mode='bicubic')
		positional = rearrange(positional,"1 d w h -> 1 (w h) d")
		
		clsToken = None
		regs = None
		embeds = x
		if classTokens > 0:
			clsToken = x[:,0:classTokens,:]
			embeds = embeds[:,classTokens:,:]
		
		if registers > 0:
			regs = x[:,-registers:,:]
			embeds = embeds[:,:-registers,:]

		embeds = embeds + positional
		toCat = [tens for tens in [clsToken,embeds,regs] if tens is not None]
		x = torch.cat(toCat, dim=1)

		return x



class ConditionalPositionalEncoding(nn.Module):
	def __init__(self, kernelPEG, patchSize, D):
		super().__init__()
		assert (kernelPEG % 2) > 0, "Kernel Size should be an odd number"
		padding = (kernelPEG - 1)//2 
		self.patchSize = patchSize
		self.conv = nn.Conv2d(D, D, kernel_size=kernelPEG, padding=padding, groups=D)


	def forward(self, x, imageW, imageH, classTokens, registers):
		w0 = imageW//self.patchSize
		h0 = imageH//self.patchSize
		if registers <= 0:
			embeds = x[:,classTokens:,:]
		else:
			embeds = x[:,classTokens:-registers,:]

		positional = rearrange(embeds,"b (w h) d -> b d w h", w=w0, h=h0)
		positional = self.conv(positional)	
		positional = rearrange(positional,"b d w h -> b (w h) d")
		
		clsToken = None
		regs = None
		embeds = x
		if classTokens > 0:
			clsToken = x[:,0:classTokens,:]
			embeds = embeds[:,classTokens:,:]
		if registers > 0:
			regs = x[:,-registers:,:]
			embeds = embeds[:,:-registers,:]
		embeds = embeds + positional
		toCat = [tens for tens in [clsToken,embeds,regs] if tens is not None]
		x = torch.cat(toCat, dim=1)
		return x



class ViT(nn.Module):
	def __init__(self,
		imageSize=(224,224),
		patchSize=16,
		channels=3,
		numHeads=12,
		Dhead=64,
		numLayers=12,
		D=768,
		mlpSize=3072,
		registers=0,
		bias_qkv=False,
		kernelPEG=None, #size of convolution used on Conditional position encoding
		layerScale=False,
		laterClassToken=0, #LayerScalePaper regularization (insert class token after N encoder blocks)
		laterRegisterToken=0, #insert registers after N encoder blocks - addition to the layer scale paper
		classAttentionLayers=0, #LayerScalePaper regularization (last N layers are devoted to the class token)
		includeRegistersOnCA=False, #add registers on ClassAttention layers - addition to the layer scale paper
		dropout=0.,
		survivalProbability = 1., #Stochastic Depth (depends on arch, 0.9 for B - 0.7 for L - 0.5 for H - 1 for the rest in DeiTIII)
		):
		super().__init__()
		self.patchEmbed = PatchEmbed(patchSize, channels, D)
		self.clsToken = nn.Parameter(torch.randn(1, 1, D))
		self.registers = nn.Parameter(torch.randn(1, registers, D)) if registers > 0 else None
		self.registersNum = registers

		self.kernelPEG = kernelPEG
		if kernelPEG is not None or kernelPEG == 0:
			self.positional = ConditionalPositionalEncoding(kernelPEG, patchSize, D)
		else:
			self.positional = PositionalEncoding(*imageSize, patchSize, D)

		self.registerPostionEncoders = nn.Parameter(torch.randn(1, registers, D))
		self.clsPositionEncoder = nn.Parameter(torch.randn(1, 1, D))
		
		self.laterClassToken = laterClassToken
		self.laterRegisterToken = laterRegisterToken
		self.classAttentionLayers = classAttentionLayers
		self.includeRegistersOnCA = includeRegistersOnCA

		#check if class token is added before CA:
		normalLayers = numLayers - classAttentionLayers
		assert laterClassToken <= normalLayers, "Class Token should be added before or within the class attention Layer"


		self.encodeLayers = nn.ModuleList(
			[
				EncoderLayer(numHeads, D, mlpSize, dimH=Dhead, dropout=dropout,layerScale=layerScale,survivalProbability=survivalProbability) 
				for _ in range(normalLayers)
			]
			+ 
			[
				ClassAttention(numHeads, D, mlpSize, 
					dimH=Dhead, dropout=dropout,survivalProbability=survivalProbability, 
					registers=(registers*int(includeRegistersOnCA)), layerScale=layerScale
				) 
				for _ in range(classAttentionLayers)
			]
		)

	def addRegisters(self,x):
		if self.registers is not None:
			registers = self.registers.expand(x.shape[0], -1, -1)
			regsPE = self.registerPostionEncoders.expand(x.shape[0], -1, -1)
			x = torch.cat((x, registers + regsPE), dim=1)
		return x

	def addCLSToken(self,x):
		clsPE = self.clsPositionEncoder.expand(x.shape[0], -1, -1)
		cls_tokens = self.clsToken.expand(x.shape[0], -1, -1)
		return torch.cat((cls_tokens + clsPE, x), dim=1)

	
	def forward(self, x):
		imW = x.shape[-2]
		imH = x.shape[-1]
		layerClassToken = self.laterClassToken
		layerRegisterToken = self.laterRegisterToken
		layerPositional = 1	if self.kernelPEG is not None else 0
		classTokens = 0
		registers = 0	

		x = self.patchEmbed(x)
		for layer in self.encodeLayers:
			if layerClassToken == 0:
				x = self.addCLSToken(x)
				classTokens = 1

			if layerRegisterToken == 0:
				x = self.addRegisters(x)
				registers = self.registersNum
			
			if layerPositional == 0:
				x = self.positional(x,imW,imH,classTokens, registers)
			
			x = layer(x)
			layerClassToken -= 1
			layerRegisterToken -= 1
			layerPositional -= 1

		return x[:, 0]





if __name__ == '__main__':
	
	vitB = ViT()
	vitB = BasicClassificationHead(vitB,embedSize=768)
	from torchinfo import summary

	summary(vitB, input_size=(50, 3, 224, 224))

	#print(vitB)
	#print(sum([param.numel() for param in vitB.parameters()]))
	#
	#vitTest = ViT((224,224),16,numHeads=2,numLayers=2,mlpSize=512, D=64, numClasses=1000)
	#image = torch.rand([100,3,224,224])
	#y = vitTest(image)
	#make_dot(y,params=dict(list(vitTest.named_parameters()))).render("VitBase", format="png")