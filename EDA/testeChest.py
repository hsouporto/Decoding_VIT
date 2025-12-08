#!..\.VENVS\torch\Scripts\python.exe
import numpy as np
import cv2
import pydicom as dicom
import os
from einops import rearrange



image_range = (39,47)
ncols = 8
nrow = 1

scale = 128


basefolder = r"..\datasets\chest\image\LIDC-IDRI-0001\01-01-2000-30178\3000566.000000-03192"
sets = os.listdir(basefolder)
xml = [x for x in sets if x.endswith(".xml")]
dcms = [os.path.join(basefolder,x) for x in sets if x.endswith(".dcm")][image_range[0]:image_range[1]]

print(len(dcms))

def readDCM(dcmPath):
	ds=dicom.dcmread(dcmPath)
	pixel_array = ds.pixel_array
	return pixel_array


acc = None
for dcm in dcms:
	partial = readDCM(dcm)
	acc = partial if acc is None else np.hstack([acc,partial])



image_shape = 512,512
n_images = len(dcms)
addition = (ncols*nrow) - n_images



if addition < 0:
	raise Exception("Insufficient Cols or Rows")


mask = np.flip(np.flip(np.flip(np.load(r"..\datasets\chest\mask\LIDC-IDRI-0001\nodule_00.npy")),axis=1),axis=2)
mask =  rearrange(mask, 'n h w -> h (n w)')[:,(image_range[0]*512):(image_range[1]*512)]
print(mask.shape)



complement = np.zeros((image_shape[0],image_shape[1]*addition),dtype="int")

mask = np.hstack([mask,complement])
mask = rearrange(mask, 'h (i j w) -> (i h) (j w)', i=nrow, j=ncols)

acc = np.hstack([acc,complement])
acc = rearrange(acc, 'h (i j w) -> (i h) (j w)', i=nrow, j=ncols)



normalized = np.uint8((acc.astype(np.float32) / acc.max()) * 255)
acc = cv2.applyColorMap(normalized, cv2.COLORMAP_BONE)
vertical = cv2.resize(acc,(ncols*scale,nrow*scale))


mask = mask.astype("uint8")*127
print(np.unique(mask))
mask = cv2.resize(mask,(ncols*scale,nrow*scale))

verticalMasked = np.copy(vertical)

verticalMasked[:,:,2] += mask




basefolder = r"..\datasets\chest\image\LIDC-IDRI-0001\01-01-2000-35511\3000923.000000-62357"
sets = os.listdir(basefolder)
xml = [x for x in sets if x.endswith(".xml")]
dcms = [os.path.join(basefolder,x) for x in sets if x.endswith(".dcm")]



acc = None
for dcm in dcms:
	partial = readDCM(dcm)
	acc = partial if acc is None else np.hstack([acc,partial])


normalized = np.uint8((acc.astype(np.float32) / acc.max()) * 255)
horizontal = cv2.applyColorMap(normalized, cv2.COLORMAP_BONE)
horizontal = cv2.resize(horizontal,(1024,512))


cv2.imshow('mask',verticalMasked)
cv2.imshow('vertical',vertical)
cv2.imshow('horizontal',horizontal)
cv2.waitKey()


