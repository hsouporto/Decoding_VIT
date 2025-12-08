#!..\.VENVS\plots\Scripts\python.exe
import numpy as np
import os
import cv2
import pandas as pd
import re
from multiprocessing import Process, Manager



MASKS = "Nodule Masks"
IMAGES = "LIDC_IDRI"
NUM_WORKERS = 10


def getMaskFolderName(patientId):
	cut = len("LIDC-IDRI-")
	patientId = int(patientId[cut:].split("_")[0])
	folderNum = patientId // 100
	zipped = (folderNum > 0)
	finalNumber = 99 if folderNum < 10 else 12
	startNumber = 0 if folderNum > 0 else 1
	folderName = f"LIDC-IDRI-{folderNum:02d}{startNumber:02d}-{folderNum:02d}{finalNumber:02d}.zip"
	return os.path.join(MASKS,folderName)

def getImagePath(patientId,imageIndex):
	splitID = patientId.split("_")
	if len(splitID) > 1:
		subDir = int(splitID[1])
	else:
		subDir = 0

	patientId = splitID[0]
	pathImage = os.path.join(IMAGES, patientId,f"CT_{subDir}", f"{imageIndex:04d}.dcm")
	if not os.path.isfile(pathImage):
		print(f"image [{pathImage}] not found")
	return pathImage

def getMaskDensityMap(patientId, nodule, maskFolder):
	masks = np.load(maskFolder)
	maskPath = None
	r = re.compile(fr"((.*\/{patientId})|{patientId})\/nodule_{nodule:02d}")
	maskPath = [x for x in masks if r.match(x)][0]

	print(maskPath)
	mask = masks[maskPath]
	mask = np.flip(mask, axis=(0,1,2))
	sumOfPixels = np.sum(mask, axis=(1,2))
	maskIndex = np.nonzero(sumOfPixels)[0]
	sumOfPixels = sumOfPixels[maskIndex]
	return zip(maskIndex,sumOfPixels)


def getMalignancy(df):
	mode = df["malignancy"].mode()[0]
	if mode == 3:
		return None
	return int(mode > 3) 


def getImages(patientId,nodule):
	maskFolder = getMaskFolderName(patientId)
	densityMap = getMaskDensityMap(patientId,nodule,maskFolder)
	return [(getImagePath(patientId,x),density) for x,density in densityMap]




def prepareFinalDF(procNum,dfSlice,storage):
	finalDF = pd.DataFrame(columns=["Patient ID","Nodule","Malignant","Image Path","Pixel Presence"])

	for patientId,nodule in np.unique(dfSlice.index):
		malignancy = getMalignancy(dfSlice.loc[(patientId,nodule)])
		nodule = nodule - 1
		for imagePath,density in getImages(patientId,nodule):
			finalDF.loc[len(finalDF.index)] = [patientId,nodule,malignancy,imagePath,density]

	storage[procNum] = finalDF



if __name__ == '__main__':
	os.chdir(r"..\Datasets\Chest")
	dfNodules = pd.read_csv("final-lidc-nodule-semantic-scores.csv")
	dfNodules = dfNodules[["patient_id","nodule","annotation_id","malignancy"]]
	dfNodules = dfNodules.set_index(["patient_id","nodule"])
	dfNodules = dfNodules.sort_index()

	#listDFs = np.array_split(dfNodules, 200)
	listDFs = np.array_split(dfNodules, NUM_WORKERS)


	manager = Manager()
	storage = manager.dict()
	procs = []

	for i in range(NUM_WORKERS):
		proc = Process(target=prepareFinalDF, args=(i,listDFs[i],storage))
		procs.append(proc)
		proc.start()


	for proc in procs:
		proc.join()


	df = pd.concat([storage[s] for s in storage], ignore_index=True, axis=0)
	df.to_csv("LIDC-dataset-info.csv",index=False)
