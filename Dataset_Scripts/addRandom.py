#!..\.VENVS\plots\Scripts\python.exe
import numpy as np
import os
import pandas as pd


IMAGES = "LIDC_IDRI"

DF_PATH = "LIDC-dataset-info.csv"


def setSeed(seed=42):
	np.random.seed(seed=seed)
	pass

def getImageFolder(patientId):
	splitID = patientId.split("_")
	if len(splitID) > 1:
		subDir = int(splitID[1])
	else:
		subDir = 0
	patientId = splitID[0]
	pathIm = os.path.join(IMAGES, patientId,f"CT_{subDir}")
	return pathIm


def getUnusedImages(usedImages,path):
	used = [x[len(path):] for x in usedImages]
	unusedImages = np.array([int(x[:-len(".dcm")]) for x in os.listdir(path) if x.endswith(".dcm") and x not in used])
	return unusedImages

def RandomPick(toPick,pickSize,imageFolder):
	return [os.path.join(imageFolder,f"{x:04d}.dcm") for x in np.random.choice(toPick,pickSize,replace=False)]

def getImageRatio(usedImages,clearRatio=0.3):
	toAdd = int(len(usedImages) * clearRatio)
	return toAdd

def getUsedImages(patientId,df):
	auxdf = df[df["Patient ID"] == patientId]
	return list(auxdf["Image Path"])


if __name__ == '__main__':
	os.chdir(r"..\Datasets\Chest")
	setSeed(42)
	df = pd.read_csv(DF_PATH)
	for patientId in np.unique(df["Patient ID"]):
		imagePath = getImageFolder(patientId)
		used = getUsedImages(patientId,df)
		unused = getUnusedImages(used,imagePath)
		pickSize = getImageRatio(used)
		toAdd = RandomPick(unused,pickSize,imagePath)
		for image in toAdd:
			df.loc[len(df.index)] = [patientId,None,-1,image,.0]

	df = df.set_index("Patient ID")
	df = df.sort_index()
	df = df.reset_index()
	df.to_csv("LIDC-dataset-info-wRandom.csv",index=False)


