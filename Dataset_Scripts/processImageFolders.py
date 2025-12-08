import os
import shutil

def removeFolderTreshold(folderPath, treshold=12):
	meta = sum([len(files) for _,_,files in os.walk(folderPath)])
	if meta < treshold:
		shutil.rmtree(folderPath)
		print("REMOOVE",folderPath)
	return meta < treshold

def contiguousFileName(folderPath):
	toRename = [x for x in os.listdir(folderPath) if x.endswith(".dcm")]
	toRename.sort(key=lambda x: (int(x[0]),int(x[2:-len(".dcm")])))
	for i,x in enumerate(toRename):
		print("RENAME", os.path.join(folderPath,x),os.path.join(folderPath,f"{i:04d}.dcm"))
		os.rename(os.path.join(folderPath,x),os.path.join(folderPath,f"{i:04d}.dcm"))
	pass


def contiguousFolder(foldersPath):
	foldersPathlist = os.listdir(foldersPath)
	for i,folder in enumerate(foldersPathlist):
		print("RENAME", os.path.join(foldersPath,folder),os.path.join(foldersPath,f"CT_{i}"))
		os.rename(os.path.join(foldersPath,folder),os.path.join(foldersPath,f"CT_{i}"))
	pass


def moveAllToParent(folderPath):
	# Iterate through each file in the parent directory
	for root,_,files in os.walk(folderPath):
		for file in files:
			source_path = os.path.join(root, file)
			destination_path = os.path.join(folderPath, file)
			shutil.move(source_path, destination_path)


CT_SCANS_BASE = r"D:\Tese\ViT-Tese\Datasets\Chest\LIDC_IDRI"


for folders in os.listdir(CT_SCANS_BASE):
	folders = os.path.join(CT_SCANS_BASE,folders)
	for folder in os.listdir(folders):
		folder = os.path.join(folders,folder)
		if not removeFolderTreshold(folder):
			moveAllToParent(folder)
			contiguousFileName(folder)
	contiguousFolder(folders)



