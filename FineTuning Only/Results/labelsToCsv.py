#!..\..\.VENVS\plots\Scripts\python.exe
import os
import numpy as np
from scipy.special import softmax
from sklearn.metrics import accuracy_score, f1_score, recall_score, precision_score, top_k_accuracy_score, roc_auc_score
import pandas as pd

groundTruths = {fileName[:-len("LABELS.npy")]: np.load(fileName) for fileName in os.listdir() if fileName.endswith(".npy") and "LABELS" in fileName}
labels = {}

for datasetName in groundTruths:
	modelsLabels = {fileName[:-len(f"{datasetName}.npy")]: softmax(np.load(fileName),axis=1) for fileName in os.listdir() if fileName.endswith(f"{datasetName}.npy") and "LABELS" not in fileName}
	modelsLabels["GT"] = groundTruths[datasetName]
	labels[datasetName] = modelsLabels


def processName(name):
	archName = name.split("_")[0]
	archType = name.split("_")[1]
	archSize = name.split("_")[2]
	return [archName,archType,archSize]



df = pd.DataFrame(columns=["dataset","Architecture","Type","Size","Top1 Accuracy", "F1-Score", "Recall","Precision","Top5 Accuracy","ROC AUC"])


for dataset in labels:
	gt = labels[dataset]["GT"]
	for model in labels[dataset]:
		if model != "GT":
			modelInfo = [dataset]
			pred = labels[dataset][model]
			predlabel = np.argmax(pred,axis=1)
			print(predlabel.shape)
			print(gt.shape)
			modelInfo += processName(model)
			modelInfo.append(accuracy_score(gt,predlabel))
			modelInfo.append(f1_score(gt,predlabel, average="macro"))
			modelInfo.append(recall_score(gt,predlabel, average="macro"))
			modelInfo.append(precision_score(gt,predlabel, average="macro"))
			modelInfo.append(top_k_accuracy_score(gt,pred,k=5))
			modelInfo.append(roc_auc_score(gt,pred, multi_class="ovr"))
			df.loc[len(df)] = modelInfo

print(df)
df.to_csv("ModelsMetrics.csv", index=False)