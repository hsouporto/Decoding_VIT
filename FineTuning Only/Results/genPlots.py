#!..\..\.VENVS\plots\Scripts\python.exe

import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("ModelsMetrics.csv")
df = df.set_index(["dataset","Architecture","Type","Size"])
df = df.stack(future_stack=True).reset_index()
df.columns = ["Dataset","Architecture","Type","Size","Metric","Value"]
df['Dataset'].str.replace('IMNET1k','Image Net 1K')

axisNum = len(pd.unique(df["Dataset"]))
fig,ax = plt.subplots(1,axisNum)
axisList = ax if axisNum > 1 else [ax]


palette = {
	"DeitIII":"#9BBDDE",
	"DINO":"#6DD684",
	"ResNet":"#F7665A",
	"ViT":"#D4BF37",
}

markers = {
	"B":"s",
	"L":"P",
	"S":"D",
	"Base":"o",
}

lines = {
	16:"dashed",
	8:"dotted",
	50:"solid",
	18: (0,(3, 1, 1, 1)),
	34:(0, (5, 1)),
}




for toPlot,datasetName in zip(axisList,pd.unique(df["Dataset"])):
	dfAux = df[df["Dataset"]==datasetName]
	for archName in pd.unique(dfAux["Architecture"]):
		color = palette[archName]
		dfArch = dfAux[dfAux["Architecture"]==archName]
		for typeName in pd.unique(dfArch["Type"]):
			marker = markers[typeName]
			dftype = dfArch[dfArch["Type"]==typeName]
			for size in pd.unique(dftype["Size"]):
				lineType = lines[size]
				dfsize = dftype[dftype["Size"]==size]
				toPlot.plot(dfsize["Metric"],dfsize["Value"],color=color, marker=marker, linestyle=lineType)


plt.show()
#print(df[df["Architecture"]=="DeitIII"])


#
#ax.plot(df[""],df["Metric"])
#
#plt.show()
#
#
#print(df[""])
#
#plt.plot(df[])


#possibleColors = ["#B69F00", "#56B4E9", "#F2A590", "#F0E442", "#3333ff"]#,"#0072B2", "#D55E00", "#CC79A7", "#0b5c21","#2a9d8f","#ff6961"] + [sns.color_palette("dark:salmon_r")[0]]
#
#
#plots = sns.FacetGrid(df,row="Dataset")
#plots.map_dataframe(
#	sns.lineplot,
#	x="Metric",y="Value",
#	style="Type",
#	hue="Architecture",
#	palette=possibleColors, 
#	markers=False, 
#	dashes=[(1,0),(2,1),(3,2),(1,3),(2,5)]
#)
#plots.add_legend()
##plots.map(sns.lineplot, "Metric", "Value", markers=True)



plt.show()