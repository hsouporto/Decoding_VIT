#!..\.VENVS\plots\Scripts\python.exe
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

import os


CSV = "LIDC-dataset-info-wRandom.csv"
os.chdir(r"..\Datasets\Chest")

df = pd.read_csv(CSV)

# remove class 3
df = df[~np.isnan(df['Malignant'])]

# filter based on the pixel count of the mask
df = df[df["Pixel Presence"] > -1]

# set classes
df.loc[df["Malignant"] == -1, "Malignant"] = 2
df = df[["Image Path","Malignant"]]
df["Malignant"] = df["Malignant"].astype(int)
df = df.reset_index(drop=True)

indexes = np.array(df.index)
classes = np.array(df["Malignant"])


#split 70 - 15 - 15
train,test,trainClasses,_ = train_test_split(indexes,classes, shuffle=True,stratify=classes, test_size=0.15,random_state=42)
train,validation = train_test_split(train, shuffle=True,stratify=trainClasses, test_size=0.1765,random_state=42)

df["Set"] = ""

df.loc[train,"Set"] = "Train"
df.loc[test,"Set"] = "Test"
df.loc[validation,"Set"] = "Validation"




print(len(df[df["Set"]=="Train"]))
print(len(df[df["Set"]=="Test"]))
print(len(df[df["Set"]=="Validation"]))

print(np.unique(df["Malignant"]))

df.to_csv("final_LIDC.csv",index=False)
