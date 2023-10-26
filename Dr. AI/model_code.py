import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report,confusion_matrix,accuracy_score,r2_score
from sklearn.preprocessing import LabelEncoder
import pickle

df=pd.read_csv("Training.csv")
df=df.drop(columns=["Unnamed: 133"])
df=df.dropna()
df=df.drop(columns=['fluid_overload'])

y_train=df.iloc[:,-1]
x_train=df.drop(columns=["prognosis"])

x_test=pd.read_csv("Testing.csv")
x_test=x_test.dropna()
x_test=x_test.drop(columns=["fluid_overload"])
y_test=x_test.iloc[:,-1]
x_test=x_test.drop(columns=["prognosis"])

le=LabelEncoder()
y_train=le.fit_transform(y_train)
y_test=le.fit_transform(y_test)

dt=DecisionTreeClassifier(criterion='entropy')
dt.fit(x_train,y_train)

pickle.dump(dt,open("model.pkl","wb"))
model=pickle.load(open("model.pkl","rb"))