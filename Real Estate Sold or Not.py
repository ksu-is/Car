import matplotlib.pyplot as plt
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
import statsmodels.api as sm
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.datasets import make_blobs
import seaborn as sns
from scipy import stats
import sklearn
from scipy.stats import norm, skew 
%matplotlib inline
plt.style.use("fivethirtyeight")
from IPython.core.pylabtools import figsize
figsize(15, 12)

from sklearn.model_selection import cross_val_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import plot_confusion_matrix

#Reading the file in csv
data = pd.read_csv("House.csv")
data.head(14)
data.isnull().any()
data["Type"].value_counts().plot.pie(figsize=(10,15), fontsize = 15, autopct='%1.1f%%')
data["Bedroom"].value_counts().plot.pie(figsize=(10,15), fontsize = 15, autopct='%1.1f%%')
data["Bathroom"].value_counts().plot.pie(figsize=(10,15), fontsize = 15, autopct='%1.1f%%')
data["Sold?"].value_counts().plot.pie(figsize=(10,15), fontsize = 15, autopct='%1.1f%%')
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
data['Sold?'] = le.fit_transform(data['Sold?'])
ds_Type = data.groupby("Type").sum()
ds_Type.head()
data.head()plt.bar(ds_Type.index, "Sold?", data = ds_Type)
plt.xlabel("House Type")
plt.ylabel("House Sales")
plt.title("House Sales vs House Type")
plt.xticks(rotation = 90)
plt.show()
data['Type'] = le.fit_transform(data['Type'])
x = data.drop('Sold?', axis=1).copy()
y = data['Sold?'].copy()
model = GaussianNB()
model.fit(x, y)
pred = model.predict(x)
accur = accuracy_score(y, pred)
accur
data.head()
a = [[0,3,2]]
model.predict(a)
