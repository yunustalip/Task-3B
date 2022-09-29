import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix
import warnings
warnings.filterwarnings('ignore')
df = pd.read_csv('MRI_and_CDRinfo_Values_X_train.csv')
df2 = pd.read_csv('CDR_Values_y_train.csv')
# df=df.dropna(axis=1)
bins = np.linspace(2.5, 8.5, 100)                         #plotting CDRGLOB=3 values, mean and sigma= -3,-2,-1,1,2,3

df["CDRGLOB"] = df2["CDRGLOB"]
means = df.groupby('CDRGLOB')['HIPPOVOL'].mean()
stds = df.groupby('CDRGLOB')['HIPPOVOL'].std()
for i in list(df.groupby('CDRGLOB').groups.keys()):

    plt.hist(df[df["CDRGLOB"]==i]['HIPPOVOL'])
    plt.axvline(means[i], color='black')
    plt.text(means[i].1,0,'blah',rotation=90)
    plt.title("CDRGLOB Group: ",i)
    plt.show()

# corr_matrix = df.corr()
# plt.figure(figsize = (12,8))
# sns.clustermap(corr_matrix, annot=True, fmt = ".1f", linewidths=.01)
# plt.title("title")
# plt.show()
