import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

data=pd.read_csv(r"D:\Code\Datasets\datasets-20240624T101648Z-001\datasets\clustering_data.csv")
data=pd.DataFrame(data)
fdata = data[data['StateName'] == 'MADHYA PRADESH']
fdata=fdata.dropna()
fdata=fdata.drop_duplicates()
fdata['Latitude'] = pd.to_numeric(fdata['Latitude'].str.strip().str.rstrip('-'), errors='coerce')
fdata['Longitude'] = pd.to_numeric(fdata['Longitude'].str.strip().str.rstrip('-'), errors='coerce')
fdata=fdata[fdata["Latitude"].astype(float).between(-90,90)]
fdata=fdata[fdata["Longitude"].astype(float).between(-180,180)]

#data filtering
def Task1_1(data):
    print(data["Pincode"])

#visualization
def Task1_2(data): 
    x=data["Latitude"].to_numpy().astype(float)
    y=data["Longitude"].to_numpy().astype(float)
    plt.scatter(x,y)
    plt.xlabel("Latitude")
    plt.ylabel("Longitude")
    plt.show()

#k-means clustering
class kMeans:
    def __init__(self,k=3,iter=500) -> None:
        self.k=k
        self.centres=None
        self.max_iter=iter

    def predict(self,test):
        test = np.array([test[:,0].astype(float), test[:,1].astype(float)]).T
        distances = np.sqrt(np.sum((test[:, np.newaxis] - self.centres) ** 2, axis=-1))
        return np.argmin(distances, axis=1)+1


    def fit(self,input):
        input=np.array([input["Latitude"].astype(float),input["Longitude"].astype(float)])
        rows,columns=input.shape
        indices=np.random.choice(rows,size=self.k)
        self.centres=np.array(input.T[indices])
        y=[]
        for _ in range(self.max_iter):
            y=[]
            for data in input.T:
                dist=np.sqrt(np.sum((self.centres-data)**2,axis=1))
                y.append(np.argmin(dist))
            y=np.array(y)
            clu=[]
            for i in range(self.k):
                clu.append(np.argwhere(y==i).reshape(-1))
            
            cent=[]
            for i, j in enumerate(clu):
                if len(j)==0:
                    cent.append(self.centres[i])
                else:
                    cent.append(np.mean(input.T[j],axis=0))
            #print(np.sqrt(np.sum((self.centres-cent)**2,axis=1)))
            if np.all(np.sqrt(np.sum((self.centres-cent)**2,axis=1))<0.00001):
                break
            else:
                self.centres=np.array(cent)
                #print(np.sqrt(np.sum((self.centres-cent)**2,axis=1)))
        return y

#clustering implementation and visualization
clsa=kMeans()
clusters=clsa.fit(fdata)
test=np.array([[23.444,44.666],[12.44,3.999]])
result=clsa.predict(test)


for i in range(clsa.k):
    plt.scatter(fdata["Latitude"].astype(float)[clusters == i], fdata["Longitude"].astype(float)[clusters == i], label=f'Area Cluster {i+1}')
plt.xlabel("LATITUDE")
plt.ylabel("LONGITUDE")
plt.scatter(test[:,0],test[:,1],c=result,marker="*",s=200)
plt.legend(title="CLUSTERS")
plt.show()