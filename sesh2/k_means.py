import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
df=pd.read_csv("clustering_data.csv")
df = df[df["StateName"] == "KARNATAKA"]
df["Latitude"] = pd.to_numeric(df["Latitude"], errors="coerce")
df["Longitude"] = pd.to_numeric(df["Longitude"], errors="coerce")

df = df.drop_duplicates()
df=df[(df["Latitude"]>=10)&(df["Latitude"]<=20)&(df["Longitude"]>=70)&(df["Longitude"]<=80)]
#print(df.columns)
lat=np.array(df["Latitude"])
lon=np.array(df["Longitude"])

plt.figure(figsize=(8,8))
plt.scatter(lon, lat, s=5)

plt.xlabel("Longitude")
plt.ylabel("Latitude")
plt.title("Karnataka Pincode Locations")

plt.show()

def euclid(x1,x2):
        distance=np.sqrt(np.sum((x2-x1)**2))
        return distance

X=np.column_stack((lat,lon))

class kmeans:
    def __init__(self,k=10,iter=50,plot_steps=False):
        self.k=k
        self.iter=iter
        self.plot_steps=plot_steps

        self.clusters=[[] for _ in range(self.k)] 
        self.centroids=[]
    
    def predict(self,X):
        self.X=X
        self.n_samples,self.n_feats=X.shape

        random_sample_idx=np.random.choice(self.n_samples,self.k,replace=False)
        self.centroids=[self.X[idx] for idx in random_sample_idx]

        for t in range(self.iter):
            self.clusters=self.create_cluster(self.centroids)

            if self.plot_steps:
                self.plot()

            centroids_old=self.centroids
            self.centroids=self.get_centroids(self.clusters)

            if self.is_converged(centroids_old,self.centroids):
                print(f"{t} iterations completed.")
                break
            
            if self.plot_steps:
                self.plot()
        return self.get_cluster_labels(self.clusters)

    def get_cluster_labels(self,clusters):
        labels=np.empty(self.n_samples)

        for cluster_idx,cluster in enumerate(clusters):
            labels[cluster]=cluster_idx
        return labels

    def create_cluster(self,centroids):
        clusters=[[] for _ in range(self.k)] 
        for idx,sample in enumerate(self.X):
            centroid_idx=self.closest_centroid(sample,centroids)
            clusters[centroid_idx].append(idx)
        return clusters

    def closest_centroid(self,sample,centroids):
        d=np.empty(self.k)
        d=[euclid(sample,point) for point in centroids]
        return np.argmin(d)
    
    def get_centroids(self,clusters):
        centroids=np.zeros((self.k,self.n_feats))
        for cluster_idx,cluster in enumerate(clusters):
            cluster_mean=np.mean(self.X[cluster],axis=0)
            centroids[cluster_idx]=cluster_mean
        return centroids

    def is_converged(self,centroids_old,centroids):
        distances=[euclid(centroids_old[i],centroids[i]) for i in range(self.k)]
        return sum(distances)<0.0001

    def plot(self):

        fig,ax=plt.subplots(figsize=(8,8))

        for i,index in enumerate(self.clusters):
            point=self.X[index]
            ax.scatter(point[:, 1], point[:, 0], s=10)

        for point in self.centroids:
            ax.scatter(point[1], point[0], marker="x", s=100,c="black")

        plt.xlabel("Longitude")
        plt.ylabel("Latitude")
        plt.title("KMeans Clustering")

        plt.show()

decision=input("Do you want to use custom k? (y/n):")
if str.lower(decision)=='y':
    decision=int(input("Enter k value:"))
elif str.lower(decision)=='n':
    print("Taking k=5 by default...")
    decision=int(5)
else:
    print("INVALID")
    exit()
    

model = kmeans(k=decision)

labels = model.predict(X)

model.plot()
