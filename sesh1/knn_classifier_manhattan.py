import numpy as np
data = [
    [150, 7.0, 1, 'Apple'],
    [120, 6.5, 0, 'Banana'],
    [180, 7.5, 2, 'Orange'],
    [155, 7.2, 1, 'Apple'],
    [110, 6.0, 0, 'Banana'],
    [190, 7.8, 2, 'Orange'],
    [145, 7.1, 1, 'Apple'],
    [115, 6.3, 0, 'Banana']
]
test_data = np.array([
    [118, 6.2, 0],  # Expected: Banana
    [160, 7.3, 1],  # Expected: Apple
    [185, 7.7, 2]   # Expected: Orange
])
rev_label={0:"Apple",1:"Banana",2:"Orange"}; label={"Apple":0,"Banana":1,"Orange":2}
X=np.array([r[:-1] for r in data], dtype=float)
y=np.array([r[-1] for r in data])
Y=np.array([label[k] for k in y])

def dist(x1,x2):
    distance=np.sum(np.abs(x2-x1))
    return distance

class KNN:
    def __init__(self,k=3):
        self.k=k
    
    def fit(self,X,Y):
        self.X_train=X
        self.Y_train=Y
    
    def predict(self,X):
        predictions=[]
        for x in X:
            predictions.append(self.predict_one(x))
        return predictions
    
    def predict_one(self,x):
        distances=[dist(x,x_train) for x_train in self.X_train]
        
        k_ind=np.argsort(distances)[:self.k]
        k_labels=[self.Y_train[i] for i in k_ind]
        
        classes,counts=np.unique(k_labels,return_counts=True)
        return classes[np.argmax(counts)]

knn=KNN()
knn.fit(X,Y)
print("Predictions:")
preds=knn.predict(test_data)
for pred in preds:
    print(rev_label[pred])

 