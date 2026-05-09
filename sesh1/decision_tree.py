import numpy as np
data = [
    [12.0, 1.5, 1, 'Wine'],
    [5.0, 2.0, 0, 'Beer'],
    [40.0, 0.0, 1, 'Whiskey'],
    [13.5, 1.2, 1, 'Wine'],
    [4.5, 1.8, 0, 'Beer'],
    [38.0, 0.1, 1, 'Whiskey'],
    [11.5, 1.7, 1, 'Wine'],
    [5.5, 2.3, 0, 'Beer']
]
reverse_map = {
    0: "Wine",
    1: "Beer",
    2: "Whiskey"
}
X = np.array([row[:-1] for row in data], dtype=float)
y = np.array([row[-1] for row in data])
Y=np.zeros(y.shape,dtype=int)
print(X)
for i in range(len(y)):
    if y[i]=="Wine": Y[i]=int(0)
    elif y[i]=="Beer": Y[i]=int(1)
    else: Y[i]=int(2)

print(Y)

def gini_check(labels):
    if len(labels)==0: 
        return 0
    classes, count=np.unique(labels,return_counts=True)
    probs=count/len(labels)
    gini=1-np.sum(probs**2)
    return gini

def best(X,Y):
    samples,features=X.shape
    bestgini=float(2)
    bestfeature=None
    bestthreshold=None
    for feature in range(features):
        thresholds=[]
        values=np.unique(X[:,feature])
        for i in range(len(values)-1):
            midpoint=(values[i]+values[i+1])/2
            thresholds.append(midpoint)
        for threshold in thresholds:
            left=X[:,feature]<threshold
            right=X[:,feature]>=threshold
            lefty=Y[left]
            righty=Y[right]
            if len(righty)==0 or len(lefty)==0:
                continue
            left_gini=gini_check(lefty)
            right_gini=gini_check(righty)
            weighted=((len(lefty)/samples)*left_gini)+((len(righty)/samples)*right_gini)
            if weighted<bestgini:
                bestgini=weighted
                bestfeature=feature
                bestthreshold=threshold
    return bestfeature,bestthreshold,bestgini

test_data = np.array([
    [6.0, 2.1, 0],   # Expected: Beer
    [39.0, 0.05, 1], # Expected: Whiskey
    [13.0, 1.3, 1]   # Expected: Wine
])

class Node:
    def __init__(
        self,
        feature_index=None,
        threshold=None,
        left=None,
        right=None,
        value=None
    ):
        self.feature_index = feature_index
        self.threshold = threshold
        self.left = left
        self.right = right
        self.value = value

class DecisionTree:
    def __init__(self, max_depth=3):
        self.max_depth = max_depth
        self.root = None
    def fit(self, X, Y):
        self.root = self.build_tree(X, Y, depth=0)
    def build_tree(self, X, y, depth):
        if len(np.unique(y))==1:
            return Node(value=y[0])
        if depth>=self.max_depth:
            return Node(value=self.majority_class(y))
        feature, threshold, gini = best(X, y)
        if feature is None:
            return Node(value=self.majority_class(y))
        left_indices = X[:, feature] < threshold
        right_indices = X[:, feature] >= threshold

        left_X = X[left_indices]
        left_y = y[left_indices]

        right_X = X[right_indices]
        right_y = y[right_indices]
        left_child = self.build_tree(
            left_X,
            left_y,
            depth + 1
        )

        right_child = self.build_tree(
            right_X,
            right_y,
            depth + 1
        )

        return Node(
            feature_index=feature,
            threshold=threshold,
            left=left_child,
            right=right_child
        )
    
    def majority_class(self, y):
        classes, counts = np.unique(y, return_counts=True)
        return classes[np.argmax(counts)]
    
    def predict(self, X):
        predictions = []
        for x in X:
            predictions.append(
                self.predict_one(x, self.root)
            )
        return np.array(predictions)
    
    def predict_one(self, x, node):
        if node.value is not None:
            return node.value
        if x[node.feature_index] < node.threshold:
            return self.predict_one(x, node.left)
        else:
            return self.predict_one(x, node.right)

tree = DecisionTree()

tree.fit(X, Y)
predictions = tree.predict(test_data)

print("Predictions:")

for pred in predictions:

    print(reverse_map[pred])