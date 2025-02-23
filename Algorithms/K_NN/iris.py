import numpy as np
import pandas
import pandas as pd

data = pd.read_csv('IRIS.csv').sample(frac=1)

trainSet = data.iloc[:len(data) * 2 // 3]
testSet = data.iloc[len(data) * 2 // 3 + 1:]

dictFlower = {
    'Iris-setosa': 0,
    'Iris-versicolor': 1,
    'Iris-virginica': 2
}

def KNN(K, dataPoint):
    train = trainSet.drop(columns=['species'])
    X2 = np.sum(train * train, 1)
    Y2 = np.sum(dataPoint * dataPoint)

    dist = X2 + Y2 - 2 * train.dot(dataPoint)

    return trainSet.loc[dist.nsmallest(K).index, 'species'].mode()[0]




def testKNN(K):
    test = testSet.drop(columns=['species'])
    # print(testSet)
    dem = 0
    for index, row in test.iterrows():

        a = KNN(K, row)
        b = testSet.loc[index, 'species']
        if a == b:
            dem += 1
    return dem * 100 / len(testSet)

print(testKNN(5))
print(testKNN(7))
print(testKNN(9))
print(testKNN(11))
print(testKNN(3))



