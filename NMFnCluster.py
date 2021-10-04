import numpy as np
import nimfa
import pandas

def distEclud(arrA, arrB):
    return np.sqrt(np.sum(np.power(arrA - arrB, 2)))

def randCent(dataSet, k):
    n = np.shape(dataSet)[1]
    centroids = np.mat(np.zeros([k, n]))
    for j in range(n):
        minj = np.min(dataSet[:, j])
        rangej = float(np.max(dataSet[:,j])-minj)
        centroids[:, j] = np.mat(minj + rangej * np.random.rand(k, 1))
    return centroids

def KMeans(dataSet, k, distEclud= distEclud, randCent= randCent):
    m = np.shape(dataSet)[0]
    clusterAssement = np.mat(np.zeros([m, 2]))
    centroids = randCent(dataSet, k)
    clusterChanged = True
    while clusterChanged:
        clusterChanged = False
        for i in range(m):
            minDist = np.inf
            minIndex = -1
            for j in range(k):
                if minDist > distEclud(dataSet[i,:], centroids[j,:]):
                    minDist = distEclud(dataSet[i,:], centroids[j,:])
                    minIndex = j
                if clusterAssement[i, 0] != minIndex:
                    clusterChanged = True
            clusterAssement[i,:] = minIndex,minDist
        for cent in range(k):
            eveInClust = dataSet[np.nonzero(clusterAssement[:, 0] == cent)[0]]
            if len(eveInClust) != 0:
                # print(eveInClust)
                centroids[cent,:] = np.mean(eveInClust, axis=0)
    return centroids, clusterAssement
    # while clusterChanged:
    #     clusterChanged = False
    #     for i in range(m):
    #         minDist = np.inf
    #         minIndex = -1
    #         for j in range(k):
    #             distJ = distEclud(centroids[j, :], dataSet[i, :])
    #             if distJ < minDist:
    #                 minDist = distJ
    #                 minIndex = j
    #         if clusterAssement[i, 0] != minIndex:
    #             clusterChanged = True
    #         clusterAssement[i, :] = minIndex, minDist ** 2
    #     for cent in range(k):
    #         ptsInClust = dataSet[np.nonzero(clusterAssement[:, 0].A == cent)[0]]
    #         centroids[cent, :] = np.mean(ptsInClust, axis=0)
    # return centroids, clusterAssement

if __name__ == '__main__':
    date = 20111116;

    dataS = str(date);
    filename = 'H:\数据挖掘\工作日数据\%s.csv'%dataS
    data = pandas.read_csv(filename)
    V = data.values.tolist()
    V = np.array(V)
    V = V[:, 3:100]

    # print(V)
    lsnmf = nimfa.Lsnmf(V, max_iter=100, rank=3)
    lsnmf_fit = lsnmf()
    W = lsnmf_fit.basis()
    # print(W)
    k = 50
    centroids, clusterAssement = KMeans(W, k, distEclud= distEclud, randCent= randCent)
    print(centroids)
