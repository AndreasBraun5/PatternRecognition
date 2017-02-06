import operator
import numpy
from functools import reduce

def readSamplesFromFile(fileName):
    samples = []
    with open(fileName) as f:
        lines = f.readlines()

    for line in lines:
        split = line.strip().split(' ')
        tmp = [float(x) for x in split if len(x) > 0]
        if(len(tmp) > 0):
            samples.append(tmp)

    return samples

def function():
    samples = readSamplesFromFile('ue10b_samples.txt')
    for count in range(2, 6):
        result = iterativeOptimalClustering(count, samples)
        print(result)


def iterativeOptimalClustering(countClasses, samples):
    n = len(samples)

    cStar = n
    clusters = {}

    i = 0
    for sample in samples:
        clusters[i] = [sample]
        i += 1

    while True:
        D_i = None
        D_j = None
        jIndex = -1
        jMin = numpy.inf

        for i, Di in clusters.items():
            for j, Dj in clusters.items():
                if i == j:
                    continue

                curJ = sumSquaredError(Di, Dj)
                if curJ < jMin:
                    D_i = Di
                    D_j = Dj
                    jIndex = j
                    jMin = curJ


        for elem in D_j:
            D_i.append(elem)

        clusters.pop(jIndex)
        cStar -= 1

        cnt = 0
        for i, cluster in clusters.items():
            cnt += len(cluster)
        if (cnt != n):
            print("ERROR")

        if countClasses == cStar:
            break


    return clusters

def sumSquaredError(cluster1, cluster2):
    squaredError1 = squaredError(cluster1)
    squaredError2 = squaredError(cluster2)
    return squaredError1 + squaredError2

def mean(cluster):
    total = None
    for sample in cluster:
        if total is None:
            total = sample
        else:
            total = [x + y for x, y in zip(total, sample)]
    ret = 0
    for x in total:
        ret += (x / len(cluster))
    return ret

def squaredError(cluster):
    m = mean(cluster)
    total = 0
    for sample in cluster:
        sampleMinusMean = [x - m for x in sample]
        sampleMinusMeanSquared = map(operator.mul, sampleMinusMean, sampleMinusMean)
        total += reduce(operator.add, sampleMinusMeanSquared)
    return total

if __name__ == "__main__":
    function()
