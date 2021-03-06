import operator
import numpy
import matplotlib.cm as cm
import matplotlib.pyplot as plot

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
        for i, cluster in result.items():
            xs = [elem[0] for elem in cluster]
            ys = [elem[1] for elem in cluster]
            plot.scatter(xs, ys)
        plot.title('iterative' + str(count))
        plot.savefig('iterative' + str(count) + '.png')


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

                curJ = sumSquaredError(i, j, clusters)
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

def sumSquaredError(clusterI, clusterJ, clusters):
    total = 0

    for i, cluster in clusters.items():
        if i == clusterI or i == clusterJ:
            continue
        else:
            total += squaredError(cluster)

    tmpClust = list(clusters[clusterI])
    tmpClust.extend(list(clusters[clusterJ]))
    total += squaredError(tmpClust)

    return total

def mean(cluster):
    total = None
    for sample in cluster:
        if total is None:
            total = sample
        else:
            total = [x + y for x, y in zip(total, sample)]
    return [x / len(cluster) for x in total]

def squaredError(cluster):
    meanvec = mean(cluster)
    total = 0
    for sample in cluster:
        sampleMinusMean = [s - m for s, m in zip(sample, meanvec)]
        elemSquared = [x * x for x in sampleMinusMean]
        total += numpy.sqrt(reduce(operator.add, elemSquared)) **2
    return total

if __name__ == "__main__":
    function()
