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
        result = bottomUpDMeanClustering(count, samples)
        print(result)
        for i, cluster in result.items():
            xs = [elem[0] for elem in cluster]
            ys = [elem[1] for elem in cluster]
            plot.scatter(xs, ys)
        #plot.show()
        plot.title('bottomupdmean' + str(count))
        plot.savefig('bottomupdmean' + str(count) + '.png')


def bottomUpDMeanClustering(countClasses, samples):
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

                curJ = distance(i, j, clusters)
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

def distance(clusterI, clusterJ, clusters):
    cluster1 = clusters[clusterI]
    cluster2 = clusters[clusterJ]
    return abs([x - y for x, y in zip(mean(cluster1), mean(cluster2))])

def mean(cluster):
    total = None
    for sample in cluster:
        if total is None:
            total = sample
        else:
            total = [x + y for x, y in zip(total, sample)]
    return [x / len(cluster) for x in total]

def abs(vector):
    elemSquared = [x * x for x in vector]
    return numpy.sqrt(reduce(operator.add, elemSquared))

if __name__ == "__main__":
    function()
