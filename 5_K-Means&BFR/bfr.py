from pyspark import SparkContext
import sys
import glob
import random
import json

import time

sc = SparkContext()
sc.setSystemProperty('spark.driver.memory', '4g')
sc.setSystemProperty('spark.executor.memory', '4g')
sc.setLogLevel("OFF")

input_path = sys.argv[1]  # folder: test1/, 1-5
n_cluster = int(sys.argv[2])  # 10/10/5/8/15
out_file1 = sys.argv[3]  # cluster_res1.json
out_file2 = sys.argv[4]  # intermediate1.csv

start_time = time.time()  # 300s


def getSquaredD(vector0, vector1):
    return sum([(c0 - c1) ** 2 for c0, c1 in zip(vector0, vector1)])


# K-means ++ initialize
def getMinDistance(vector):
    return min([getSquaredD(centroid, vector) for centroid in centroids])


def getCluster(vector):
    compareList = [getSquaredD(vector, c) for c in centroids]
    return compareList.index(min(compareList))


def getCenter(points, num_points):
    matrix = [point[1] for point in points]
    return [sum(c) / num_points for c in zip(*matrix)]


# [[index0, vector0], [i1,v1],...
def getSumandSQ(IVlist):
    Sum = list()
    SumSQ = list()

    pointIndexes = list()
    matrix = list()
    for i, v in IVlist:
        pointIndexes.append(i)
        matrix.append(v)

    for c in zip(*matrix):
        Sum.append(sum(c))
        SumSQ.append(sum(ci ** 2 for ci in c))
    return [Sum, SumSQ, pointIndexes]


def getCentroids(N, Sum):
    return [point / N for point in Sum]


def getVariance(N, Sum, SumSQ):
    return [sq / N - (Sum[i] / N) ** 2 for i, sq in enumerate(SumSQ)]


def determineDSCS(vector):
    indicator = True
    for idx, cent in enumerate(DScentroids):
        MD = sum([(pi - cent[i]) ** 2 / DSvariance[idx][i] for i, pi in enumerate(vector)])

        if MD < a2d:
            indicator = False
            return [idx, vector]
            break

    if indicator:
        for idx, cent in enumerate(CScentroids):
            MD = sum([(pi - cent[i]) ** 2 / CSvariance[idx][i] for i, pi in enumerate(vector)])

            if MD < a2d:
                indicator = False
                return [idx + lenDS, vector]
                break

    if indicator:
        return [-1, vector]


def getVectorSum(vector0, vector1):
    return [sum(c) for c in zip(vector0, vector1)]


def getMatrixSum(matrix):
    return [sum(c) for c in zip(*matrix)]


# CSCS merge
def detectMerge(vector):
    mergeList = list()
    for i, cs in enumerate(CScentroids):
        distance = getSquaredD(vector, cs)
        if distance < a2d4:
            mergeList.append([i + lenDS, distance])
    if len(mergeList) == 0:
        return []
    else:
        if len(mergeList) >= 2:
            minD = mergeList[0][1]
            index = mergeList[0][0]
            for candidate in mergeList:
                if candidate[1] < minD:
                    minD = candidate[1]
                    index = candidate[0]
            return index
        else:
            return mergeList[0][0]


# DSCS merge
def detectMerge1(vector):
    mergeList = list()
    for i, ds in enumerate(DScentroids):
        distance = getSquaredD(vector, ds)
        if distance < a2d4:
            mergeList.append([i, distance])
    if len(mergeList) == 0:
        return []
    else:
        if len(mergeList) >= 2:
            minD = mergeList[0][1]
            index = mergeList[0][0]
            for candidate in mergeList:
                if candidate[1] < minD:
                    minD = candidate[1]
                    index = candidate[0]
            return index
        else:
            return mergeList[0][0]


intermediate = list()
for round_id, filename in enumerate(glob.glob(input_path + "*.txt")):
    if round_id == 0:
        # initialize
        # x: point(x[0]: point index, x[1:]: vector)
        # x[0]: index, x[1]: vector
        dataRDD = sc.textFile(filename) \
            .map(lambda x: [float(point) for point in x.strip("\n").split(',')]) \
            .map(lambda x: [str(int(x[0])), x[1:]]) \
            .cache()

        num_sample = int(dataRDD.count() / 2)
        random.seed(5)
        sampleList = random.sample(dataRDD.collect(), num_sample)

        random.seed(5)
        centroids = [random.choice(sampleList)[1]]

        # d: dimension
        d = len(centroids[0])
        a = 2  # hyperparameter: 2,3,4
        a2d = a ** 2 * d  # if md(Mahalanobis D)**2 < a2d
        # 4a**2*d: btw centroids
        a2d4 = 4 * a2d
        kn = 2 * n_cluster  # 2~5

        sampleRDD = sc.parallelize(sampleList).cache()
        # get initial kn centroids by using K-means++
        for i in range(1, kn):
            maxDistance = sampleRDD \
                .map(lambda x: [x[1], getMinDistance(x[1])]) \
                .sortBy(lambda x: x[1], False) \
                .take(1)

            centroids.append(maxDistance[0][0])

        cnt = 0
        diff = 100
        while (diff > 20 and cnt < 40):
            cnt += 1
            # clusterRDD x: [0, [[1, [1,2,3]], [2, [4,5,6]]]]
            clusterRDD = sampleRDD \
                .map(lambda x: [getCluster(x[1]), x]) \
                .groupByKey() \
                .map(lambda x: [x[0], list(x[1])]) \
                .cache()

            newCenRDD = clusterRDD \
                .map(lambda x: [x[0], getCenter(x[1], len(x[1]))]) \
                .sortBy(lambda x: x[0]).cache()

            diff = newCenRDD \
                .map(lambda x: [1, getSquaredD(x[1], centroids[x[0]])]) \
                .reduceByKey(lambda a, b: a + b) \
                .map(lambda x: x[1]).take(1)[0]

            # new centroids
            centroids = newCenRDD.map(lambda x: x[1]).collect()

        # clusterRDD x: [0, [[1, [1,2,3]], [2, [4,5,6]]]]
        # remove RS from sample
        sampleRDD = clusterRDD.filter(lambda x: len(x[1]) != 1) \
            .flatMap(lambda x: x[1]).cache()

        random.seed(5)
        centroids = [random.choice(sampleRDD.collect())[1]]

        # get initial k centroids by using K-means++
        for i in range(1, n_cluster):
            maxDistance = sampleRDD \
                .map(lambda x: [x[1], getMinDistance(x[1])]) \
                .sortBy(lambda x: x[1], False) \
                .take(1)

            centroids.append(maxDistance[0][0])

        cnt = 0
        diff = 100
        while (diff > 4 and cnt < 40):
            cnt += 1
            # clusterRDD x: [0, [[1, [1,2,3]], [2, [4,5,6]]]]
            clusterRDD = sampleRDD \
                .map(lambda x: [getCluster(x[1]), x]) \
                .groupByKey() \
                .map(lambda x: [x[0], list(x[1])]) \
                .cache()

            newCenRDD = clusterRDD \
                .map(lambda x: [x[0], getCenter(x[1], len(x[1]))]) \
                .sortBy(lambda x: x[0]).cache()

            diff = newCenRDD \
                .map(lambda x: [1, getSquaredD(x[1], centroids[x[0]])]) \
                .reduceByKey(lambda a, b: a + b) \
                .map(lambda x: x[1]).take(1)[0]

            # new centroids
            centroids = newCenRDD.map(lambda x: x[1]).collect()

        # DS
        # getSumandSQ: [Sum, SumSQ, pointIndexes]
        # [clusterNum, N, [Sum, SumSQ, pointIndexes]]
        DSrdd = clusterRDD \
            .map(lambda x: [x[0], len(x[1]), getSumandSQ(x[1])]) \
            .sortBy(lambda x: x[0]).cache()

        # [cln, [N, Sum, SumSQ]]
        DS = DSrdd.map(lambda x: [x[0], [x[1], x[2][0], x[2][1]]]) \
            .collect()
        lenDS = DSrdd.count()
        # [cln, pointIndexes]
        DSList = DSrdd.map(lambda x: [x[0], x[2][2]]) \
            .collect()

        DScentroids = DSrdd.map(lambda x: getCentroids(x[1], x[2][0])).collect()
        DSvariance = DSrdd.map(lambda x: getVariance(x[1], x[2][0], x[2][1])).collect()
        num_DS = DSrdd \
            .map(lambda x: [1, x[1]]) \
            .reduceByKey(lambda a, b: a + b) \
            .map(lambda x: x[1]).take(1)[0]

        DSpoints = DSrdd.flatMap(lambda x: x[2][2]) \
            .collect()

        # CS and RS: kn with remain points
        dataRDD = dataRDD \
            .filter(lambda x: x[0] not in DSpoints) \
            .cache()

        centroids = [random.choice(dataRDD.collect())[1]]

        for i in range(1, kn):
            maxDistance = dataRDD \
                .map(lambda x: [x[1], getMinDistance(x[1])]) \
                .sortBy(lambda x: x[1], False) \
                .take(1)

            centroids.append(maxDistance[0][0])

        cnt = 0
        diff = 100
        while (diff > 4 and cnt < 40):
            cnt += 1
            # clusterRDD x: [0, [[1, [1,2,3]], [2, [4,5,6]]]]]
            clusterRDD = dataRDD \
                .map(lambda x: [getCluster(x[1]), x]) \
                .groupByKey() \
                .map(lambda x: [x[0], list(x[1])]) \
                .cache()

            newCenRDD = clusterRDD \
                .map(lambda x: [x[0], getCenter(x[1], len(x[1]))]) \
                .sortBy(lambda x: x[0]).cache()

            diff = newCenRDD \
                .map(lambda x: [1, getSquaredD(x[1], centroids[x[0]])]) \
                .reduceByKey(lambda a, b: a + b) \
                .map(lambda x: x[1]).take(1)[0]

            # new centroids
            centroids = newCenRDD.map(lambda x: x[1]).collect()

        RSrdd = clusterRDD.filter(lambda x: len(x[1]) == 1) \
            .map(lambda x: [x[1][0][0], x[1][0][1]]).cache()
        lenRS = RSrdd.count()
        RS = RSrdd.collect()

        # getSumandSQ: [Sum, SumSQ, pointIndexes]
        # [N, [Sum, SumSQ, pointIndexes]]
        CSrdd = clusterRDD.filter(lambda x: len(x[1]) != 1) \
            .map(lambda x: [len(x[1]), getSumandSQ(x[1])]) \
            .cache()

        CScentroids = CSrdd.map(lambda x: getCentroids(x[0], x[1][0])).collect()
        CSvariance = CSrdd.map(lambda x: getVariance(x[0], x[1][0], x[1][1])).collect()
        num_CS = CSrdd \
            .map(lambda x: [1, x[0]]) \
            .reduceByKey(lambda a, b: a + b) \
            .map(lambda x: x[1]).take(1)[0]

        lenCS = CSrdd.count()

        # [N, Sum, SumSQ]
        CS = CSrdd \
            .map(lambda x: [x[0], x[1][0], x[1][1]]).collect()
        CSList = CSrdd \
            .map(lambda x: x[1][2]).collect()

        # CS re-numbering
        # CS [[10, [N, Sum, SumSQ]],
        # CSList [[10, pointIndexes],
        CS = [[idx + lenDS, cs] for idx, cs in enumerate(CS)]
        CSList = [[idx + lenDS, cs] for idx, cs in enumerate(CSList)]

        DSCS = DS + CS
        DSCSList = DSList + CSList

        intermediate.append([round_id + 1, lenDS, num_DS, lenCS, num_CS, lenRS])

        del sampleList

        # if md(Mahalanobis D**2) < a2d
        # sum((pi-ci)**2/variance of i)
    else:
        # [index, [clusterNum, vector]]
        dataRDD = sc.textFile(filename) \
            .map(lambda x: [float(point) for point in x.strip("\n").split(',')]) \
            .map(lambda x: [str(int(x[0])), determineDSCS(x[1:])]).cache()

        addRS = dataRDD.filter(lambda x: x[1][0] == -1) \
            .map(lambda x: [x[0], x[1][1]]).collect()

        RS += addRS
        lenRS = len(RS)
        # [index, [clusterNum, vector]]
        # -> [clusterNum, [index, vector]]
        # -> [clusterNum, N, [Sum, SumSQ, pointIndexes]]
        DSCSrdd = dataRDD.filter(lambda x: x[1][0] != -1) \
            .map(lambda x: [x[1][0], [x[0], x[1][1]]]) \
            .groupByKey() \
            .map(lambda x: [x[0], len(x[1]), getSumandSQ(x[1])]) \
            .cache()
        # [[clusterNum, [N, Sum, SumSQ]]
        addDSCS = DSCSrdd.map(lambda x: [x[0], [x[1], x[2][0], x[2][1]]]) \
            .collect()

        # update DSCS
        for dscs in addDSCS:
            clNum = dscs[0]
            dscs0 = DSCS[clNum]
            N = dscs[1][0] + dscs0[1][0]
            Sum = getVectorSum(dscs[1][1], dscs0[1][1])
            SumSQ = getVectorSum(dscs[1][2], dscs0[1][2])
            DSCS[clNum] = [clNum, [N, Sum, SumSQ]]

        addDSCSList = DSCSrdd.map(lambda x: [x[0], x[2][2]]) \
            .collect()

        # update DSCSList
        for points in addDSCSList:
            clNum = points[0]
            dscs0 = DSCSList[clNum]
            DSCSList[clNum][1] = points[1] + dscs0[1]

        DS = DSCS[:n_cluster]
        CS = DSCS[n_cluster:]
        lenCS = len(CS)

        DSrdd = sc.parallelize(DS).cache()
        lenDS = DSrdd.count()
        DScentroids = DSrdd.map(lambda x: getCentroids(x[1][0], x[1][1])).collect()
        DSvariance = DSrdd.map(lambda x: getVariance(x[1][0], x[1][1], x[1][2])).collect()
        num_DS = DSrdd \
            .map(lambda x: [1, x[1][0]]) \
            .reduceByKey(lambda a, b: a + b) \
            .map(lambda x: x[1]).take(1)[0]

        if lenRS <= kn:  # no clustering
            CSrdd = sc.parallelize(CS).cache()
            CScentroids = CSrdd.map(lambda x: getCentroids(x[1][0], x[1][1])).collect()
            CSvariance = CSrdd.map(lambda x: getVariance(x[1][0], x[1][1], x[1][2])).collect()
            num_CS = CSrdd \
                .map(lambda x: [1, x[1][0]]) \
                .reduceByKey(lambda a, b: a + b) \
                .map(lambda x: x[1]).take(1)[0]

            intermediate.append([round_id + 1, lenDS, num_DS, lenCS, num_CS, lenRS])

        # if len(RS) > kn, kn means for RS to generate new CS
        # update RS
        # merge old and new CS
        else:
            dataRDD = sc.parallelize(RS).cache()

            random.seed(5)
            centroids = [random.choice(dataRDD.collect())[1]]

            for i in range(1, kn):
                maxDistance = dataRDD \
                    .map(lambda x: [x[1], getMinDistance(x[1])]) \
                    .sortBy(lambda x: x[1], False) \
                    .take(1)

                centroids.append(maxDistance[0][0])

            cnt = 0
            diff = 100
            while (diff > 4 and cnt < 40):
                cnt += 1
                # clusterRDD x: [0, [[1, [1,2,3]], [2, [4,5,6]]]]]
                clusterRDD = dataRDD \
                    .map(lambda x: [getCluster(x[1]), x]) \
                    .groupByKey() \
                    .map(lambda x: [x[0], list(x[1])]) \
                    .cache()

                newCenRDD = clusterRDD \
                    .map(lambda x: [x[0], getCenter(x[1], len(x[1]))]) \
                    .sortBy(lambda x: x[0]).cache()

                diff = newCenRDD \
                    .map(lambda x: [1, getSquaredD(x[1], centroids[x[0]])]) \
                    .reduceByKey(lambda a, b: a + b) \
                    .map(lambda x: x[1]).take(1)[0]

                # new centroids
                centroids = newCenRDD.map(lambda x: x[1]).collect()

            # clusterRDD x: [0, [[1, [1,2,3]], [2, [4,5,6]]]]
            RS = clusterRDD.filter(lambda x: len(x[1]) == 1) \
                .map(lambda x: [x[1][0][0], x[1][0][1]]).collect()

            # [N, Sum, SumSQ, pointIndexes]
            addCS = clusterRDD.filter(lambda x: len(x[1]) != 1) \
                .map(lambda x: [len(x[1]), getSumandSQ(x[1])]) \
                .map(lambda x: [x[0], x[1][0], x[1][1], x[1][2]]) \
                .collect()

            # renumbering
            # [clusterNum, [N, Sum, SumSQ, pointIndexes]]
            addCS = [[i, cs] for i, cs in enumerate(addCS)]

            addCSrdd = sc.parallelize(addCS) \
                .cache()

            # find pair of old and new CS
            # [clusterNum, [N, Sum, SumSQ, pointIndexes]
            CSCSpair = addCSrdd \
                .map(lambda x: [x[0], getCentroids(x[1][0], x[1][1])]) \
                .map(lambda x: [x[0], detectMerge(x[1])]) \
                .filter(lambda x: x[1] != []) \
                .map(lambda x: [x[1], x[0]]) \
                .groupByKey() \
                .map(lambda x: [x[0], list(x[1])]) \
                .collect()

            if CSCSpair == []:  # no merge
                # [N, Sum, SumSQ]
                addCS0 = addCSrdd \
                    .map(lambda x: [x[0] + lenDS + lenCS, [x[1][0], x[1][1], x[1][2]]]) \
                    .collect()
                # point indexes
                addCSList0 = addCSrdd \
                    .map(lambda x: [x[0] + lenDS + lenCS, x[1][3]]).collect()

            else:
                # merge CS and CS
                merger = list()
                for pair in CSCSpair:
                    p1 = pair[1]
                    N = 0
                    sumList = list()
                    sumSQlist = list()
                    indexes = list()
                    for i in p1:
                        N += addCS[i][1][0]
                        sumList.append(addCS[i][1][1])
                        sumSQlist.append(addCS[i][1][2])
                        indexes += addCS[i][1][3]

                    merger.append([N, getMatrixSum(sumList), getMatrixSum(sumSQlist), indexes])
                # update DSCS, DSCSList by merging newCS
                for i, pair in enumerate(CSCSpair):
                    clNum = pair[0]
                    dscs = DSCS[clNum]
                    N = dscs[1][0] + merger[i][0]
                    Sum = getVectorSum(dscs[1][1], merger[i][1])
                    SumSQ = getVectorSum(dscs[1][2], merger[i][2])
                    DSCS[clNum] = [clNum, [N, Sum, SumSQ]]
                    DSCSList[clNum][1] += merger[i][3]

                mergeNum = sc.parallelize(CSCSpair) \
                    .map(lambda x: x[1]) \
                    .flatMap(lambda x: x).collect()

                addCS = addCSrdd \
                    .filter(lambda x: x[0] not in mergeNum) \
                    .map(lambda x: x[1]).collect()

                # renumbering
                # [clusterNum, [N, Sum, SumSQ, pointIndexes]]
                addCS = [[i + lenDS + lenCS, cs] for i, cs in enumerate(addCS)]

                addCSrdd = sc.parallelize(addCS) \
                    .cache()

                # [N, Sum, SumSQ]
                addCS0 = addCSrdd \
                    .map(lambda x: [x[0], [x[1][0], x[1][1], x[1][2]]]) \
                    .collect()
                # point indexes
                addCSList0 = addCSrdd \
                    .map(lambda x: [x[0], x[1][3]]).collect()

            DSCS += addCS0
            DSCSList += addCSList0

            CS = DSCS[n_cluster:]

            CSrdd = sc.parallelize(CS).cache()
            CScentroids = CSrdd.map(lambda x: getCentroids(x[1][0], x[1][1])).collect()
            CSvariance = CSrdd.map(lambda x: getVariance(x[1][0], x[1][1], x[1][2])).collect()
            lenCS = CSrdd.count()
            num_CS = CSrdd \
                .map(lambda x: [1, x[1][0]]) \
                .reduceByKey(lambda a, b: a + b) \
                .map(lambda x: x[1]).take(1)[0]

            intermediate.append([round_id + 1, lenDS, num_DS, lenCS, num_CS, len(RS)])

# merge DS CS
DSCSpair = sc.parallelize(DSCS[n_cluster:]) \
    .map(lambda x: [x[0], getCentroids(x[1][0], x[1][1])]) \
    .map(lambda x: [x[0], detectMerge1(x[1])]) \
    .filter(lambda x: x[1] != []) \
    .map(lambda x: [x[1], x[0]]) \
    .groupByKey() \
    .map(lambda x: [x[0], list(x[1])]) \
    .collect()

for pair in DSCSpair:
    ds = pair[0]
    cslist = pair[1]
    for cs in cslist:
        DSCSList[ds][1] += DSCSList[cs][1]

DSList = DSCSList[:n_cluster]
num_DS = sc.parallelize(DSList) \
    .flatMap(lambda x: x[1]) \
    .count()

mergeNum = sc.parallelize(DSCSpair) \
    .flatMap(lambda x: x[1]).collect()

newCSrdd = sc.parallelize(DSCSList[n_cluster:]) \
    .filter(lambda x: x[0] not in mergeNum) \
    .cache()

lenCS = newCSrdd.count()
csrdd = newCSrdd.flatMap(lambda x: x[1]).cache()
num_CS = csrdd.count()
CSList = csrdd.collect()

# revise the last intermediate
intermediate[-1] = [round_id + 1, lenDS, num_DS, lenCS, num_CS, len(RS)]

f1 = open(out_file1, mode="w+")
result = dict()
for clusterList in DSList:
    clnum = clusterList[0]
    points = clusterList[1]
    for point in points:
        result.update({point: clnum})
for point in CSList:
    result.update({point: -1})

for point in RS:
    result.update({point[0]: -1})
json.dump(result, f1)

f1.close()

f2 = open(out_file2, mode="w+")

f2.write(
    "round_id,nof_cluster_discard,nof_point_discard,nof_cluster_compression,nof_point_compression,nof_point_retained\n")
for item in intermediate:
    f2.write(str(item).strip("[]"))
    f2.write("\n")

f2.close()

print("Duration: %d" % int(time.time() - start_time))
