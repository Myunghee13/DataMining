from pyspark import SparkContext
import sys
import json

import time

sc = SparkContext()
sc.setSystemProperty('spark.driver.memory', '4g')
sc.setSystemProperty('spark.executor.memory', '4g')
sc.setLogLevel("OFF")

train_file = sys.argv[1]  # train_review.json
test_file = sys.argv[2]  # test_review.json
model_file = sys.argv[3]  # task3item.model or task3user.model
output_file = sys.argv[4]  # task3item.predict or task3user.predict
cf_type = sys.argv[5]  # item_based or user_based

start_time = time.time()


def getSorted(iterable):
    return sorted(list(iterable), key=lambda x: x[1], reverse=True)


def getKeys(pairs):
    return [pair[0] for pair in pairs]


def getPrediction(value0, ratingList):
    values = getKeys(ratingList)  # [v1, v2, v3]

    # modelList: [[v1,[[v2,w1],[v3,w2], ...] sorted by w

    weightList = None
    for weightCand in modelList:
        if weightCand[0] == value0:
            weightList = weightCand[1]
            break

    if weightList == None:
        return None

    else:
        weightVector = list()
        ratingVector = list()
        cnt = 0
        for valueWeight in weightList:
            if valueWeight[0] in values:
                key = valueWeight[0]
                weightVector.append(valueWeight[1])
                ratingVector.append(ratingList[values.index(key)][1])

                cnt += 1
                if cnt == N:
                    break

        if len(weightVector) == 0:
            return None

        else:
            denominator = 0
            numerator = 0
            for idx, weight in enumerate(weightVector):
                denominator += weight
                numerator += weight * ratingVector[idx]

            return numerator / denominator


if cf_type == "item_based":
    # key: user, value: business, r: rating (star), w: sim (weight)
    # testRDD: [k1, [v1,v2,v3..]] from test_review.json groupByKey
    testRDD = sc.textFile(test_file) \
        .map(lambda x: [json.loads(x)['user_id'], json.loads(x)['business_id']]) \
        .groupByKey().map(lambda x: [x[0], list(x[1])]) \
        .sortBy(lambda x: x[0]).cache()

    # teskeys: [k1, k2,...] sorted
    testKeys = testRDD.map(lambda x: x[0]).collect()

    # [[idx, [k1, [v1, v2,..]]],
    idxKeyValues = [[idx, keyValues] for idx, keyValues in enumerate(testRDD.collect())]
    testRDD.unpersist()

    # multiple (u, b) pairs -> average ratings
    # key: user, value: business, r: rating (star), w: sim (weight)
    # trainList: [[k1, [[v1, r1], [v2, r2]]], [k2, [[v3, r3], [v4, ...
    # -> [[[v1, r1], [v2, r2]], [[v3, r3], [v4, ...
    trainList = sc.textFile(train_file) \
        .map(lambda x: [json.loads(x)['user_id'] + json.loads(x)['business_id'], json.loads(x)["stars"]]) \
        .aggregateByKey((0, 0), lambda R, v: (R[0] + v, R[1] + 1), lambda R1, R2: (R1[0] + R2[0], R1[1] + R2[1])) \
        .map(lambda x: [x[0], x[1][0] / x[1][1]]) \
        .map(lambda x: [x[0][:22], [x[0][22:], x[1]]]) \
        .groupByKey().map(lambda x: [x[0], list(x[1])]) \
        .filter(lambda x: x[0] in testKeys) \
        .sortBy(lambda x: x[0]) \
        .map(lambda x: x[1]).collect()

    # [[v1,[[v2,w1],[v3,w2], ...] sorted by sim(weight)
    modelList = sc.textFile(model_file) \
        .map(lambda x: [json.loads(x)['b1'], json.loads(x)['b2'], json.loads(x)['sim']]) \
        .map(lambda x: [[x[0], [x[1], x[2]]], [x[1], [x[0], x[2]]]]) \
        .flatMap(lambda x: x) \
        .groupByKey().map(lambda x: [x[0], getSorted(x[1])]) \
        .collect()

    N = 3  # or 5

    # [[idx, [k1, [v1, v2,..]]]
    # -> [k1, [v1,v2..],[[v3,r1],[v4,r2]...]] from trainList, idx
    # -> [[k1, ratingList], v1] by flatMapValues
    # -> [k1, v1, [[r1, r2], [sim(v1,v2), sim(v1,v3)]]] from modelList
    # -> select N (3 or 5), r1*sim1+r2*sim2 / sim1+sim2
    testList = sc.parallelize(idxKeyValues) \
        .map(lambda x: [x[1][0], x[1][1], trainList[x[0]]]) \
        .map(lambda x: [[x[0], x[2]], x[1]]) \
        .flatMapValues(lambda x: x) \
        .map(lambda x: [x[0][0], x[1], getPrediction(x[1], x[0][1])]) \
        .filter(lambda x: x[2] != None) \
        .map(lambda x: {"user_id": x[0], "business_id": x[1], "stars": x[2]}) \
        .collect()

elif cf_type == "user_based":
    def getPrediction2(value0, ratingList):

        # avg_dic: {v1: avg1, v2: avg2, ...}
        avg0 = avg_dic[value0]  #
        values = getKeys(ratingList)  # [v1, v2, v3]

        # modelList: [[v1,[[v2,w1],[v3,w2], ...] sorted by w

        weightList = None
        for weightCand in modelList:
            if weightCand[0] == value0:
                weightList = weightCand[1]
                break

        if weightList == None:
            return None

        else:
            weightVector = list()
            ratingVector = list()
            cnt = 0
            for valueWeight in weightList:
                if valueWeight[0] in values:
                    key = valueWeight[0]
                    weightVector.append(valueWeight[1])

                    # normalized ( -avg)
                    avg1 = avg_dic[key]  #

                    ratingVector.append(ratingList[values.index(key)][1] - avg1)  #

                    cnt += 1
                    if cnt == N:
                        break

            if len(weightVector) == 0:
                return None

            else:
                denominator = 0
                numerator = 0
                for idx, weight in enumerate(weightVector):
                    denominator += weight
                    numerator += weight * ratingVector[idx]

                return avg0 + numerator / denominator  #


    # key: business, value: user, r: rating (star), w: sim (weight)
    # testRDD: [k1, [v1,v2,v3..]] from test_review.json groupByKey
    testRDD0 = sc.textFile(test_file) \
        .map(lambda x: [json.loads(x)['business_id'], json.loads(x)['user_id']]) \
        .cache()

    trainRDD0 = sc.textFile(train_file) \
        .map(lambda x: [json.loads(x)['business_id'] + json.loads(x)['user_id'], json.loads(x)["stars"]]).cache()

    # teskeys: [k1, k2,...] sorted
    testKeys = set(testRDD0.map(lambda x: x[0]).distinct().collect())

    trainKeys = trainRDD0.map(lambda x: x[0][:22]).distinct().collect()

    keys = sorted(list(testKeys.intersection(trainKeys)))

    testRDD = testRDD0 \
        .groupByKey().map(lambda x: [x[0], list(x[1])]) \
        .filter(lambda x: x in keys) \
        .sortBy(lambda x: x[0]).collect()

    # [[idx, [k1, [v1, v2,..]]],
    idxKeyValues = [[idx, keyValues] for idx, keyValues in enumerate(testRDD)]
    testRDD0.unpersist()

    # multiple (b, u) pairs -> average ratings
    # trainList: [[k1, [[v1, r1], [v2, r2]]], [k2, [[v3, r3], [v4, ...
    # -> [[[v1, r1], [v2, r2]], [[v3, r3], [v4, ...

    trainList = trainRDD0 \
        .aggregateByKey((0, 0), lambda R, v: (R[0] + v, R[1] + 1), lambda R1, R2: (R1[0] + R2[0], R1[1] + R2[1])) \
        .map(lambda x: [x[0][:22], [x[0][22:], x[1][0] / x[1][1]]]) \
        .groupByKey().map(lambda x: [x[0], list(x[1])]) \
        .filter(lambda x: x[0] in keys) \
        .sortBy(lambda x: x[0]) \
        .map(lambda x: x[1]).collect()
    trainRDD0.unpersist()

    # v1:avg1, v2: avg2, ...
    avg_file = open("../resource/asnlib/publicdata/user_avg.json", "r")
    avg_dic = json.load(avg_file)
    avg_file.close()

    # [[v1,[[v2,w1],[v3,w2], ...] sorted by sim(weight)
    modelList = sc.textFile(model_file) \
        .map(lambda x: [json.loads(x)['u1'], json.loads(x)['u2'], json.loads(x)['sim']]) \
        .map(lambda x: [[x[0], [x[1], x[2]]], [x[1], [x[0], x[2]]]]) \
        .flatMap(lambda x: x) \
        .groupByKey().map(lambda x: [x[0], getSorted(x[1])]) \
        .collect()

    N = 5  # 3 or 5

    # [[idx, [k1, [v1, v2,..]]]
    # -> [k1, [v1,v2..],[[v3,r1],[v4,r2]...]] from trainList, k1
    # -> [[k1, ratingList], v1] by flatMapValues
    # -> [k1, v1, getPrediction2(v1, ratingList)] using modelList
    # -> select N (3 or 5)
    testList = sc.parallelize(idxKeyValues) \
        .map(lambda x: [x[1][0], x[1][1], trainList[x[0]]]) \
        .map(lambda x: [[x[0], x[2]], x[1]]) \
        .flatMapValues(lambda x: x) \
        .map(lambda x: [x[0][0], x[1], getPrediction2(x[1], x[0][1])]) \
        .filter(lambda x: x[2] != None) \
        .map(lambda x: {"user_id": x[1], "business_id": x[0], "stars": x[2]}) \
        .collect()

f = open(output_file, mode="w+")
for item in testList:
    json.dump(item, f)
    f.write("\n")
f.close()

print("Duration: %d" % int(time.time() - start_time))
