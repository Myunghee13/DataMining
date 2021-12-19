from pyspark import SparkContext
import sys
import json
import math
from statistics import mean
import numpy as np
import time

sc = SparkContext()
sc.setSystemProperty('spark.driver.memory', '4g')
sc.setSystemProperty('spark.executor.memory', '4g')
sc.setLogLevel("OFF")

input_file = "../resource/asnlib/publicdata/train_review.json"

start_time = time.time()


def getSorted(iterable):
    return sorted(list(iterable), key=lambda x: x[1], reverse=True)


def normalizer(vector):
    avg = mean(vector)
    array = np.array(vector)
    return array - avg


def getCosineS(vector0, vector1):
    numerator = sum([a * b for a, b in zip(vector0, vector1)])
    denominator = math.sqrt(np.sum(vector0 ** 2) * np.sum(vector1 ** 2))

    return numerator / denominator if denominator != 0 else 0


# Pearson correlation of items pair with at least 3 co-rated users
# multiple (item, user) pairs -> average ratings
rdd = sc.textFile(input_file) \
    .map(lambda x: [tuple((json.loads(x)['business_id'], json.loads(x)['user_id'])), json.loads(x)["stars"]]) \
    .aggregateByKey((0, 0), lambda R, v: (R[0] + v, R[1] + 1), lambda R1, R2: (R1[0] + R2[0], R1[1] + R2[1])) \
    .map(lambda x: [x[0][0], x[0][1], x[1][0] / x[1][1]]).cache()

# user model
# user_items: [[u1, [[it1, r1], [it2, r2]]], [u2, [[it3, r3], [it4, ...
user_items = rdd.map(lambda x: [x[1], [x[0], x[2]]]) \
    .groupByKey().map(lambda x: [x[0], list(x[1])]).cache()

# with open("user_items.json", "w+") as f:
#    json.dump({"user_items": user_items.collect()}, f)

M = 6
limit = 0.2
# item index
itemList = rdd.map(lambda x: x[0]).distinct().collect()

user_itemIDs = rdd.map(lambda x: [x[1], [itemList.index(x[0]), x[2]]]) \
    .groupByKey().map(lambda x: [x[0], list(x[1])]) \
    .filter(lambda x: len(x[1]) >= M).cache()

userList = user_itemIDs.map(lambda x: x[0]).collect()
userN = user_itemIDs.count()
user_itemList = user_itemIDs.collect()


def getPearson0(uid, ratingPairs0):
    pair_list = list()
    items0 = list(zip(*ratingPairs0))[0]

    for j in range(uid + 1, userN):
        ratingPairs1 = user_itemList[j][1]
        items1 = list(zip(*ratingPairs1))[0]

        comItem = set(items1).intersection(items0)

        # at least 3 co-rated users
        if len(comItem) >= M:
            vector0 = [ratingPairs0[items0.index(item)][1] for item in comItem]
            vector1 = [ratingPairs1[items1.index(item)][1] for item in comItem]

            n_vector0 = normalizer(vector0)
            n_vector1 = normalizer(vector1)

            cosineS = getCosineS(n_vector0, n_vector1)
            if cosineS >= limit:
                pair_list.append([userList[uid], userList[j], cosineS])

    return pair_list


idx_userItems = user_itemIDs \
    .map(lambda x: getPearson0(userList.index(x[0]), x[1])) \
    .filter(lambda x: x != []) \
    .flatMap(lambda x: x) \
    .flatMap(lambda x: [[x[0], [x[1], x[2]]], [x[1], [x[0], x[2]]]]) \
    .groupByKey().map(lambda x: [x[0], getSorted(x[1])])

with open("user_model.json", "w+") as f:
    json.dump({"model": idx_userItems.collect()}, f)

print("Duration: %d" % int(time.time() - start_time))
