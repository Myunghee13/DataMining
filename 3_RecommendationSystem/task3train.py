
from pyspark import SparkContext
import sys
import json
import math

import time

sc = SparkContext()
sc.setSystemProperty('spark.driver.memory', '4g')
sc.setSystemProperty('spark.executor.memory', '4g')
sc.setLogLevel("OFF")

input_file = sys.argv[1] # train_review.json
output_file = sys.argv[2] # task3item.model or task3user.model
cf_type = sys.argv[3] # item_based or user_based

start_time = time.time()

def getKeys (pairs):
    return [pair[0] for pair in pairs]

def getAverage (vector):
    sum0 = 0
    for i in vector:
        sum0 += i
    cnt = len(vector)
    return sum0/cnt

def normalizer (vector):
    avg = getAverage (vector)
    return [i-avg for i in vector]

def getCosineS (vector0, vector1):
    numerator = 0
    squaredSum0 = 0
    squaredSum1 = 0
    for idx, i in enumerate (vector0):
        numerator += i*vector1[idx]
        squaredSum0 += i**2
        squaredSum1 += vector1[idx]**2

    denominator = math.sqrt(squaredSum0 * squaredSum1)

    return numerator/denominator if denominator != 0 else 0

# computing the Pearson correlation for the business pairs having at least three co-rated users
if cf_type == "item_based":
    # [i, [[k1, r1], [k2, r2]]] # k: uid
    def getPearsonCorItem (index, ratingPairs0):
        pair_list = list()
        for j in range(index+1, N):
            # indexedRatings[j]: [j, [[k2, r1], [k3, r2]]]
            ratingPairs1 = indexedRatings[j][1]

            keys0 = getKeys (ratingPairs0)
            keys1 = getKeys (ratingPairs1)

            commonKey = set(keys0).intersection(keys1)

            # at least three co-rated users
            if len(commonKey) >= 3:
                vector0 = [ratingPairs0[keys0.index(key)][1] for key in commonKey]
                vector1 = [ratingPairs1[keys1.index(key)][1] for key in commonKey]

                n_vector0 = normalizer(vector0)
                n_vector1 = normalizer(vector1)

                cosineS = getCosineS (n_vector0 , n_vector1)

                if cosineS > 0:
                    pair_list.append([key_list[index], key_list[j], cosineS])

        return pair_list

    # multiple (b, u) pairs -> average ratings
    # b_users groupBy: [[b1, [[u1, r1], [u2, r2]]], [b2, [[u3, r3], [u4, ...

    b_users = sc.textFile(input_file)\
        .map(lambda x: [json.loads(x)['business_id'] + json.loads(x)['user_id'], json.loads(x)["stars"]])\
        .aggregateByKey((0,0), lambda R, v: (R[0]+v, R[1]+1), lambda R1, R2: (R1[0]+R2[0], R1[1]+R2[1]))\
        .map(lambda x: [x[0], x[1][0]/x[1][1]])\
        .map(lambda x: [x[0][:22], [x[0][22:], x[1]]])\
        .groupByKey().map(lambda x: [x[0], list(x[1])])\
        .filter(lambda x: len(x[1]) >= 3).cache()

    # [b1, b2, b3...]
    key_list = b_users.map(lambda x: x[0]).collect()
    N = len(key_list)

    # [[[u1, r1], [u2, r2]], [[u3, r3], [u4, ...
    userRatings = b_users.map(lambda x: x[1]).collect()

    b_users.unpersist()

    # [[0, [[u1, r1], [u2, r2]]], [1, [[u3, r3], [u4, ...
    indexedRatings = [[idx, ratings] for idx, ratings in enumerate(userRatings)]

    del userRatings

    pearsonCorList = sc.parallelize(indexedRatings)\
        .map(lambda x: getPearsonCorItem (x[0], x[1]))\
        .flatMap(lambda x: x)\
        .filter(lambda x: x != [])\
        .map(lambda x: {"b1": x[0], "b2": x[1], "sim": x[2]})\
        .collect()

    del indexedRatings

# (1) candidates user pairs with co-rated businesses, Minhash, LSH
# (2) Pearson correlation if JS >= 0.01 and at least 3 co-rated businesses
elif cf_type == "user_based":

    # multiple (u, b) pairs -> average ratings
    # key: user, value: business, rating
    # valueKeys: [[v1, [k1+r1, k2+r2]], [v2, [k3+r3, k4+r4]], ...
    valueKeys = sc.textFile(input_file)\
        .map(lambda x: [json.loads(x)['business_id'] + json.loads(x)['user_id'], json.loads(x)["stars"]])\
        .aggregateByKey((0,0), lambda R, v: (R[0]+v, R[1]+1), lambda R1, R2: (R1[0]+R2[0], R1[1]+R2[1]))\
        .map(lambda x: [x[0], round(x[1][0]/x[1][1],1)])\
        .map(lambda x: [x[0][:22], x[0][22:]+str(x[1])])\
        .groupByKey().map(lambda x: [x[0], list((x[1]))])\
        .sortBy(lambda x: x[0]).collect()

    m = len(valueKeys) # # of users
    # [v_idx1, [k1+r1, k2+r2, k3+r3]]
    indexKeys = [[idx, pair[1]] for idx, pair in enumerate(valueKeys)]
    del valueKeys

    # [[k1, [v1+r1, v2+r2], [k2, [v3+r3, v4+r4, v5+r5]], ...]
    keyValues = sc.parallelize(indexKeys)\
        .flatMapValues(lambda x: x)\
        .map(lambda x: [x[1][:22], str(x[0])+x[1][22:]])\
        .groupByKey().map(lambda x: [x[0], list(x[1])])\
        .filter(lambda x: len(x[1]) >= 3)\
        .sortByKey(lambda x: x[0]).cache()

    # n = # of hash functions n = b X r
    n = 120 # 120
    b = 40 # # of band
    r = int(n/b)

    N = keyValues.count() # # of users
    a = 37 # hash function constant

    # LSH bucket number
    def getBucketNum (band_vector):
        bucket_num = 0
        for idx, value in enumerate(band_vector):
            bucket_num += value * (17*(idx+1)) + 19
        return (bucket_num) % 183 #180  # x: 185

    # Minhash
    # values: [v3+r3, v4+r4, v5+r5]
    def signature (values):
        # initialize sign_vector
        sign_vector = [m for i in range(n)]

        for value in values:
            idx = int(value[:-3])
            # permutation new row ids
            row_ids = [(idx + a*c)% m for c in range(n)]

            for num, row_id in enumerate(row_ids):
                if sign_vector[num] > row_id:
                    sign_vector[num] = row_id

        # LSH get bucket num
        return [getBucketNum (sign_vector[band_num*r:band_num*r+r]) for band_num in range(b)]

    # [[k1, [v1+r1, v2+r2], [k2, [v3+r3, v4+r4, v5+r5]], ...]
    signature_matrix = keyValues\
        .map(lambda x: signature (x[1])).collect()
    matrix = [[idx, vector] for idx, vector in enumerate(signature_matrix)]
    # matrix: [[0, signature_vector1], [1, signature_vecotr2], [2, ..

    del signature_matrix

    # [[k1, [v1+r1, v2+r2], [k2, [v3+r3, v4+r4, v5+r5]], ...]
    keyValues = keyValues.collect()

    def getJScoS (index, j):
        # JS >= 0.01 and at least 3 co-rated businesses
        # keyValues: [[k1, [v1+r1, v2+r2], [k2, [v3+r3, v4+r4, v5+r5]], ...]

        # [v1+r1, v2+r2]
        ratingPairs0 = keyValues[index][1]
        ratingPairs1 = keyValues[j][1]

        # [v1, v2]
        keys0 = [pair[:-3] for pair in ratingPairs0]
        keys1 = [pair[:-3] for pair in ratingPairs1]

        commonKey = set(keys0).intersection(keys1)
        length = len(commonKey)

        # at least three co-rated users
        if length >= 3:
            # JS >= 0.01
            if length / len(set(keys0).union(keys1)) >= 0.01:
                # get Pearson correlation

                vector0 = [float(ratingPairs0[keys0.index(key)][-3:]) for key in commonKey]
                vector1 = [float(ratingPairs1[keys1.index(key)][-3:]) for key in commonKey]

                n_vector0 = normalizer(vector0) #
                n_vector1 = normalizer(vector1) #

                return getCosineS (n_vector0 , n_vector1)

            else:
                return 0
        else:
            return 0

    def getPearsonCorUser (index, signature_vector):
        pair_list = list()
        for j in range(index+1, N):
            vector1 = matrix[j][1]
            commonRows = set(signature_vector).intersection(vector1)
            if len(commonRows) != 0:
                for k in range(b):
                    if signature_vector[k] == vector1[k]:
                        cosineS = getJScoS (index, j)

                        if cosineS > 0:
                            pair_list.append([keyValues[index][0], keyValues[j][0], cosineS])

                        break

        return pair_list

    pearsonCorList = sc.parallelize(matrix)\
        .map(lambda x: getPearsonCorUser (x[0], x[1]))\
        .flatMap(lambda x: x)\
        .filter(lambda x: x != [])\
        .map(lambda x: {"u1": x[0], "u2": x[1], "sim": x[2]})\
        .collect()

    del matrix
    del keyValues

f = open(output_file, mode = "w+")
for item in pearsonCorList:
    json.dump(item, f)
    f.write("\n")
f.close()

print("Duration: %d" %int(time.time() -start_time))
