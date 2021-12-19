
from pyspark import SparkContext
import sys
import json

import time

sc = SparkContext()
sc.setSystemProperty('spark.driver.memory', '4g')
sc.setSystemProperty('spark.executor.memory', '4g')
sc.setLogLevel("OFF")

input_file = sys.argv[1] # train_review.json
output_file = sys.argv[2] # task1.res

start_time = time.time()

# key: business, value: user index
# [v1, [k1, k2]], [v2, [k3, k4, k5]], ...]
valueKeys = sc.textFile(input_file)\
    .map(lambda x: (json.loads(x)['user_id'], json.loads(x)['business_id']))\
    .groupByKey().map(lambda x: [x[0], list(set(x[1]))]).sortBy(lambda x: x[0]).collect()

m = len(valueKeys) # # of users
indexKeys = [[idx, pair[1]] for idx, pair in enumerate(valueKeys)]
del valueKeys

# [[k1, [v1, v2]], [k2, [v3, v4, v5]], ...]
keyValues = sc.parallelize(indexKeys)\
    .flatMapValues(lambda x: x)\
    .map(lambda x: [x[1], x[0]])\
    .groupByKey().map(lambda x: [x[0], list(x[1])])\
    .sortBy(lambda x: x[0]).cache()

# n = # of hash functions n = b X r
n = 160
b = 40 # # of band
r = int(n/b)

N = keyValues.count() # # of businesses
a = 113 # hash function constant

# LSH bucket number
def getBucketNum (band_vector):
    bucket_num = 0
    for idx, value in enumerate(band_vector):
        bucket_num += value * (177*(idx+1)) + 199
    return (bucket_num) % 100#80

# Minhash
def signature (values):
    # initialize sign_vector
    sign_vector = [m for i in range(n)]

    for idx in values:
        # permutation new row ids
        row_ids = [(idx + a*c)% m for c in range(n)]

        for num, row_id in enumerate(row_ids):
            if sign_vector[num] > row_id:
                sign_vector[num] = row_id

    # LSH getBucket Num
    return [getBucketNum (sign_vector[band_num*r:band_num*r+r]) for band_num in range(b)]

def checkJS(i, signature_vector):
    pair_list = list()
    for j in range(i+1, N):
        vector1 = matrix[j][1]
        commonRows = set(signature_vector).intersection(vector1)
        if len(commonRows) != 0:
            for k in range(b):
                if signature_vector[k] == vector1[k]:
                    p0 = set(keyValues[i][1])
                    p1 = keyValues[j][1]
                    JS = len(p0.intersection(p1)) / len(p0.union(p1))
                    if JS >= 0.05:
                        pair_list.append([keyValues[i][0], keyValues[j][0], JS])

                    break

    return pair_list

signature_matrix = keyValues.map(lambda x: signature (x[1])).collect()
matrix = [[idx, vector] for idx, vector in enumerate(signature_matrix)]
# matrix: [[0, signature_vector1], [1, signature_vecotr2], [2, ..

del signature_matrix

keyValues = keyValues.collect()

JS_result = sc.parallelize(matrix)\
    .map(lambda x: checkJS(x[0], x[1]))\
    .flatMap(lambda x: x)\
    .filter(lambda x: x != [])\
    .map(lambda x: {"b1": x[0], "b2": x[1], "sim": x[2]})\
    .collect()

del matrix
del keyValues

f = open(output_file, mode = "w+")
for item in JS_result:
    json.dump(item, f)
    f.write("\n")

f.close()

print("Duration: %d" %int(time.time() -start_time))
