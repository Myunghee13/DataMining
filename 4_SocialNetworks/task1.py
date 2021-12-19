from pyspark import SparkContext
from graphframes import *
from pyspark.sql import *
import sys

import time

sc = SparkContext('local[3]')
sc.setSystemProperty('spark.driver.memory', '4g')
sc.setSystemProperty('spark.executor.memory', '4g')
sc.setLogLevel("OFF")

ss = SparkSession.builder \
    .master('local[3]') \
    .getOrCreate()

ft_th = int(sys.argv[1])  # filter threshold: 7
input_file = sys.argv[2]  # ub_sample_data.csv
output_file = sys.argv[3]

start_time = time.time()  # 200s

# Graph Construction

# [[u1, [b1, b2..]], [u2, [b3, b4, ..]],
rdd = sc.textFile(input_file) \
    .map(lambda x: x.split(',')) \
    .groupByKey().map(lambda x: [x[0], set(x[1])]) \
    .filter(lambda x: len(x[1]) >= ft_th).cache()

# [u1, u2,..
user_list = rdd.map(lambda x: x[0]).collect()
n0 = len(user_list)

# [[b1, b2..], [b3, b4, ..],
bset_list = rdd.map(lambda x: x[1]).collect()


def getUserPairs(node, bset0):
    neighbors = list()
    for idx in range(n0):
        if len(bset0.intersection(bset_list[idx])) >= ft_th:
            neighbors.append(user_list[idx])

    neighbors.remove(node)
    return [node, sorted(neighbors)]


# [[u1, [b1, b2..]], [u2, [b3, b4, ..]],
# -> [[u1, [u2, u4, ..]], [u2, [u1, u5, u6,..]],
graphRDD = rdd.map(lambda x: getUserPairs(x[0], x[1])) \
    .filter(lambda x: x[1] != []) \
    .sortBy(lambda x: x[0]).cache()

# [u1, u2, ..]
vertices0 = graphRDD.map(lambda x: (x[0],))  # , user_list.index(x[0])])

vertices = ss.createDataFrame(vertices0, ["id"])


def getEdges(node, nList):
    return [sorted([node, node1]) for node1 in nList]


# [[u1, [u2, u4, ..]], [u2, [u5, u6,..]],
# -> [[u1, u2], [u1, u4], ...
edges0 = graphRDD.map(lambda x: getEdges(x[0], x[1])) \
    .flatMap(lambda x: x) \
    .map(lambda x: tuple(x)) \
    .distinct() \
    .sortBy(lambda x: x[0]) \
    .map(lambda x: [[x[0], x[1]], [x[1], x[0]]]) \
    .flatMap(lambda x: x)

edges = ss.createDataFrame(edges0, ["src", "dst"])

g = GraphFrame(vertices, edges)

result = g.labelPropagation(maxIter=5).rdd \
    .map(lambda x: [x[1], x[0]]) \
    .groupByKey() \
    .map(lambda x: sorted(list(x[1]))) \
    .sortBy(lambda x: x) \
    .sortBy(lambda x: len(x)).collect()

del bset_list

f = open(output_file, mode="w+")
for item in result:
    f.write(str(item).strip("[]"))
    f.write("\n")

f.close()

print("Duration: %d" % int(time.time() - start_time))
