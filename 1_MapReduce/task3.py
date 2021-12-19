# !/usr/bin/env python
# coding: utf-8

from pyspark import SparkContext
import sys
import json

sc = SparkContext()
sc.setSystemProperty('spark.driver.memory', '4g')
sc.setSystemProperty('spark.executor.memory', '4g')
sc.setLogLevel("OFF")

review_file = sys.argv[1]
output_file = sys.argv[2]
partition_type = sys.argv[3]  # default or customized
n_part = int(sys.argv[4])  # effective only for customized
n = int(sys.argv[5])


# compute the businesses with more than n reviews (1pts).
# show the number of partitions for the RDD and the number of items per partition with either default or customized partition function (1pts).
# design a customized partition function to improve the computational efficiency, i.e., reducing the time duration of execution (1pts).

def num_item(iterator):
    yield len(list(iterator))


if partition_type == "default":
    review = sc.textFile(review_file).map(lambda x: (json.loads(x)['business_id']))
    review_rdd = review.map(lambda x: [x, 1]).cache()

    n_partitions = review_rdd.getNumPartitions()
    n_items = review_rdd.mapPartitions(num_item).collect()

    result = review_rdd.reduceByKey(lambda a, b: a + b).filter(lambda x: x[1] > n).collect()

elif partition_type == "customized":

    def partitioner(key):
        return hash(key)


    review = sc.textFile(review_file).map(lambda x: (json.loads(x)['business_id']))
    review_rdd = review.map(lambda x: [x, 1]).partitionBy(n_part, partitioner).cache()

    n_partitions = review_rdd.getNumPartitions()
    n_items = review_rdd.mapPartitions(num_item).collect()

    result = review_rdd.reduceByKey(lambda a, b: a + b).filter(lambda x: x[1] > n).collect()

answer = {"n_partitions": n_partitions, "n_items": n_items, "result": result}

f = open(output_file, mode="w+")
json.dump(answer, f)
f.close()
