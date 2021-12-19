# !/usr/bin/env python
# coding: utf-8

from pyspark import SparkContext
import sys
import json
import re
import gc

sc = SparkContext()
sc.setSystemProperty('spark.driver.memory', '12g')
sc.setSystemProperty('spark.executor.memory', '12g')
sc.setLogLevel("OFF")

input_file = sys.argv[1]
output_file = sys.argv[2]
stopwords_file = sys.argv[3]

y = str(sys.argv[4])
m = int(sys.argv[5])
n = int(sys.argv[6])

review_RDD = sc.textFile(input_file).cache()

# B. The number of reviews in a given year, y (0.5pts)
B = review_RDD.filter(lambda x: json.loads(x)['date'][:4] == y).count()

# C. The number of distinct users who have written the reviews (0.5pts)
user_RDD = review_RDD.map(lambda x: json.loads(x)['user_id']).cache()
C = user_RDD.distinct().count()

# A. The total number of reviews (0.5pts)
A = user_RDD.count()

# D. Top m users who have the largest number of reviews and its count (0.5pts)

D = user_RDD.map(lambda x: [x, 1]).reduceByKey(lambda a, b: a + b) \
    .sortBy(lambda x: x[0]).sortBy(lambda x: x[1], False).take(m)

user_RDD.unpersist()

# E. Top n frequent words in the review text
stopwords = open(stopwords_file).read().split("\n")

textRDD = review_RDD.map(lambda x: json.loads(x)['text']).cache()
review_RDD.unpersist()
gc.collect()

E = textRDD.flatMap(lambda x: [w.lower() for w in re.split("[(\[,.!?:;\]) ]+", x)]) \
    .filter(lambda x: x not in stopwords).map(lambda x: [x, 1]).reduceByKey(lambda x, y: x + y) \
    .sortBy(lambda x: x[1]).sortBy(lambda x: x[1], False).map(lambda x: x[0]).take(n)

answer = {"A": A, "B": B, "C": C, "D": D, "E": E}

f = open(output_file, mode="w+")
json.dump(answer, f)
f.close()
