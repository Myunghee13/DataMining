
from pyspark import SparkContext
import sys
import json
import binascii
import math
import re

import time

sc = SparkContext()
sc.setSystemProperty('spark.driver.memory', '4g')
sc.setSystemProperty('spark.executor.memory', '4g')
sc.setLogLevel("OFF")

input1 = sys.argv[1] # business_first.json: set up the bit array for Bloom fitering
input2 = sys.argv[2] # business_second.json: prediction
output_file = sys.argv[3] # out.csv

start_time = time.time() # 60s

# make a Bloom Filter

def getHashset (x):
    value = int(binascii.hexlify(x.encode('utf8')),16)

    hashset = set()
    for i in range(k):
        hashset.add((value+b*i)%n)
    return list(hashset)

rdd1 = sc.textFile(input1)\
    .map(lambda x: json.loads(x)['city'])\
    .distinct().filter(lambda x: x != "")\
    .cache()

k = 10 # number of hash functions
m = rdd1.count() #  number of set elements
n = int(m*k/math.log(2))  # bit array size
b = int(n/k)

bloomFilter= set(rdd1.map(lambda x: getHashset(x))\
    .flatMap(lambda x: x)\
    .distinct().collect())

# prediction
def getCity(x):
    try:
        return json.loads(x)['city']
    except:
        return ""

def getPrecition(x):
    if x != "":
        cnt = len(bloomFilter.intersection(getHashset (x)))

        if cnt == 10:
            return 1
        else:
            return 0
    else:
        return 0


prediction = sc.textFile(input2)\
    .map(lambda x: getCity(x))\
    .map(lambda x: getPrecition(x)).collect()

with open(output_file, mode = "w+") as f:
    for i in prediction:
        f.write(str(i)+" ")

print("Duration: %d" %int(time.time() -start_time))