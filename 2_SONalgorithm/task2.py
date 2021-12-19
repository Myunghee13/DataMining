# !/usr/bin/env python
# coding: utf-8

from pyspark import SparkContext
import sys
from collections import Counter
import itertools
import re
import time

sc = SparkContext()
sc.setSystemProperty('spark.driver.memory', '4g')
sc.setSystemProperty('spark.executor.memory', '4g')
sc.setLogLevel("OFF")

k = int(sys.argv[1])  # k: filter threshold
support = int(sys.argv[2])
input_file = sys.argv[3]
output_file = sys.argv[4]

start_time = time.time()

raw_rdd = sc.textFile(input_file).cache()
header = raw_rdd.first()
rdd = raw_rdd.filter(lambda x: x != header).map(lambda x: x.split(','))


def getCandidatekItem(chunk_kFreq, kFreq_element, k):
    candidate_k = []
    for combination in \
            itertools.combinations(kFreq_element, k):
        idx = 0
        for subset in itertools.combinations(combination, k - 1):
            if tuple(sorted(subset)) not in chunk_kFreq:
                break
            else:
                idx += 1
        if idx == k:
            candidate_k.append(combination)
    candidate_k.sort()
    # [(e1, e2, ...ek), ...]
    return candidate_k


def getChunkFreq(candidate_k, partition):
    dic = {}
    chunk_kFreq = set()
    kFreq_element = set()
    for key in candidate_k:
        key = tuple(sorted(key))
        for basket in partition:
            if set(key).issubset(basket):
                if key in dic:
                    dic[key] += 1
                    if dic[key] >= p:
                        chunk_kFreq.add(key)
                        kFreq_element.update(key)
                        break
                else:
                    dic.update({key: 1})
    # chunk_kFreq: {(e1, e2, ...ek), ...}
    # kFreq_element {e1, e2, ek...}
    return chunk_kFreq, kFreq_element


def PCY(iterator):
    partition = list(iterator)

    dic2 = Counter()
    dic = Counter()  ## PCY: counting bucket
    for basket in partition:
        dic2.update(basket)

        for pair in itertools.combinations(basket, 2):  ##
            hash_value = str((hash(pair[0]) + hash(pair[1])) % 6000000)  ## hash function
            dic[hash_value] += 1  ##

    bitmap0_list = []  ##
    for key in dic:  ##
        if dic[key] >= p:  ##
            bitmap0_list.append(int(key))  ##

    candidates = []
    for key in dic2:
        if dic2[key] >= p:
            candidates.append((key,))
    candidates.sort()
    length = len(candidates)

    dic = {}
    chunk_kFreq = set()
    kFreq_element = set()
    for key in itertools.combinations( \
            [ele[0] for ele in candidates], 2):
        if ((hash(key[0]) + hash(key[1])) % 6000000) in bitmap0_list:  ##
            key = tuple(sorted(key))
            for basket in partition:
                if set(key).issubset(basket):
                    if key in dic:
                        dic[key] += 1
                        if dic[key] >= p:
                            chunk_kFreq.add(key)
                            kFreq_element.update(key)
                            break
                    else:
                        dic.update({key: 1})

    for k in range(3, length + 1):
        if len(chunk_kFreq) == 0:
            break
        elif len(chunk_kFreq) == 1:
            candidates += sorted(list(chunk_kFreq))
            break
        else:
            candidates += sorted(list(chunk_kFreq))
            candidate_k = getCandidatekItem(chunk_kFreq, kFreq_element, k)
            chunk_kFreq, kFreq_element = getChunkFreq(candidate_k, partition)

    yield candidates


def map_function(x):
    key_list = []
    for key in map1:
        if set(key).issubset(x):
            key_list.append((key, 1))
    return key_list


def out_sort(result_list):
    result_list.sort()

    sorted_list = []
    t = 1
    i = 1
    while (t != 0):
        t = 0
        for c in result_list:
            if len(c) == i:
                sorted_list.append(c)
                t += 1
        i += 1
    return sorted_list


baskets = rdd.groupByKey().map(lambda x: (x[0], list(set(x[1])), len(list(set(x[1]))))) \
    .filter(lambda x: x[2] > k).map(lambda x: x[1]).cache()
n_partitions = baskets.getNumPartitions()
p = support / n_partitions

# 1 pass: get candidates
map1 = baskets.mapPartitions(PCY).flatMap(lambda x: x).distinct().collect()

candidates = out_sort(map1)

# 2 pass: get frequent itemsets
map2 = baskets.flatMap(map_function).reduceByKey(lambda x, y: x + y) \
    .filter(lambda x: x[1] >= support).map(lambda x: x[0]).collect()

frequents = out_sort(map2)

f = open(output_file, mode="w+")
f.write("Candidates: \n")
num_candidates = len(candidates)
for idx, c in enumerate(candidates):
    if num_candidates == 0:
        f.write("\n")
    elif num_candidates == 1:
        f.write(re.sub(",\)", ")", str(candidates[0]) + "\n\n"))
    elif len(candidates[num_candidates - 1]) == 1:
        if idx != num_candidates - 1:
            f.write(re.sub(",\)", ")", str(c)) + ",")
        else:
            f.write(re.sub(",\)", ")", str(c)) + "\n\n")

    else:
        if len(candidates[idx]) == 1 and len(candidates[idx + 1]) == 1:
            f.write(re.sub(",\)", ")", str(c)) + ",")
        elif len(candidates[idx]) == 1 and len(candidates[idx + 1]) == 2:
            f.write(re.sub(",\)", ")", str(c)) + "\n\n")
        elif len(candidates[idx]) > 1 and idx != num_candidates - 1:
            if len(candidates[idx]) == len(candidates[idx + 1]):
                f.write(str(c) + ",")
            else:
                f.write(str(c) + "\n\n")
        else:
            f.write(str(c) + "\n\n")

f.write("Frequent Itemsets: \n")
num_frequents = len(frequents)
for idx, c in enumerate(frequents):
    if num_frequents == 0:
        f.write("\n")
    elif num_frequents == 1:
        f.write(re.sub(",\)", ")", str(frequents[0])))
    elif len(frequents[num_frequents - 1]) == 1:
        if idx != num_frequents - 1:
            f.write(re.sub(",\)", ")", str(c)) + ",")
        else:
            f.write(re.sub(",\)", ")", str(c)))

    else:
        if len(frequents[idx]) == 1 and len(frequents[idx + 1]) == 1:
            f.write(re.sub(",\)", ")", str(c)) + ",")
        elif len(frequents[idx]) == 1 and len(frequents[idx + 1]) == 2:
            f.write(re.sub(",\)", ")", str(c)) + "\n\n")
        elif len(frequents[idx]) > 1 and idx != num_frequents - 1:
            if len(frequents[idx]) == len(frequents[idx + 1]):
                f.write(str(c) + ",")
            else:
                f.write(str(c) + "\n\n")
        else:
            f.write(str(c))

f.close()

print("Duration: %d" % int(time.time() - start_time))