# !/usr/bin/env python
# coding: utf-8

import sys
import json

review_file = sys.argv[1]
business_file = sys.argv[2]
output_file = sys.argv[3]
if_spark = sys.argv[4]  # no_spark, spark
n = int(sys.argv[5])

if if_spark == "spark":
    from pyspark import SparkContext

    sc = SparkContext()
    sc.setSystemProperty('spark.driver.memory', '4g')
    sc.setSystemProperty('spark.executor.memory', '4g')
    sc.setLogLevel("OFF")

    review = sc.textFile(review_file).map(lambda x: (json.loads(x)['business_id'], json.loads(x)['stars']))
    business = sc.textFile(business_file).map(lambda x: (json.loads(x)['business_id'], json.loads(x)['categories']))

    # (business_id, average)

    review1 = review.aggregateByKey((0, 0), lambda R, v: (R[0] + v, R[1] + 1),
                                    lambda R1, R2: (R1[0] + R2[0], R1[1] + R2[1]))
    review2 = review1.map(lambda x: (x[0], round(x[1][0] / x[1][1], 3)))

    # (business_id, [category1, category2, ...])
    business_rdd = business.map(lambda x: (x[0], [y.strip() for y in str(x[1]).split(',')]))

    # (id, [ave, [cat1, cat2, ...]])
    join = review2.join(business_rdd)

    # [cat, ave]
    cat_ave = join.flatMap(lambda x: [(a, x[1][0]) for a in x[1][1]])
    cat1 = cat_ave.aggregateByKey((0, 0), lambda R, v: (R[0] + v, R[1] + 1),
                                  lambda R1, R2: (R1[0] + R2[0], R1[1] + R2[1]))
    cat2 = cat1.map(lambda x: [x[0], round(x[1][0] / x[1][1], 1)])

    R = cat2.sortBy(lambda x: x[0]).sortBy(lambda x: x[1], False).take(n)

elif if_spark == "no_spark":
    from functools import reduce
    from itertools import groupby

    review = []
    with open(review_file) as f:
        for line in f:
            review.append((json.loads(line)['business_id'], json.loads(line)['stars']))

    business = []
    with open(business_file) as f:
        for line in f:
            business.append((json.loads(line)['business_id'], json.loads(line)['categories']))

    review.sort()
    business.sort()

    len2 = len(business)

    # (business_id, average)
    business = list(map(lambda x: (x[0], [y.strip() for y in str(x[1]).split(',')]), business))

    sum_fnc = lambda a, b: a + b
    cnt_fnc = lambda a, b: a + 1

    rate_sum = list(map(lambda x: (x[0], reduce(sum_fnc, map(lambda d: d[1], x[1]))), groupby(review, lambda x: x[0])))
    cnt = list(map(lambda x: reduce(cnt_fnc, map(lambda d: d[1], x[1]), 0), groupby(review, lambda x: x[0])))

    ave_list = []
    len1 = len(cnt)

    for i in range(len1):
        ave = rate_sum[i][1] / cnt[i]
        ave_list.append((rate_sum[i][0], round(ave, 3)))

    category_rate = []
    s = 0
    for i in range(len1):
        for j in range(s, len2):
            if ave_list[i][0] > business[j][0]:
                s = j + 1
                continue
            elif ave_list[i][0] == business[j][0]:
                category_rate.append((business[j][1], ave_list[i][1]))
                s = j + 1
                break
            else:
                s = j
                break

    category_rate = list(map(lambda x: [(category, x[1]) for category in x[0]], category_rate))

    flat_cat = []
    for items in category_rate:
        for item in items:
            flat_cat.append(item)

    flat_cat.sort()

    rate_sum = list(
        map(lambda x: (x[0], reduce(sum_fnc, map(lambda d: d[1], x[1]))), groupby(flat_cat, lambda x: x[0])))
    cnt = list(map(lambda x: reduce(cnt_fnc, map(lambda d: d[1], x[1]), 0), groupby(flat_cat, lambda x: x[0])))
    ave_list = []

    length = len(cnt)

    for i in range(length):
        ave = rate_sum[i][1] / cnt[i]
        ave_list.append([rate_sum[i][0], round(ave, 1)])

    sorted_cat = sorted(sorted(ave_list, key=lambda d: d[0]), key=lambda d: d[1], reverse=True)

    R = sorted_cat[:n]

answer = {"result": R}

f = open(output_file, mode="w+")
json.dump(answer, f)
f.close()
