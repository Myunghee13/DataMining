
from pyspark import SparkContext
import sys
import json
import re
from collections import Counter

import time

sc = SparkContext()
sc.setSystemProperty('spark.driver.memory', '4g')
sc.setSystemProperty('spark.executor.memory', '4g')
sc.setLogLevel("OFF")

input_file = sys.argv[1] # train_review.json
output_file = sys.argv[2] # task2.model
stopwords_file = sys.argv[3] # stopwords

start_time = time.time()

stopwords = open(stopwords_file).read().split("\n")

def getTextSplit (text):
    return [w.lower() for w in re.split("[\n\r !\"#$%&\'()*+,-./0123456789:;=?@\[\]^_`{|}~\\\]+",text) if w.lower() not in stopwords]

# input: each document per business
def Combiner(text):
    words = getTextSplit (text)
    TF_dic = Counter()
    for word in words:
        TF_dic[word] += 1

    words_list = sorted(list(set(words)))

    return [[word, TF_dic[word]] for word in words_list]

# input: word counts per each document
# [[w1, c1], [w2, c2]] -> top200 TF-IDF, boolean
def getVector (wc_list) :
    # IDF_list: [[w1, idf1], [w2,idf2],..
    wordsVector = list()
    wc_idx = 0
    for w_idf in IDF_list:
        word0 = w_idf[0]
        try:
            word1 = wc_list[wc_idx][0]

            #idf = w_idf[1]
            #tf = wc_list[wc_idx][1]
            if word0 == word1:
                wordsVector.append(w_idf[1]*wc_list[wc_idx][1])
                wc_idx += 1

            elif word0 < word1:
                wordsVector.append(0)

            else:
                try:
                     while (word0 > wc_list[wc_idx][0]):
                        wc_idx += 1
                     word1 = wc_list[wc_idx][0]
                     if word0 == word1:
                         wordsVector.append(w_idf[1]*wc_list[wc_idx][1])
                         wc_idx += 1 #
                     elif word0 < word1:
                         wordsVector.append(0)

                except:
                    wordsVector.append(0)
        except:
            wordsVector.append(0)

    top200 = sorted(wordsVector, reverse = True)[199]

    b =""
    cnt = 0
    for i in wordsVector:
        if i >= top200 and cnt <200:
            b +="1"
            cnt += 1
        else:
            b +="0"

    return b # binary vector

def getUserProfile (bidList):
    int0 = int(businessProfile[businessList.index(bidList[0])][1],2)
    for bid in bidList:
        int0 = int0 | int(businessProfile[businessList.index(bid)][1],2)

    return int0

# Concatenating all the review texts for the business as the document
# [[b1, d1], [b2, d2], ...] by reduceByKey
# [b1, [[w1, c1], [w2, c2]]]
business_docs = sc.textFile(input_file)\
    .map(lambda x: [json.loads(x)['business_id'], json.loads(x)['text']])\
    .reduceByKey(lambda a, b: a+b).cache()

businessList = business_docs.map(lambda x: x[0]).collect()
N = len(businessList)

# word counts per each document
combinerRDD = business_docs\
    .map(lambda x: Combiner(x[1])).cache()

# total words count
# rdd0: [w1, c11], [w2, c21], [w1, c12]..
words_RDD0 = combinerRDD.flatMap(lambda x: x).cache()

words_count = words_RDD0.map(lambda x: [1, x[1]])\
    .reduceByKey(lambda x, y: x+y)\
    .map(lambda x: x[1]).collect()

# [w1, c1], [w2, c2]
filtered_words = words_RDD0.reduceByKey(lambda x, y: x+y).cache()\
    .filter(lambda x: x[1] >= words_count[0]*0.000001)\
    .map(lambda x: x[0]).collect()

# IDF_list: [[w1, idf1], [w2,idf2],..
IDF_list = words_RDD0\
    .map(lambda x: [x[0],1])\
    .reduceByKey(lambda x, y: x+y)\
    .filter(lambda x: x[0] in filtered_words)\
    .map(lambda x: [x[0], N/x[1]])\
    .sortBy(lambda x: x[0])\
    .collect()

words_RDD0.unpersist()
del filtered_words

# combiner: word counts per each document
# [[w1, c1], [w2, c2]] -> top200 TF-IDF, boolean
businessVector = combinerRDD\
    .map(lambda x: getVector(x))\
    .collect()

combinerRDD.unpersist()
business_docs.unpersist()

del IDF_list

businessProfile = [[bid, businessVector[idx]] for idx, bid in enumerate(businessList)]

# userProfile: [[u1, [b1, b2]], [u2, [b3, b4, b5]], ...]

userProfile = sc.textFile(input_file)\
    .map(lambda x: [json.loads(x)["user_id"] , json.loads(x)['business_id']])\
    .groupByKey().map(lambda x: [x[0], list(set(x[1]))])\
    .map(lambda x: [x[0], getUserProfile(x[1])]).collect()

del businessList

answer = {"businessProfile": businessProfile, "userProfile": userProfile}

f = open(output_file, mode = "w+") #, encoding='utf-8')
json.dump(answer, f)
f.close()

print("Duration: %d" %int(time.time() -start_time))