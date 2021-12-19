
from pyspark import SparkContext
import sys
import json
from statistics import mean
import time

sc = SparkContext()
sc.setSystemProperty('spark.driver.memory', '4g')
sc.setSystemProperty('spark.executor.memory', '4g')
sc.setLogLevel("OFF")

test_file = sys.argv[1] # test_review.json
output_file = sys.argv[2] # item.predict
N = 20 # 5, 7..

start_time = time.time()

# user, ratingList
def getPrediction0(user, ratingList0):
    ratingList = list(zip(*ratingList0))
    users = ratingList[0]

    try:
        simList = user_modelList[userList.index(user)]
    except:
        return mean(ratingList[1])  # no model

    try:
        avg = user_avg[user]
        weightVector = list()
        ratingVector = list()
        cnt = 0

        for user, weight in simList:
            if user in users:
                weightVector.append(weight)
                # normalized ( -avg)
                avg1 = user_avg[user]
                ratingVector.append(ratingList0[users.index(user)][1]-avg1)

                cnt += 1
                if cnt == N:
                    break

        if len(weightVector) == 0: # need to design
            return mean(ratingList[1])

        else:
            denominator = sum(weightVector)
            numerator = sum([a*b for a, b in zip(weightVector, ratingVector)])

            return avg+numerator/denominator

    except: # not normalize
        weightVector = list()
        ratingVector = list()
        cnt = 0
        for user, weight in simList:
            if user in users:
                weightVector.append(weight)
                ratingVector.append(ratingList0[users.index(user)][1])

                cnt += 1
                if cnt == N:
                    break

        if len(weightVector) == 0: # need to design
            return mean(ratingList[1])

        else:
            denominator = sum(weightVector)
            numerator = sum([a*b for a, b in zip(weightVector, ratingVector)])

            return numerator/denominator
def getAvg(user, item):
    try:
        return business_avg[item]
    except:
        try:
            return user_avg[user]
        except:
            return 4.8
#user, item, simList: [[u1,w1],[u2,w2]]
def getPrediction1 (user, item, simList):
    try:
        avg0 = business_avg[item]

        avgSimList = list(zip(*[[business_avg[v],sim] for v, sim in simList]))

        denominator = sum(avgSimList[1])

        numerator = sum([a*b for a, b in zip(avgSimList[0], avgSimList[1])])
        return (avg0 + numerator/denominator)/2
    except:
        return getAvg(user, item)

train_file = "../resource/asnlib/publicdata/train_review.json"
with open("user_model.json", "r") as f:
    user_model = sc.parallelize(json.load(f)["model"]).cache()

with open("../resource/asnlib/publicdata/user_avg.json", "r") as f:
    user_avg = json.load(f)
with open("../resource/asnlib/publicdata/business_avg.json", "r") as f:
    business_avg = json.load(f)

# multiple (item, user) pairs -> average ratings
rdd = sc.textFile(train_file)\
    .map(lambda x: [tuple((json.loads(x)['business_id'],json.loads(x)['user_id'])), json.loads(x)["stars"]])\
    .aggregateByKey((0,0), lambda R, v: (R[0]+v, R[1]+1), lambda R1, R2: (R1[0]+R2[0], R1[1]+R2[1]))\
    .map(lambda x: [x[0][0], x[0][1], x[1][0]/x[1][1]]).cache()

# userCF
#item_users: [[it1, [[u1, r1], [u2, r2]]], [it2, [[u3, r3], [u4, r4]...
item_users = rdd.map(lambda x: [x[0], [x[1], x[2]]])\
    .groupByKey().map(lambda x: [x[0], list(x[1])]).cache()

# [u1, u2...]
userList = user_model.map(lambda x: x[0]).collect()
# [[u2, sim2], [u4, sim4]]
user_modelList = user_model.map(lambda x: x[1]).collect()

# key: user, value: business/item, r: rating (star), w: sim (weight)
# testRDD: [user1, item1]
testRDD = sc.textFile(test_file)\
    .map(lambda x: [json.loads(x)['business_id'], json.loads(x)['user_id']]).cache()

# [it, [user, [[u1,r1],[u2,r2]]]]
joinRDD = testRDD.leftOuterJoin(item_users).cache()
# user, item, ratingList
userPrediction = joinRDD.filter(lambda x: x[1][1] != None)\
    .map(lambda x: [x[1][0], x[0], getPrediction0(x[1][0], x[1][1])])\
    .cache()

# user, item
joinRDD2 = joinRDD.filter(lambda x: x[1][1] == None)\
    .map(lambda x: [x[1][0], x[0]])

# [user, [it, [[u1,w1],[u2,w2]]]]
halfjoin = joinRDD2.leftOuterJoin(user_model).cache()

result00 = halfjoin.filter(lambda x: x[1][1] == None)\
    .map(lambda x: [x[0], x[1][0], getAvg(x[0], x[1][0])])

result01 = halfjoin.filter(lambda x: x[1][1] != None)\
    .map(lambda x: [x[0], x[1][0], getPrediction1(x[0], x[1][0], x[1][1][:N])])

result= userPrediction.union(result01).union(result00)\
    .map(lambda x: {"user_id": x[0], "business_id": x[1], "stars": x[2]})\
    .collect()

with open(output_file, mode = "w+") as f:
    for item in result:
        json.dump(item, f)
        f.write("\n")

print("Duration: %d" %int(time.time() -start_time))
