from pyspark import SparkContext
import sys
import json
import re
import math

import time

sc = SparkContext()
sc.setSystemProperty('spark.driver.memory', '4g')
sc.setSystemProperty('spark.executor.memory', '4g')
sc.setLogLevel("OFF")

input_file = sys.argv[1]  # test_review.json
model_file = sys.argv[2]  # task2.model
output_file = sys.argv[3]  # task2.predict

start_time = time.time()


def count1bits(bits):
    return len(re.sub("0", "", bits))


def getCosineD(uid, bid):
    try:
        vector0int = userProfile[uidList.index(uid)][1]
        vector0bits = bin(vector0int)[2:]

        vector1bits = businessProfile[bidList.index(bid)][1]
        vector1int = int(vector1bits, 2)

        denominator = math.sqrt(count1bits(vector0bits) * count1bits(vector1bits))

        if denominator == 0:
            return None
        else:
            numerator = count1bits(bin(vector0int & vector1int)[2:])
            if numerator == 0:
                return None
            else:
                cosineSim = numerator / denominator
                if cosineSim >= 0.01:
                    return cosineSim
                else:
                    return None
    except:
        return None


filehandle = open(model_file, 'r')

model = json.load(filehandle)

businessProfile = model["businessProfile"]
userProfile = model["userProfile"]

filehandle.close()

bidList = sc.parallelize(businessProfile) \
    .map(lambda x: x[0]).collect()
uidList = sc.parallelize(userProfile) \
    .map(lambda x: x[0]).collect()

testList = sc.textFile(input_file) \
    .map(lambda x: [json.loads(x)["user_id"], json.loads(x)["business_id"]]) \
    .map(lambda x: [x[0], x[1], getCosineD(x[0], x[1])]) \
    .filter(lambda x: x[2] != None) \
    .map(lambda x: {"user_id": x[0], "business_id": x[1], "sim": x[2]}) \
    .collect()

f = open(output_file, mode="w+")

# answer = {"bidList": bidList[:3], "p1":businessProfile[:3], "uidList": uidList[:3],  "p2": businessProfile[:3]}
# json.dump(answer, f)

for item in testList:
    json.dump(item, f)
    f.write("\n")

f.close()

print("Duration: %d" % int(time.time() - start_time))