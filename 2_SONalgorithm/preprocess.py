
#!/usr/bin/env python
# coding: utf-8

from pyspark import SparkContext
import json

sc = SparkContext()
sc.setSystemProperty('spark.driver.memory', '4g')
sc.setSystemProperty('spark.executor.memory', '4g')
sc.setLogLevel("OFF")

review_file = "review.json"
business_file = "business.json"
output_file = "user_business.csv"

review = sc.textFile(review_file).map(lambda x: (json.loads(x)['business_id'], json.loads(x)['user_id']))
business = sc.textFile(business_file).map(lambda x: (json.loads(x)['business_id'], json.loads(x)['state']))\
.filter(lambda x: x[1] =="NV")

# [business_id, (user_id, state)]
join1 = review.join(business)
output = join1.map(lambda x: [x[1][0], x[0]]).collect()

f = open(output_file, mode = "w+")
f.write("user_id,business_id\n")
for x in output:
    f.write("%s,%s\n" % (x[0], x[1]))
f.close()