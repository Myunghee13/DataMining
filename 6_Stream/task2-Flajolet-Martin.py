# java -cp generate_stream.jar StreamSimulation business.json 9999 100 > /dev/null
from pyspark import SparkContext
from pyspark.streaming import StreamingContext
from datetime import datetime
from statistics import mean
from statistics import median

import sys
import json
import binascii

sc = SparkContext()
sc.setSystemProperty('spark.driver.memory', '4g')
sc.setSystemProperty('spark.executor.memory', '4g')
sc.setLogLevel("OFF")

port_num = int(sys.argv[1])
output_file = sys.argv[2]  # out.csv

g = 8  # # of groups
k = 9  # # of hash per group
N = g * k  # total # of hashing

n = 12407
b = int(n / N)
a = 19181


def getTrailing(x):
    value = int(binascii.hexlify(x.encode('utf8')), 16)
    return [len(bin((value * a + b * (j + 1)) % (n + i * 10000)).split("1")[-1]) for i in range(g) for j in range(k)]


def getResult(rdd):
    time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

    truth = str(rdd.map(lambda x: [x, 1]) \
                .reduceByKey(lambda a, b: a + b) \
                .count())

    pred = rdd.map(lambda x: getTrailing(x)).collect()

    maxtrailing = [max(c) for c in zip(*pred)]

    prediction = str(int(2 ** median([mean(c) for c in zip(*[iter(maxtrailing)] * k)])))

    result = time + "," + truth + "," + prediction + "\n"
    with open(output_file, 'a') as f:
        f.write(result)


ssc = StreamingContext(sc, 5)
ssc.checkpoint("checkpoint")

lines = ssc.socketTextStream("localhost", port_num)
with open(output_file, 'w') as f:
    f.write("Time,Gound Truth,Estimation\n")

city = lines.map(lambda x: json.loads(x)['city']) \
    .filter(lambda x: x != "") \
    .window(30, 10) \
    .foreachRDD(getResult)

ssc.start()  # Start the computation
ssc.awaitTermination()  # Wait for the computation to terminate