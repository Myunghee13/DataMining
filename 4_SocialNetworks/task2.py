from pyspark import SparkContext
import sys
import itertools

import time

sc = SparkContext()
sc.setSystemProperty('spark.driver.memory', '4g')
sc.setSystemProperty('spark.executor.memory', '4g')
sc.setLogLevel("OFF")

ft_th = int(sys.argv[1])  # filter threshold: 7
input_file = sys.argv[2]  # ub_sample_data.csv
btw_output_file = sys.argv[3]
cmm_output_file = sys.argv[4]

start_time = time.time()  # 250s


# rootG: [0, [1, 3]]
def getBTW(rootG):
    # get BFS
    # initialize depth 0 & 1
    BFS = [[rootG]]  # [[[0, [1, 3]]]]
    sibling = rootG[1]  # [1, 3]
    preNodes = {rootG[0]}.union(sibling)  # {0, 1, 3}

    # depth 2~
    while (len(sibling) != 0):
        nextParentSet = set()
        par_children = list()
        for node in sibling:  # nextParentSet: [1, 3]
            # node: 1, graph[node][1]: [0, 4, 5]
            children = set(graph[node][1]) - preNodes  # children: {4, 5}

            if len(children) != 0:
                nextParentSet = children.union(nextParentSet)
                par_children.append([node, list(children)])
        sibling = nextParentSet
        if par_children != []:
            preNodes = preNodes.union(nextParentSet)
            BFS.append(par_children)

    # get betweennness

    # get number of paths
    rootGraph = BFS[0][0]
    parent0 = rootGraph[1]
    depth = len(BFS)
    path_num = {rootGraph[0]: 1}
    for node in parent0:
        path_num.update({node: 1})

    for level in range(1, depth):  # d: depth
        for unitG in BFS[level]:
            for node in unitG[1]:
                if node in path_num:
                    path_num[node] += path_num[unitG[0]]
                else:
                    path_num.update({node: path_num[unitG[0]]})

    node_value = dict()
    edge_value = dict()

    for subG in BFS[depth - 1]:
        parent = subG[0]
        node_value[parent] = 1
        for child in subG[1]:
            weight = path_num[parent] / path_num[child]
            edge_value.update({tuple(sorted([parent, child])): weight})
            node_value[parent] += weight

    for i in range(depth - 1):
        for subG in BFS[depth - i - 2]:
            parent = subG[0]
            node_value[parent] = 1
            for child in subG[1]:
                try:
                    edge = (path_num[parent] / path_num[child]) * node_value[child]
                    edge_value.update({tuple(sorted([parent, child])): edge})
                    node_value[parent] += edge

                except:
                    weight = path_num[parent] / path_num[child]
                    edge_value.update({tuple(sorted([parent, child])): weight})
                    node_value[parent] += weight

    return [[[(edge[0], edge[1]), edge_value[edge]] for edge in edge_value], preNodes]


# Graph Construction
# [[u1, [b1, b2..]], [u2, [b3, b4, ..]],
rdd = sc.textFile(input_file) \
    .map(lambda x: x.split(',')) \
    .groupByKey().map(lambda x: (x[0], set(x[1]))) \
    .filter(lambda x: len(x[1]) >= ft_th).cache()

# [u1, u2,..
user_list = rdd.map(lambda x: x[0]).collect()
n0 = len(user_list)

# [[b1, b2..], [b3, b4, ..],
bset_list = rdd.map(lambda x: x[1]).collect()


def getUserPairs(node, bset0):
    neighbors = list()
    for idx in range(n0):
        if len(bset0.intersection(bset_list[idx])) >= ft_th:
            neighbors.append(user_list[idx])

    neighbors.remove(node)
    return [node, sorted(neighbors)]


# [[u1, [b1, b2..]], [u2, [b3, b4, ..]],
# -> [[u1, [u2, u4, ..]], [u2, [u5, u6,..]],
graphRDD0 = rdd.map(lambda x: getUserPairs(x[0], x[1])) \
    .filter(lambda x: x[1] != []).cache()

user_list = graphRDD0.map(lambda x: x[0]).collect()
n = len(user_list)

# [[u1, [u2, u4, ..]], [u2, [u5, u6,..]],
# -> [[idx1, [idx2, idx4, ..]], [idx2, [idx5, idx6,..]],
graphRDD = graphRDD0 \
    .map(lambda x: [user_list.index(x[0]), [user_list.index(node) for node in x[1]]]) \
    .cache()

del bset_list

adjList = graphRDD.collect()  # A: Adjacency List, original graph

graph = graphRDD.collect()  # will remove edge continuously
degreeRDD = graphRDD.map(lambda x: len(x[1])).cache()
degreeList = degreeRDD.collect()  # Degree matrix

m = int(degreeRDD.map(lambda x: (1, x)) \
        .reduceByKey(lambda a, b: a + b) \
        .map(lambda x: x[1] / 2).take(1)[0])  # # of edges
m2 = m * 2

# {(0, 1): A01-k0k1/2m, (0,2): ...}
AKdic = dict()


def getAKdic(i):
    neighbors = adjList[i][1]
    ki = degreeList[i]
    for j in range(i + 1, n):
        # kj = degreeList[j]
        if j in neighbors:
            AKdic.update({(i, j): 1 - ki * degreeList[j] / m2})
        else:
            AKdic.update({(i, j): -ki * degreeList[j] / m2})
    return 1


for i in range(n):
    getAKdic(i)

btwRDD = graphRDD.map(lambda x: getBTW(x)).cache()

btwrdd = btwRDD.map(lambda x: x[0]) \
    .flatMap(lambda x: x) \
    .reduceByKey(lambda a, b: a + b) \
    .map(lambda x: [tuple(sorted([user_list[x[0][0]], user_list[x[0][1]]])), x[1] / 2]) \
    .sortBy(lambda x: x[0]).sortBy(lambda x: x[1], False).cache()

btw = btwrdd.collect()

# modularlity

# x[1]: set
partitions = [btwRDD.map(lambda x: tuple(sorted(x[1]))).distinct() \
                  .sortBy(lambda x: x).collect()]

# btw for modularity: converting uid -> index of user_list
btw2 = btwrdd.map(lambda x: [(user_list.index(x[0][0]), user_list.index(x[0][1])), x[1]]).collect()

btwrdd.unpersist()
btwRDD.unpersist()
degreeRDD.unpersist()
graphRDD.unpersist()
graphRDD0.unpersist()
rdd.unpersist()


def getNewGraph(graph):  # removing the highest btw edge
    (n1, n2) = btw2[0][0]
    graph[n1][1].remove(n2)
    graph[n2][1].remove(n1)

    return graph


def getQ(partition):
    q = 0
    for pair in itertools.combinations(partition, 2):
        q += AKdic[tuple(sorted(pair))]
    return q


Q_dic = dict()
Q_list = [0]

for i in range(m):
    graph = getNewGraph(graph)  # removing the highest btw edge

    # [[[(e1, e2), btw1]...], group1]
    btwRDD = sc.parallelize(graph).map(lambda x: getBTW(x)).cache()

    btw2 = btwRDD.map(lambda x: x[0]) \
        .flatMap(lambda x: x) \
        .reduceByKey(lambda a, b: a + b) \
        .map(lambda x: [x[0], x[1] / 2]) \
        .sortBy(lambda x: x[1], False).collect()

    # x[1]: set
    group = btwRDD.map(lambda x: tuple(sorted(x[1]))).distinct() \
        .sortBy(lambda x: x).collect()

    # get modularity since new group
    # Q_dic={(partition1): q1, (partition2): q2 }
    if group != partitions[-1]:
        partitions.append(group)

        Q = 0
        for partition in group:
            if len(partition) != 0:
                t_part = tuple(partition)
                if t_part in Q_dic:
                    Q += Q_dic[t_part]

                else:
                    q = getQ(partition)
                    Q_dic.update({t_part: q})
                    Q += q

        Q_list.append(Q)

        if len(Q_list) - Q_list.index(max(Q_list)) > 15:
            break


def getUID(indexList):
    return sorted([user_list[index] for index in indexList])


result = sc.parallelize(partitions[Q_list.index(max(Q_list))]) \
    .map(lambda x: getUID(x)) \
    .sortBy(lambda x: x) \
    .sortBy(lambda x: len(x)).collect()

f = open(btw_output_file, mode="w+")
for item in btw:
    f.write(str(item).strip("[]"))
    f.write("\n")

f.close()

f2 = open(cmm_output_file, mode="w+")
for item in result:
    f2.write(str(item).strip("[]"))
    f2.write("\n")

f2.close()

print("Duration: %d" % int(time.time() - start_time))
