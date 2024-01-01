"""
Author: Ye Chen ^(=•ェ•=)^
Date: 2023-06-11
In this step, we change the dynamic network into a interaction streaming source.
eTILES format:
+ SNODE ENODE TIME
    + or - denotes if the edge is generated or vanished.
     SNODE and ENODE represents the start node and end node of the edge.
     TIME records the time when this events happens.
"""

import time
import numpy as np
import pandas as pd
from tqdm import trange
print("---------------STEP2---------------")
# 1. Create timestamp, eTILES requires UNIX timestamp.
"""
Notice: 
    The time interval of many gene expression experiment design is short. 
    Due to the best obs param, we treat the PPIN as a social network and Enlarge the time interval to one day.
"""
unixStamp = []
num = 12
initTime = int(time.time())
for i in range(num):
    unixStamp.append(initTime + 60 * 60 * 24 * i)
print("》》》》New time interval:")
print(unixStamp)

# 2. Create initialization status
# 【totalEvent】Store events that occur at each moment (edge addition, edge disappearance)
totalEvent = pd.DataFrame(columns=['action', 'sNode', 'eNode', 'unixStamp'])
# 【actualGraph】Store the real edges in the network at the current time
actualGraph = []

# Read initial state
inPath = "../data/Dynamic Network Demo/Output Dynamic Network/"
outPath = "../data/Dynamic Network Demo/"
initDf = pd.read_csv(inPath + 'T0.csv')

# Initialize real edges
actualGraph = list(zip(list(initDf['0']), list(initDf['1'])))
# Add initial state
totalEvent['sNode'] = list(initDf['0'])
totalEvent['eNode'] = list(initDf['1'])
totalEvent['unixStamp'] = [unixStamp[0]] * len(initDf)
totalEvent['action'] = ['+'] * len(initDf)

# Take the network for each time step
for i in trange(1, num):
    # Read previous and current graphs
    curr_path = inPath + 'T' + str(i) + '.csv'
    curr_df = pd.read_csv(curr_path, usecols=[1, 2])
    # All edges at this moment
    curr_status = list(zip(curr_df['0'], curr_df['1']))

    # action flag: Record whether the actual Graph exists at this moment
    action_flag = np.zeros((len(actualGraph),))
    # action index
    actual_index = np.arange(0, len(actualGraph), 1)

    # Store [Add Edge]
    new_edges = []
    # Store [vanish edge]
    vanish_edges = []

    # Traverse the graph at this moment
    for j in range(len(curr_status)):
        edge = curr_status[j]
        try:
            # If the entry exists, set to true
            edge_idx = actualGraph.index(edge)
            action_flag[edge_idx] = 1
        except ValueError as e:
            # If the entry does not exist, add a new edge
            new_edges.append(edge)

    # Obtain edges in action_flag that are still 0
    vanish_index = actual_index[action_flag == 0]
    print("============ TIME:%d ============" % (i + 1))
    print("======Compared to the actual graph, the vanish edge is:")
    print(len(vanish_index))
    print("======Compared to the actual graph, the generated edge is:")
    print(len(new_edges))

    for vid in vanish_index:
        vanish_edges.append(actualGraph[vid])

    # After comparison, add the status to the [totalEvent] at this time
    tempdf = pd.DataFrame(columns=['action', 'sNode', 'eNode', 'unixStamp'])
    tempdf['action'] = ['+'] * len(new_edges) + ['-'] * len(vanish_edges)
    tempdf['unixStamp'] = [unixStamp[i]] * (len(new_edges) + len(vanish_edges))

    if len(new_edges) == 0:
        print("There is no new edge added at this moment")
        new_sNode, new_eNode = (), ()
    else:
        new_sNode, new_eNode = zip(*new_edges)

    if len(vanish_edges) == 0:
        print("There is no disappearing edge at this moment")
        vanish_sNode, vanish_eNode = (), ()
    else:
        vanish_sNode, vanish_eNode = zip(*vanish_edges)

    tempdf['sNode'] = new_sNode + vanish_sNode
    tempdf['eNode'] = new_eNode + vanish_eNode

    # After comparison, add the temporary status to the [totalEvent]
    totalEvent = totalEvent.append(tempdf, ignore_index=True)

    # 1 Delete disappearing edges
    for eg in vanish_edges:
        actualGraph.remove(eg)

    # 2 Add new edges
    actualGraph = actualGraph + new_edges

totalEvent['sNode'] = totalEvent['sNode'].astype(int)
totalEvent['eNode'] = totalEvent['eNode'].astype(int)
totalEvent.to_csv(outPath + "streaming source.tsv", sep='\t', index=False, header=False)

# After generating the streaming source, we are ready to use the eTILES.
