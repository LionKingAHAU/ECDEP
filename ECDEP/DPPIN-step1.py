"""
Author: Ye Chen ^(=•ェ•=)^
Date: 2023-05-30
In this step, we utilize 3-sigma rule, gene expression and PPI network to generate dynamic network.
"""

# import the packages needed
import numpy as np
import pandas as pd
from tqdm import trange

# You can change the path to your own configs
inPath = "../data/Dynamic Network Demo/Input Data/"
outPath = "../data/Dynamic Network Demo/Output Dynamic Network/"
print("---------------STEP1---------------")

def algoAver(x):
    """
    :param x: an array
    :return: the arithmetic mean of x
    """
    return sum(x) / len(x)


def normalization(x):
    MAX = x.ravel().max()
    x = x / MAX
    x -= x.mean(axis=0)
    return x


def getThreshold(v):
    """
    Threshold function utilizing 3-sigma
    :arg:
        v(int): an array of gene expression data on a certain time step.
    :return:
        int: active_threshold, the value to
    """
    μv = algoAver(v)
    s = 0
    for i in v:
        s = s + pow((i - μv), 2)
    sigma2v = s / (len(v) - 1)
    sigmav = np.sqrt(sigma2v)
    # Fv reflects the fluctuation of expression curve
    Fv = 1 / (1 + sigma2v)
    # active Threshold assesses if a protein is active in this time step
    aThreshold = μv + 3 * sigmav * (1 - Fv)
    return aThreshold


if __name__ == '__main__':
    # 1. Import data
    gse = normalization(np.load(inPath + "gse.npy"))
    ppi = pd.read_csv(inPath + "staticPPI.txt", sep="\t", header=None)

    """
    In our present demo, we use GSE3431 and Krogan Yeast PPI network.
    GSE3431 concludes three consecutive metabolic cycles, every metabolic cycle holds 12 time steps.
    Thus we calculate the mean as the final expression data. 
    """
    mean_gse = np.zeros((len(gse), 12))
    for node in range(len(gse)):
        temp = gse[node]
        num = len(temp)
        temp = sum(temp)
        mean = temp / num
        mean_gse[node] = mean

    # 2. Calculate the active threshold
    active_threshold_list = []
    print("》》》》》》》》》》Initiating calculation of the active threshold...")
    for v in trange(len(mean_gse)):
        temp = mean_gse[v]
        active_threshold = getThreshold(temp)
        active_threshold_list.append(active_threshold)

    # 3. Use threshold to determine in every time step
    print("》》》》》》》》》》Computing the active matrix...")
    active_matrix = np.zeros((12, len(mean_gse)))
    for i in trange(12):
        time = mean_gse[:, i]
        aThreshold = active_threshold_list[i]
        for j in range(len(time)):
            if time[j] > aThreshold:
                active_matrix[i][j] = 1

    # 4. use active matrix to generate the temporal PPIN
    print("》》》》》》》》》》Unleashing to generate the dynamic PPIN...")
    for time in trange(12):
        temp_df = pd.DataFrame()
        status = active_matrix[time]
        for line in range(len(ppi)):
            start = ppi.iloc[line][0]
            end = ppi.iloc[line][1]
            if status[start] == 1.0 and status[end] == 1.0:
                temp_df = temp_df.append(ppi.iloc[line], ignore_index=True)
        temp_df.to_csv(outPath + "T" + str(time) + ".csv")

# Now we have a series slices of PPIN, we need to make it fit the input requirement of eTILES algorithm, please head
# to streaming-step2.py
