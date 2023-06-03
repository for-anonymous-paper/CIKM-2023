#Filename:	stat.py
#Author:	Wang Yongjie
#Email:		yongjie.wang@ntu.edu.sg
#Date:		Jum 12 Agu 2022 09:27:25  WIB

import numpy as np
import matplotlib
import brewer2mpl
import json
import sys
from scipy.spatial.distance import cdist
from scipy.spatial.distance import euclidean


def HausdorffScore(cfs1, cfs2):
    """
    compute the modified Hausdorff score for measuring the consistency between two sets of counterfactual explanations.
    Arguments:
        cfs1: the first set of counterfactual explanations.
        cfs2: the second set of counterfactual explanations.
    returns: 
        modified Hausdorff distance between two sets.
    """
    cfs1, cfs2 = np.array(cfs1), np.array(cfs2)
    pairwise_distance = cdist(cfs1, cfs2)
    h_A_B = pairwise_distance.min(1).mean()
    h_B_A = pairwise_distance.min(0).mean()

    return max(h_A_B, h_B_A)

def stat(filename):
    with open(filename, 'r') as f:
        data = json.load(f)
    
    distance = []
    proximity = []
    sparsity = []
    aps = []

    num = data['num']
    cfs = data['cf']
    for i in range(len(cfs)):
        print(i)
        cf_list = cfs[i]
        cf_data_list = []
        cf2_data_list = []
        for j in range(len(cf_list)):
            cf = cf_list[j]
            cf_data_list.extend(cf['cf'])
            cf2_data_list.extend(cf['cf2'])
            proximity.append(cf['proximity'])
            sparsity.append(cf['sparsity'])
            aps.append(cf['aps'])

        if len(cf_list) > 0:
            distance.append(HausdorffScore(cf_data_list, cf2_data_list))

    return np.mean(distance), np.mean(proximity), np.mean(sparsity), np.mean(aps)

def stat_cfmss(filename):
    with open(filename, 'r') as f:
        data = json.load(f)

    distance = []
    proximity = []
    sparsity = []
    aps = []

    num = data['num']
    cfs = data['cf']
    cf2 = data['cf2']
    for i in range(len(cfs)):
        print(i)
        cf_list = cfs[i]
        cf_data_list = []
        for j in range(num[i]):
            cf = cf_list[j]
            cf_data_list.extend(cf['cf'])
            proximity.append(cf['proximity'])
            sparsity.append(cf['sparsity'])
            aps.append(cf['aps'])
        cf2_data_list = cf2[i]
        distance.append(HausdorffScore(cf_data_list, cf2_data_list))

    return np.mean(distance), np.mean(proximity), np.mean(sparsity), np.mean(aps)

if __name__ == "__main__":
    filename = "./Hepatitis_cfmss.json"
    print(stat_cfmss(filename))
