#Filename:	plot.py
#Author:	Wang Yongjie
#Email:		yongjie.wang@ntu.edu.sg
#Date:		Rab 04 Mei 2022 10:25:29 

import matplotlib.pyplot as plt
import json
import sys
import numpy as np
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
        cf_list = cfs[i]
        cf_data_list = []
        cf2_data_list = []
        for j in range(num[i]):
            cf = cf_list[j]
            cf_data_list.extend(cf['cf'])
            cf2_data_list.extend(cf['cf2'])
            proximity.append(cf['proximity'])
            sparsity.append(cf['sparsity'])
            aps.append(cf['aps'])

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

    thyroid_gs = "../../cfgen/thyroid/thyroid_growingsphere.json"
    thyroid_cfproto = "../../cfgen/thyroid/Thyroid_cfproto.json"
    thyroid_plaincf = "../../cfgen/thyroid/thyroid_plaincf.json"
    thyroid_dice = "../../cfgen/thyroid/thyroid_dice.json"
    thyroid_cfmss = "../../cfgen/thyroid/thyroid_cfmss.json"
    
    distance_gs, proximity_gs, sparsity_gs, aps_gs = stat(thyroid_gs)
    distance_cfproto, proximity_cfproto, sparsity_cfproto, aps_cfproto = stat(thyroid_cfproto)
    distance_plaincf, proximity_plaincf, sparsity_plaincf, aps_plaincf = stat(thyroid_plaincf)
    distance_dice, proximity_dice, sparsity_dice, aps_dice = stat(thyroid_dice)
    distance_cfmss, proximity_cfmss, sparsity_cfmss, aps_cfmss = stat_cfmss(thyroid_cfmss)
    
    xaxis_label = ["GS", "CFProto", "PlainCF", "DiCE", "CFMSS"]


    plt.rcParams["figure.figsize"] = (20,5)
    fig, axs = plt.subplots(1, 4)

    x_tick = np.arange(5)

    axs[0].bar(x_tick,  [distance_gs, distance_cfproto, distance_plaincf, distance_dice, distance_cfmss], color=(0.2,0.4,0.6,0.6))
    axs[0].set_title(r"Consistency$\downarrow$")
    axs[0].set_xticks(x_tick, xaxis_label)

    axs[1].bar(x_tick, [sparsity_gs, sparsity_cfproto, sparsity_plaincf, sparsity_dice, sparsity_cfmss], color=(0.2,0.4,0.6,0.6))
    axs[1].set_title(r"Sparsity$\uparrow$")
    axs[1].set_xticks(x_tick, xaxis_label)

    axs[2].bar(x_tick, [proximity_gs, proximity_cfproto, proximity_plaincf, proximity_dice, proximity_cfmss], color=(0.2,0.4,0.6,0.6))
    axs[2].set_title(r"Proximity$\downarrow$")
    axs[2].set_xticks(x_tick, xaxis_label)

    axs[3].bar(x_tick, [aps_gs, aps_cfproto, aps_plaincf, aps_dice, aps_cfmss], color=(0.2,0.4,0.6,0.6))
    axs[3].set_title(r"APS$\downarrow$")
    axs[3].set_xticks(x_tick, xaxis_label)
    
    plt.savefig("thyroid.pdf", dpi = 600, bbox_inches = 'tight')
    plt.show()

