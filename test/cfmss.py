#Filename:	../test/cfmss.py
#Author:	Wang Yongjie
#Email:		yongjie.wang@ntu.edu.sg
#Date:		Kam 28 Apr 2022 10:05:00 

import pandas as pd
import numpy as np
import torch
import init
import json
from util.nn_model import NNModel
from cf.cfmss import *
from util.evaluator import *

#Local Test on Synthetic Dataset
if __name__ == "__main__":

    dataset = pd.read_csv("../data/synthetic/synthetic_data_simple.csv")
    dataset = dataset.to_numpy().astype(np.float32)
    lens = len(dataset)
    train, val, test = dataset[0:int(lens * 0.5), ], dataset[int(lens * 0.5):int(lens * 0.75), ], dataset[int(lens*0.75):, ]
    train_x, train_y = train[:, 0:4], train[:, 4:5]
    val_x, val_y = val[:, 0:4], val[:, 4:5]
    test_x, test_y = test[:, 0:4], test[:, 4:5]

    model = NNModel('../train/synthetic/synthetic_model_simple.pt')

    # obtain true negative set of test set
    idx = np.where(test_y == 0)[0]
    pred_y = model.predict(test_x)
    idx1 = np.where(pred_y == 0)[0]
    tn_idx = set(idx).intersection(idx1)
    abnormal_test = test_x[list(tn_idx)]

    # obtain true positive set of train set
    idx2 = np.where(train_y == 1)[0]
    pred_ty = model.predict(train_x)
    idx3 = np.where(pred_ty == 1)[0]
    tp_idx = set(idx2).intersection(idx3)
    normal_test = train_x[list(tp_idx)]
    
    # set the value to replace
    #to_replace = np.array([[-0.35, -0.25, -0.2, -0.2]]).astype(np.float32)
    to_replace = np.array([[-0.2, -0.15, -0.1, -0.1]]).astype(np.float32)
    
    # initialize the evaluator
    evaluator = Evaluator(train_x, normal_test)

    desired_pred = 1
    n = 4
    data_lists = []
    num_lists = []
    cf_lists = []
    for i in range(len(abnormal_test)):
        input_x = abnormal_test[i:i+1]
        mapsolver = MapSolver(n)
        cfsolver = CFSolver(n, model, input_x, to_replace, desired_pred = 1)
        num_of_cf = 0
        cf_list = []
        for text, cf, mask in FindCF(cfsolver, mapsolver):
            tmp_result = {}
            num_of_cf += 1
            tmp_result['cf'] = cf
            tmp_result['mask'] = mask
            tmp_result['sparsity'] = evaluator.sparsity(input_x, cf)
            tmp_result['aps'] = evaluator.average_percentile_shift(input_x, cf)
            tmp_result['proximity'] = evaluator.proximity(cf)
            cf_list.append(tmp_result)
        
        data_lists.append(input_x)
        num_lists.append(num_of_cf)
        cf_lists.append(cf_list)
        break

    results = {}
    results["data"] = data_lists
    results["num"] = num_lists
    results["cf"] = cf_lists
    
    with open("synthetic_cfmss.json", "w") as f:
        json.dump(result, f)
