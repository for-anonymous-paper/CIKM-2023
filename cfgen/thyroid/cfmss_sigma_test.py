# Filename:	../test/cfmss.py
# Author:	Wang Yongjie
# Email:		yongjie.wang@ntu.edu.sg
# Date:		Kam 28 Apr 2022 10:05:00

import pandas as pd
import numpy as np
from numpyencoder import NumpyEncoder
import torch
import init
import json
from util.nn_model import NNModel
from cf.cfmss import *
from util.evaluator import *
from sklearn.preprocessing import StandardScaler

if __name__ == "__main__":

    dataset = pd.read_csv("../../data/thyroid/thyroid_dataset.csv")
    dataset = dataset.to_numpy().astype(np.float32)
    lens = len(dataset)
    train, test = dataset[0:int(lens * 0.7), ], dataset[int(lens * 0.7):, ]
    train_x, train_y = train[:, 0:4], train[:, 4]
    test_x, test_y = test[:, 0:4], test[:, 4]

    scaler = StandardScaler()
    train_x = scaler.fit_transform(train_x)
    test_x = scaler.transform(test_x)
    train_y = train_y.astype(np.int64)
    test_y = test_y.astype(np.int64)

    model = NNModel('../../train/thyroid/thyroid_data.pt')

    # obtain true positive set of test set
    idx = np.where(test_y == 0)[0]
    pred_y = model.predict(test_x)
    idx1 = np.where(pred_y == 0)[0]
    tn_idx = set(idx).intersection(idx1)
    abnormal_test = test_x[list(tn_idx)]

    # obtain true negative set of train set
    idx2 = np.where(train_y == 1)[0]
    pred_ty = model.predict(train_x)
    idx3 = np.where(pred_ty == 1)[0]
    tp_idx = set(idx2).intersection(idx3)
    normal_test = train_x[list(tp_idx)]

    # set the normal_range
    # normal_range = np.vstack([scaler.mean_- np.sqrt(scaler.var_), scaler.mean_])
    # normal_range = scaler.transform(normal_range).astype(np.float32)
    # normal_range = np.array([[10.3, 0.4, 0.92, 58],
    #    [35, 4.2, 2.78, 140]])
    normal_range = np.array([[85, 0.4, 0.92, 58],
                             [140, 4.2, 2.78, 140]])
    normal_range = scaler.transform(normal_range).astype(np.float32)

    # initialize the evaluator
    evaluator = Evaluator(train_x, normal_test)

    desired_pred = 1
    n = 4
    data_lists = []
    num_lists = []
    cf_lists = []
    cf2_lists = []
    # diversity_lists = []
    # diversity2_lists = []

    for i in range(len(abnormal_test)):
        input_x = abnormal_test[i:i + 1]
        to_replace = np.where(input_x < normal_range[0, :], normal_range[0, :], input_x)
        to_replace = np.where(to_replace > normal_range[1, :], normal_range[1, :], to_replace)
        mapsolver = MapSolver(n)
        cfsolver = CFSolver(n, model, input_x, to_replace, desired_pred=1)
        num_of_cf = 0
        cf_list = []
        for text, cf, mask in FindCF(cfsolver, mapsolver):
            # tmp_result = {}
            num_of_cf += 1
            # tmp_result['cf'] = cf
            # tmp_result['mask'] = mask
            # tmp_result['sparsity'] = evaluator.sparsity(input_x, cf)
            # tmp_result['aps'] = evaluator.average_percentile_shift(input_x, cf)
            # tmp_result['proximity'] = evaluator.proximity(cf)
            cf_list.extend(cf)
        cf_list = np.reshape(cf_list, (-1, input_x.shape[1]))
        print(str(i) * 2, num_of_cf)
        # random select a cf as target to move
        # idx = np.random.randint(0, num_of_cf, size = 1).item()
        # ratio = np.random.uniform(0, 0.2, 1).astype(np.float32)
        # new_input = input_x + ratio * (cf_list[idx]['cf'] - input_x)

        # 修改此处，改为添加高斯噪声
        from util.sigma_test_util import get_noised_cfs


        # 修改此处，改为添加高斯噪声
        def generate_cfs_func(idx, input_x):
            mapsolver1 = MapSolver(n)
            cfsolver1 = CFSolver(n, model, input_x, to_replace, desired_pred=1)
            _cfs = []
            for text, cf, mask in FindCF(cfsolver1, mapsolver1):
                _cfs.extend(cf)

            _cfs = np.reshape(_cfs, (-1, input_x.shape[1]))
            return _cfs


        cf2_list = get_noised_cfs(input_x, cf_list, [0.0001, 0.001, 0.01, 0.1], evaluator, generate_cfs_func)

        data_lists.append(input_x)
        num_lists.append(num_of_cf)
        cf_lists.append(cf_list)
        cf2_lists.append(cf2_list)
        # diversity_lists.append(diversity)
        # diversity2_lists.append(diversity2)

    results = {}
    results["data"] = data_lists
    results["num"] = num_lists
    results["cf"] = cf_lists
    results["cf2"] = cf2_lists
    # results['diversity'] = diversity_lists
    # results['diversity2'] = diversity2_lists

    with open("thyroid_cfmss_sigma.json", "w") as f:
        json.dump(results, f, cls=NumpyEncoder)

