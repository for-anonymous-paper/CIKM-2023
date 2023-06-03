#Filename:	dice.py
#Author:	Wang Yongjie
#Email:		yongjie.wang@ntu.edu.sg
#Date:		Sel 19 Apr 2022 06:34:13 

import init
import pandas as pd
import numpy as np
from util.nn_model import NNModel
from cf.dice import DiCE
from cf.cfproto import CFProto

if __name__ == "__main__":

    dataset = pd.read_csv("../data/synthetic/synthetic_data_simple.csv")
    dataset = dataset.to_numpy().astype(np.float32)
    lens = len(dataset)
    train, val, test = dataset[0:int(lens * 0.5), ], dataset[int(lens * 0.5):int(lens * 0.75), ], dataset[int(lens*0.75):, ]
    train_x, train_y = train[:, 0:4], train[:, 4:5]
    val_x, val_y = val[:, 0:4], val[:, 4:5]
    test_x, test_y = test[:, 0:4], test[:, 4:5]
    idx = np.where(test_y == 0)[0]
    model = NNModel('../train/synthetic/synthetic_model.pt')
    pred_y = model.predict(test_x)
    idx1 = np.where(pred_y == 0)[0]
    tn_idx = set(idx).intersection(idx1)
    abnormal_test = test_x[list(tn_idx)]

    idx2 = np.where(test_y == 1)[0]
    idx3 = np.where(pred_y == 1)[0]
    tp_idx = set(idx2).intersection(idx3)
    normal_test = test_x[list(tp_idx)]
    inital_points = normal_test.mean(0)[np.newaxis,:]

    prototype = np.array([[-0.2, -0.15, -0.1, -0.1]]).astype(np.float32)

    target = 0.8
    cfproto = CFProto(target, model)

    for i in range(10):
        print(str(i) * 30)
        input_x = abnormal_test[i:i+1]
        print(input_x)
        cf = cfproto.generate_counterfactuals(input_x, inital_points, prototype)
        print(model.predict(cf))
        print(cf)

