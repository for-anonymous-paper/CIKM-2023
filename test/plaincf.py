#Filename:	plaincf.py
#Author:	Wang Yongjie
#Email:		yongjie.wang@ntu.edu.sg
#Date:		Sen 18 Apr 2022 07:22:40 

import init
import pandas as pd
import numpy as np
from util.nn_model import NNModel
from cf.plaincf import PlainCF

if __name__ == "__main__":

    dataset = pd.read_csv("../data/synthetic/synthetic_data_simple.csv")
    dataset = dataset.to_numpy().astype(np.float32)
    lens = len(dataset)
    train, val, test = dataset[0:int(lens * 0.5), ], dataset[int(lens * 0.5):int(lens * 0.75), ], dataset[int(lens*0.75):, ]
    train_x, train_y = train[:, 0:4], train[:, 4:5]
    val_x, val_y = val[:, 0:4], val[:, 4:5]
    test_x, test_y = test[:, 0:4], test[:, 4:5]
    idx = np.where(test_y == 0)[0]
    abnormal_test = test_x[idx]

    model = NNModel('../train/synthetic_model_simple.pt')
    target = 0.5
    pcf = PlainCF(target, model)
    _lambda = 5
    optimizer = 'adam'
    lr = 0.001

    for i in range(10):
        input_x = abnormal_test[i:i+1]
        cf = pcf.generate_counterfactuals(input_x, _lambda, optimizer, lr)
        print(cf)
        
