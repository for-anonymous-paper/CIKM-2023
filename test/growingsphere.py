#Filename:	growingsphere.py
#Author:	Wang Yongjie
#Email:		yongjie.wang@ntu.edu.sg
#Date:		Sen 18 Apr 2022 08:21:20 

import init
import pandas as pd
import numpy as np
from util.nn_model import NNModel
from cf.growingsphere import GrowingSphere

if __name__ == "__main__":

    dataset = pd.read_csv("../data/synthetic/synthetic_data_simple.csv")
    dataset = dataset.to_numpy().astype(np.float32)
    lens = len(dataset)
    train, val, test = dataset[0:int(lens * 0.5), ], dataset[int(lens * 0.5):int(lens * 0.75), ], dataset[int(lens*0.75):, ]
    train_x, train_y = train[:, 0:4], train[:, 4:5]
    val_x, val_y = val[:, 0:4], val[:, 4:5]
    test_x, test_y = test[:, 0:4], test[:, 4:5]
    idx = np.where(test_y == 0)[0]
    model = NNModel('../train/synthetic_model.pt')
    pred_y = model.predict(test_x)
    idx1 = np.where(pred_y == 0)[0]
    tp_idx = set(idx).intersection(idx1)
    abnormal_test = test_x[list(tp_idx)]
    target = 1
    eta = 5
    gs = GrowingSphere(target, model)

    for i in range(0, 50):
        print(str(i) *30)
        input_x = abnormal_test[i:i+1]
        cf = gs.generate_counterfactual(input_x, eta)
        print(input_x)
        print(cf)
    
