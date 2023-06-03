#Filename:	test_xgb_api.py
#Author:	Wang Yongjie
#Email:		yongjie.wang@ntu.edu.sg
#Date:		Sel 14 Jun 2022 12:30:18 

import sys
sys.path.insert(0, "../")
from util.xgb_model import XGBModel
import pandas as pd
import numpy as np
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, accuracy_score, precision_recall_fscore_support

if __name__ == "__main__":

    data = pd.read_csv('../data/Hepatitis/HepatitisC_dataset_processed.csv')
    standard_sc = StandardScaler() 

    X = data.drop(['Category'],axis=1)
    y = data["Category"]
    train_x, test_x, train_y, test_y = train_test_split(X,y,test_size=0.3,random_state=42,stratify=y)

    train_x=standard_sc.fit_transform(train_x).astype(np.float32)
    test_x=standard_sc.transform(test_x).astype(np.float32)
    
    model = XGBClassifier()
    model.load_model("../train/Hepatitis/hepetitis_model.json")

    y_pred = model.predict(test_x)
    
    print(confusion_matrix(y_pred, test_y))
    pr, rc, fs, sup = precision_recall_fscore_support(test_y, y_pred, average='macro')
    res = {"Accuracy": round(accuracy_score(test_y, y_pred), 4),
                              "Precision": round(pr, 4), "Recall":round(rc, 4), "FScore":round(fs, 4)}
    print(res)

