import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import xgboost as xgb
from BM25 import tokenisation
from task1 import mAP, mNDCG

if __name__=="__main__":
    # construct the training, validation and testing datasets
    Xy_train_arr = np.load('input_df_train_sub.npy')
    Xy_train = xgb.DMatrix(Xy_train_arr[:, 2:-1], label=Xy_train_arr[:, -1])
    Xy_train_qid, Xy_train_pid = Xy_train_arr[:, 0], Xy_train_arr[:, 1]

    Xy_val_arr = np.load('input_df_val_nosub.npy')
    Xy_val = xgb.DMatrix(Xy_val_arr[:, 2:-1], label=Xy_val_arr[:, -1])
    Xy_val_qid, Xy_val_pid = Xy_val_arr[:, 0], Xy_val_arr[:, 1]

    Xy_test_arr =  np.load('input_df_test_nosub.npy')
    Xy_test = xgb.DMatrix(Xy_test_arr[:, 2:-1], label=Xy_test_arr[:, -1])
    Xy_test_qid, Xy_test_pid = Xy_test_arr[:, 0], Xy_test_arr[:, 1]

