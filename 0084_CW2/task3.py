import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import xgboost as xgb
from BM25 import tokenisation
from task1 import mAP, mNDCG

if __name__=="__main__":
    # config the parameters
    etas = [0.1, 0.2, 0.3, 0.4, 0.5]
    max_depths = [3, 4, 5, 6, 7]
    num_boost_rounds = [10, 20, 30, 40, 50]
    objective = 'rank:pairwise'

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

    for eta in etas:
        for max_depth in max_depths:
            for num_boost_round in num_boost_rounds:
                params = {
                    'objective': objective,
                    'eta': eta,
                    'max_depth': max_depth,
                    'num_boost_round': num_boost_round
                }
                pass
