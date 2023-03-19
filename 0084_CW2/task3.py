import pandas as pd
import numpy as np
import xgboost as xgb
from BM25 import select_top_passages
from task1 import mAP, mNDCG

if __name__ == "__main__":
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

    Xy_test_arr = np.load('input_df_test_nosub.npy')
    Xy_test = xgb.DMatrix(Xy_test_arr[:, 2:-1], label=Xy_test_arr[:, -1])
    Xy_test_qid, Xy_test_pid = Xy_test_arr[:, 0], Xy_test_arr[:, 1]

    # construct the dataframe to save the results
    df_res = pd.DataFrame(
        [], columns=['eta', 'max_depth', 'num_boost_round', 'mAP', 'mNDCG']
    )
    # train the model with different parameters
    for eta in etas:
        for max_depth in max_depths:
            for num_boost_round in num_boost_rounds:
                params = {
                    'objective': objective,
                    'eta': eta,
                    'max_depth': max_depth,
                    'num_boost_round': num_boost_round,
                }
                model = xgb.train(
                    params=params,
                    dtrain=Xy_train,
                    evals=[(Xy_train, 'train'), (Xy_val, 'validation')],
                )
                y_pred = model.predict(Xy_test)

                df_val = pd.DataFrame(
                    {
                        'qid': Xy_val_qid,
                        'pid': Xy_val_pid,
                        'relevancy': Xy_val.get_label(),
                        'score': model.predict(Xy_val),
                    }
                )
                top_n = 100

                df_val = select_top_passages(
                    df_raw=df_val,
                    save_raw=False,
                    save_top=False,
                    filename='t3_test',
                    top_n=top_n,
                )
                # get the mAP and mNDCG
                mAP_value = mAP(df=df_val)
                mNDCG_value = mNDCG(BM25_topN_df=df_val, top_n=top_n)
                # save the results
                df_res = df_res.append(
                    {
                        'eta': eta,
                        'max_depth': max_depth,
                        'num_boost_round': num_boost_round,
                        'mAP': mAP_value,
                        'mNDCG': mNDCG_value,
                    },
                    ignore_index=True,
                )
                # print the results
                print(
                    f'LambdaMART with: eta: {eta}, max_depth: {max_depth}, num_boost_round: {num_boost_round}'
                )
                print(f'mAP: {mAP_value}, mNDCG: {mNDCG_value}')
    # save the results
    df_res.to_csv('t3_res.csv', index=False)
