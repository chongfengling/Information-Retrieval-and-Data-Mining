import pandas as pd
import numpy as np
import xgboost as xgb
from BM25 import select_top_passages
from task1 import mAP, mNDCG
from task2 import pred_test


def feature_process(file_path):
    """add new features"""
    # shape = (n, 204)
    array = np.load(file_path)
    qid, pid = array[:, 0], array[:, 1]
    qid_vec, pid_vec = array[:, 3:103], array[:, 103:203]
    relevancy = array[:, -1]
    similarity = np.sum(qid_vec * pid_vec, axis=1) / (
        np.linalg.norm(qid_vec, axis=1) * np.linalg.norm(pid_vec, axis=1)
    )
    features = np.concatenate(
        (similarity.reshape(-1, 1), qid_vec, pid_vec), axis=1
    )  # shape = (n, 201)
    return qid, pid, features, relevancy


def parameter_tuning(
    Xy_train_qid,
    Xy_train_data,
    Xy_train_label,
    Xy_val_qid,
    Xy_val_pid,
    Xy_val_data,
    Xy_val_label,
):
    df_res = pd.DataFrame(
        [], columns=['eta', 'max_depth', 'n_estimators', 'mAP', 'mNDCG']
    )
    etas = [5e-4, 1e-3, 5e-3, 1e-2, 5e-2]
    max_depths = [5, 6, 7]
    n_estimators_lst = [100, 200, 300]

    # train the model with different parameters
    for eta in etas:
        for max_depth in max_depths:
            for n_estimators in n_estimators_lst:
                model = xgb.XGBRanker(
                    booster='gbtree',
                    objective='rank:ndcg',
                    eta=eta,
                    max_depth=max_depth,
                    n_estimators=n_estimators,
                )
                print(
                    f'Model: LambdaMART with: eta: {eta}, max_depth: {max_depth}, n_estimators: {n_estimators}'
                )
                model.fit(
                    X=Xy_train_data,
                    y=Xy_train_label,
                    qid=Xy_train_qid,
                )
                y_val_pred = model.predict(Xy_val_data)

                df_val = pd.DataFrame(
                    {
                        'qid': Xy_val_qid,
                        'pid': Xy_val_pid,
                        'relevancy': Xy_val_label,
                        'score': y_val_pred,
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
                df_res = pd.concat(
                    [
                        df_res,
                        pd.DataFrame(
                            {
                                'eta': eta,
                                'max_depth': max_depth,
                                'n_estimators': n_estimators,
                                'mAP': mAP_value,
                                'mNDCG': mNDCG_value,
                            },
                            index=[0],
                        ),
                    ],
                    ignore_index=True,
                )
                # print the results
                print(
                    f'mAP: {mAP_value}, mNDCG: {mNDCG_value}, n_estimators: {n_estimators}'
                )
                print('----------------------------------------')
                print(df_res)


if __name__ == "__main__":

    # construct the training, validation and testing datasets
    [Xy_train_qid, Xy_train_pid, Xy_train_data, Xy_train_label] = feature_process(
        'input_df_train.npy'
    )
    [Xy_val_qid, Xy_val_pid, Xy_val_data, Xy_val_label] = feature_process(
        'input_df_val.npy'
    )
    [Xy_test_qid, Xy_test_pid, Xy_test_data, Xy_test_label] = feature_process(
        'input_df_test.npy'
    )
    # hyper-parameter tuning
    '''
    parameter_tuning(
        Xy_train_qid,
        Xy_train_data,
        Xy_train_label,
        Xy_val_qid,
        Xy_val_pid,
        Xy_val_data,
        Xy_val_label,
    )
    '''
    # Predict the test data
    print('start predicting the test data')
    LambdaMART_model = xgb.XGBRanker(
        booster='gbtree',
        objective='rank:ndcg',
        eta=0.0050,
        max_depth=5,
        n_estimators=300,
    )
    print(f'Model: LambdaMART with: eta: {0.005}, max_depth: {5}, n_estimators: {300}')
    LambdaMART_model.fit(
        X=Xy_train_data,
        y=Xy_train_label,
        qid=Xy_train_qid,
    )
    pred_test(model=LambdaMART_model, model_type='LM', X_test=Xy_test_data, qid_test=Xy_test_qid, pid_test=Xy_test_pid)

