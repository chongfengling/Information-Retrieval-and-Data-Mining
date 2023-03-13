# implementation of evaluation metrics and data representation for three models
import pandas as pd
import numpy as np
from tqdm import tqdm


def mAP(df: pd.DataFrame):
    """get mean AP for a set of retrieved passages of some queries

    Parameters
    ----------
    df : pd.DataFrame
        columns = ['qid','pid','score', 'relevancy']
            'qid': int
            'pid': int
            'score': float
            'relevancy': float, 0.0 (non-relevant) or 1.0 (relevant)
        top n passages retrieved by BM25 for each query. passages are listed in order.
    """
    # record the rank for passages of each query
    df['rank'] = df.groupby('qid').cumcount() + 1
    # count the relevant passaged retrieved so far
    df['index_of_relevancy'] = df.groupby('qid')['relevancy'].cumsum()
    # compute the precision
    df['precision'] = df['relevancy'] * df['index_of_relevancy'] / df['rank']
    # * replace for mean() operation
    df['precision'].replace(0.0, np.nan, inplace=True)
    # compute average precision for each query
    AP_df = df.groupby('qid')['precision'].mean()
    # ! none of relevant passages retrieved, at this time we set the precision of this query be 0
    AP_df.replace(np.nan, 0.0, inplace=True)
    # save the result to evalation
    df.to_csv('tmp.csv')
    AP_df.to_csv('tmp1.csv')
    # return the mAP
    return AP_df.mean()


def DCG(df: pd.DataFrame):
    """get DCG for a set of retrieved passages of some queries

    Parameters
    ----------
    df : pd.DataFrame
        _description_
    """
    # record the rank for passages of each query
    df['rank'] = df.groupby('qid').cumcount() + 1
    df['gain'] = 2 ** df['relevancy'] - 1
    df['discount'] = np.log2(1 + df['rank'])
    # DCG for each passage of a query
    df['DCG_p'] = df['gain'] / df['discount']
    # DCG of queries (evaluation)
    df_DCG = df.groupby('qid')['DCG_p'].apply(sum)
    # DCG of queries (perfect)
    return df_DCG


if __name__ == '__main__':
    BM25_top100_df = pd.read_csv(
        '0084_CW2/bm25_ordered.csv', names=['qid', 'pid', 'score', 'relevancy']
    )
    print(mAP(BM25_top100_df))  # 0.23475818881692506
