# implementation of evaluation metrics and data representation for three models
import pandas as pd
import numpy as np
from tqdm import tqdm
from BM25 import select_top_passages


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
    """get Discounted Cumulative Gain for a set of retrieved passages of some queries

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


def mNDCG(BM25_topN_df: pd.DataFrame, top_n: int):
    """get mean Normalized DCG

    Parameters
    ----------
    BM25_topN_df : pd.DataFrame
        BM25 result
    top_n : int
        top n

    Returns
    -------
    _type_
        _description_
    """
    # load the whole validation data
    val_df = pd.read_csv('0084_CW2/validation_data.tsv', sep='\t')
    # delete unused save memory
    del val_df['queries']
    del val_df['passage']
    # select top n passages for each query in the given validation data ranked by relevancy.
    val_opt = select_top_passages(
        df_raw=val_df,
        save_raw=False,
        save_top=False,
        top_n=top_n,
        filename='opt_retrieval',
        group_col='qid',
        rank_col='relevancy',
    )
    # DCG
    DCG_BM25 = DCG(BM25_topN_df)
    # idea DCG
    IDCG = DCG(val_opt)
    # NDCG
    NDCG = DCG_BM25 / IDCG
    # return mean NDCG
    return np.mean(NDCG)


if __name__ == '__main__':
    BM25_top3_df = pd.read_csv(
        '0084_CW2/bm25_ordered_top3.csv', names=['qid', 'pid', 'score', 'relevancy']
    )
    BM25_top10_df = pd.read_csv(
        '0084_CW2/bm25_ordered_top10.csv', names=['qid', 'pid', 'score', 'relevancy']
    )
    BM25_top100_df = pd.read_csv(
        '0084_CW2/bm25_ordered_top100.csv', names=['qid', 'pid', 'score', 'relevancy']
    )

    # print(mAP(BM25_top100_df))  # 0.23475818881692506

    print(mNDCG(BM25_top3_df, top_n=3))  # top003 = 0.19853070832150987
    print(mNDCG(BM25_top10_df, top_n=10))  # top010 = 0.28584393775886474
    print(mNDCG(BM25_top100_df, top_n=100))  # top100 = 0.35337428212970406
