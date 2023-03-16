from gensim.models import Word2Vec
from BM25 import load_document, tokenisation, select_top_passages
import nltk
from tqdm import tqdm
import pandas as pd
import numpy as np


class LogisticRegression:
    def __init__(self) -> None:
        pass

    def fit():
        pass

    def predict():
        pass


def text_preprocess(data:pd.DataFrame, save_name: str = None):
    """from texts to vectors

    Parameters
    ----------
    data : pd.DataFrame
        dataset
    save_name : str, optional
        save or not, by default None

    Returns
    -------
    pd.DataFrame
        embedding dataset
    """
    # sample: subsample
    sampled_df = subsampling(data).reset_index(drop=False)
    # tokenisation of query and passage text
    sampled_query = sampled_df['queries'].apply(tokenisation, remove=True)
    sampled_passage = sampled_df['passage'].apply(tokenisation, remove=True)
    # build model
    query_model = Word2Vec(
        sentences=sampled_query, vector_size=100, window=5, min_count=1, workers=4
    )
    passage_model = Word2Vec(
        sentences=sampled_passage, vector_size=100, window=5, min_count=1, workers=4
    )
    # average embedding
    query_ae = average_embedding(model=query_model, sentences=sampled_query)
    passage_ae = average_embedding(model=passage_model, sentences=sampled_passage)
    # create new pd.dataFrame
    res_df = pd.concat(
        [
            sampled_df['index'],
            sampled_df['qid'],
            sampled_df['pid'],
            pd.DataFrame(
                list(zip(query_ae, passage_ae)), columns=['query_ae', 'passage_ae']
            ),
            sampled_df['relevancy'],
        ],
        axis=1,
    )
    if save_name:
        res_df.to_csv(f'input_df_{save_name}.csv')
    return res_df


def subsampling(
    raw_df: pd.DataFrame = None,
    samples_n: int = 20,
    to_csv: bool = True,
    seed: int = None,
) -> pd.DataFrame:
    """sample from a big dataset. All positive samples remained, select samples_n negative samples for each label

    Parameters
    ----------
    raw_df : pd.DataFrame, optional
        big dataset, by default None
    samples_n : int, optional
        number of negative samples selected, by default 20
    to_csv : bool, optional
        if save the dataframe, by default True
    seed : int, optional
        random seed, by default None

    Returns
    -------
    pd.DataFrame
        the smaller dataset after sampling
    """
    if seed:
        np.random.seed(seed=seed)
    else:
        # works as  filename
        seed = 'None'
    print("Sarting sampling ...")
    # split positive and negative samples. sorted for iteration
    rel_df, irrel_df = raw_df[raw_df['relevancy'] == 1], raw_df[
        raw_df['relevancy'] == 0
    ].sort_values('qid')
    # get the number of negative samples for each label
    irrel_df_freq = irrel_df['qid'].value_counts()
    # iterates from the first row of raw_df
    index = 0
    # store the selected rows' index
    samples_index = []
    for i in irrel_df['qid'].unique():
        qid_num = irrel_df_freq[i]
        # no enough negative samples
        if qid_num <= samples_n:
            samples_index.extend(list(range(index, index + qid_num)))
        # selected samples randomly
        else:
            samples_index.extend(np.random.randint(index, index + qid_num, samples_n))
        # start at different index. raw_df sorted by qid
        index += qid_num
    # return selected negative samples
    irrel_sampled_df = irrel_df.iloc[samples_index]
    # concatenate two dataframe
    sampled_df = pd.concat([rel_df, irrel_sampled_df]).sort_index()
    # save or not
    if to_csv:
        sampled_df.to_csv(f'{seed}_sampled.csv')
    return sampled_df


def average_embedding(model, sentences):
    """average embedding for a list of tokens

    Parameters
    ----------
    model : Word2Vec model
        _description_
    sentences : list
        processed query/passage

    Returns
    -------
    np.array
        average embedding for a list of tokens
    """
    res = []
    for i in sentences:
        # not empty
        if i:
            words_vectors = model.wv[i]
            res.append(np.mean(words_vectors, axis=0))
        # empty:
        else:
            res.append(np.zeros(model.vector_size))
    return res


if __name__ == '__main__':

    train_data = load_document(
        '/Users/ling/MyDocuments/COMP0084/0084_CW2/train_data.tsv', names=None
    )

    train_processed_df = text_preprocess(train_data, save_name='train')
