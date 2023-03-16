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


def text_preprocess(
    data: pd.DataFrame, save_name: str = None, do_subsample: bool = True
) -> pd.DataFrame:
    """from texts to vectors

    Parameters
    data : pd.DataFrame
        dataset
    save_name : str, optional
        save or not, by default None
    do_subsample: bool, optional
        subsample for train dataset only

    Returns
    -------
    pd.DataFrame
        the whole embedding dataset with columns []
    """
    # sample: subsample
    print('Estimated Time: about 3 mins on M1 Pro')
    if do_subsample:
        print("Start sampling ...")
        sampled_df = subsampling(data).reset_index(drop=False)
    else:
        sampled_df = data.reset_index(drop=False)
    # tokenisation of query and passage text
    print("Start tokenisation ...")
    tqdm.pandas(desc='Tokenisation(1/2) ...')
    sampled_query = sampled_df['queries'].progress_apply(tokenisation, remove=True)
    tqdm.pandas(desc='Tokenisation(2/2) ...')
    sampled_passage = sampled_df['passage'].progress_apply(tokenisation, remove=True)
    # build model
    print("Start building model ...")
    query_model = Word2Vec(
        sentences=sampled_query, vector_size=100, window=5, min_count=1, workers=4
    )
    passage_model = Word2Vec(
        sentences=sampled_passage, vector_size=100, window=5, min_count=1, workers=4
    )
    # average embedding
    print("Start embedding ...")
    query_ae = average_embedding(model=query_model, sentences=sampled_query)
    passage_ae = average_embedding(model=passage_model, sentences=sampled_passage)
    # create new pd.dataFrame
    res_df = pd.concat(
        [
            sampled_df['index'],
            sampled_df['qid'],
            sampled_df['pid'],
            pd.DataFrame(
                list(zip([1] * len(query_ae), query_ae, passage_ae)),
                columns=['intercept', 'query_ae', 'passage_ae'],
            ),
            sampled_df['relevancy'],
        ],
        axis=1,
    )
    if save_name:
        # difficult to load (str to array)
        # res_df.to_csv(f'input_df_{save_name}.csv')

        # save X, y into a .npy file
        X = np.asarray(
            res_df.apply(
                lambda x: np.hstack((x['intercept'], x['query_ae'], x['passage_ae'])),
                axis=1,
            ).values.tolist()
        )
        y = np.asarray(res_df['relevancy']).reshape(-1, 1)
        X_y = np.hstack((X, y))
        np.save(f'input_df_{save_name}.npy', X_y)

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
    list
        average embeddings of listx of tokens
    """
    res = []
    for i in tqdm(sentences, desc="Average Embedding ..."):
        # not empty
        if i:
            words_vectors = model.wv[i]
            res.append(np.mean(words_vectors, axis=0))
        # empty:
        else:
            res.append(np.zeros(model.vector_size))
    return res


if __name__ == '__main__':
    '''process and save to .npy files
    train_data = load_document(
        '/Users/ling/MyDocuments/COMP0084/0084_CW2/train_data.tsv', names=None
    )
    text_preprocess(train_data, save_name='train')

    val_data = load_document(
        '/Users/ling/MyDocuments/COMP0084/0084_CW2/validation_data.tsv', names=None
    )
    text_preprocess(val_data, save_name='val', do_subsample=False)
    '''

    # load from .npy file
    # intercept 1 + embedding 200 + label 1
    # shape = (95874, 202)
    Xy_train = np.load('input_df_train_sub.npy')
    # shape = (1103039, 202)
    Xy_val = np.load('input_df_val_nosub.npy')
    pass
