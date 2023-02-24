from task1 import tokenisation, occurrence_counter
from task2 import load_document, load_terms, II_counts
import pandas as pd
import numpy as np
from tqdm import tqdm
import nltk

def IDF(document: pd.DataFrame, terms: list):
    """Inverse Document Frequency of terms in a document collection
    IDF_t = log (N / n_t)
        t: term t
        N: number of documents in collection (what's collection?)
        n_t: number of documents in which term t appears

    Parameters
    ----------
    document : pd.DataFrame
        candidate-passages-top1000.tsv
    terms : list
        saved terms (stopwords removed or kept), unique      

    Returns
    -------
    np.array
        IDF of corresponding terms
    """
    N = len(document['pid'].unique())
    II_counts_dict = II_counts(terms=terms, document=document)
    n_terms = []
    for _, counts in II_counts_dict.items():
        n_terms.append(len(counts)) # occurrence, not frequency
    n_terms = np.asarray(n_terms) # temrs - n_terms, one to one
    IDF_ts = np.log10(N / n_terms)
    return IDF_ts

def select_top_passages(df_raw:pd.DataFrame, save_raw:bool=True, save_top:bool=True):
    # name the output files by current time
    import datetime
    now = datetime.datetime.now()
    H_M = now.strftime('%H_%M')

    df_top100 = df_raw.sort_values(by='score', ascending=False).groupby('qid').apply(lambda x: x.nlargest(100, columns='score')).reset_index(drop=True)

    if save_raw: df_raw.to_csv(f'TFIDF_{H_M}.csv', header=False, index=False)
    if save_top: df_top100.to_csv(f'TFIDF_TOP100_{H_M}.csv', header=False, index=False)
