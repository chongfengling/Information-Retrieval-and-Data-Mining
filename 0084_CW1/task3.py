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
