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

def TF_IDF(document: pd.DataFrame, query: pd.DataFrame, terms: list, IDF_ts: np.array, save_raw:bool=True, save_top:bool=True):
    """TF_t,d = Term frequency of term t in document d
    TF_t,d = 

    Parameters
    ----------
    document : pd.DataFrame
        _description_
    query : pd.DataFrame
        _description_
    terms : list
        _description_
    IDF_ts : np.array
        Inverse Document Frequency of terms in a document collection
    save_raw : bool
        save the raw TFIDF data or not
    save_top : bool
        save the sorted top100 TFIDF data or not
    """
    
    # II_counts_dict = II_counts(terms=terms, document=document)
    tmp = []
    # df_score = pd.DataFrame(names=['qid', 'pid', 'score'])
    for (qid, query) in tqdm(zip(query['qid'], query['query']), desc="Get TF-IDF for query"):
        qid_terms = set(tokenisation(query))
        passages = document.loc[document['qid'] == qid]
        common_term = [term for term in qid_terms if term in terms]
        for (pid, passage) in zip(passages['pid'], document['passage']):
            score = 0.0
            for term in common_term:
                IDF_t = IDF_ts[terms.index(term)]
                tf_t = passage.count(term)
                score += IDF_t * tf_t
            tmp.append([qid, pid, score])
    df_score = pd.DataFrame(tmp, columns =['qid', 'pid', 'score'], dtype = float).astype(dtype = {'qid': int, 'pid': int, 'score': float})
    select_top_passages(df_score, save_raw=save_raw, save_top=save_top)
    
def select_top_passages(df_raw:pd.DataFrame, save_raw:bool=True, save_top:bool=True):
    # name the output files by current time
    import datetime
    now = datetime.datetime.now()
    H_M = now.strftime('%H_%M')

    df_top100 = df_raw.sort_values(by='score', ascending=False).groupby('qid').apply(lambda x: x.nlargest(100, columns='score')).reset_index(drop=True)

    if save_raw: df_raw.to_csv(f'TFIDF_{H_M}.csv', header=False, index=False)
    if save_top: df_top100.to_csv(f'TFIDF_TOP100_{H_M}.csv', header=False, index=False)
