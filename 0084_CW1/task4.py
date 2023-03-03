from task1 import tokenisation
from task2 import load_document, load_terms
from task3 import select_top_passages
import pandas as pd
import numpy as np
from tqdm import tqdm
import nltk
from itertools import chain
from collections import Counter


def query_llh_model(document: pd.DataFrame, query: pd.DataFrame, terms: list, eps: float = 0.1, mu=50, model: str = 'LS'):
    """P(Q|M_D) = \Pai (P(q|M_D), ...) for term q in the query Q and a language model M_D based on a document D

    Parameters
    ----------
    document : pd.DataFrame
        candidate-passages-top1000.tsv in pd.DataFrame, header = ['qid', 'pid', 'query', 'passage']
    terms : list
        saved terms (stopwords removed or kept), unique  
    eps : float, optional
        parameter in Lidstone correction, by default 0.1
    mu : int, optional
        parameter in Dirichlet smooth, by default 50
    model : str, optional
        choose one from ['LS', 'LC', 'DS] for Laplace Smooth, Lidstone Correction, Dirichlet Smooth, by default 'LS'
    """

    V_len = len(
        terms)  # number fo unique terms in the entire collection without/with stop words. (Vocabulary size)
    tqdm.pandas(desc='Tokenisation(1/2) ......')
    document['query_token'] = document['query'].progress_apply(
        tokenisation, remove=True)  # stop words removed
    tqdm.pandas(desc='Tokenisation(2/2) ......')
    document['passage_token'] = document['passage'].progress_apply(
        tokenisation, remove=True)  # stop words removed
    all_token_one_list = list(chain(*document['passage_token'].tolist()))
    # number of tokens in the document collection
    DD_len = len(all_token_one_list)
    # term with its frequency in the collection
    C_q = Counter(all_token_one_list)
    tmp_all = []
    for (qid, pid, query_token, passage_token) in tqdm(zip(document['qid'], document['pid'], document['query_token'], document['passage_token']), desc="Iterating passages"):
        D_len = len(passage_token)  # size of the passage
        Q_len = len(query_token)  # size of the query
        # frequency of query terms in the document d
        m_q = np.asarray([passage_token.count(term) for term in query_token])
        # frequency of query terms in the document collection D
        c_q = np.asarray([C_q[term] for term in query_token])
        if model == 'laplace':
            score = Laplace_smooth(m_q=m_q, D_len=D_len,
                                   V_len=V_len, Q_len=Q_len)
        elif model == 'lidstone':
            score = Lidstone_correction(
                m_q=m_q, D_len=D_len, V_len=V_len, eps=eps)
        elif model == 'dirichlet':
            score = Dirichlet_smooth(
                D_len=D_len, mu=mu, c_q=c_q, m_q=m_q, DD_len=DD_len)

        tmp_all.append([qid, pid, score])
    df_score = pd.DataFrame(tmp_all, columns=['qid', 'pid', 'score'], dtype=float).astype(
        dtype={'qid': int, 'pid': int, 'score': float})
    select_top_passages(df_raw=df_score, query=query,
                        filename=model, save_raw=False)


def Laplace_smooth(m_q: list, D_len: int, V_len: int, Q_len: int) -> float:
    """calculate score by laplace smooth algorithm

    Parameters
    ----------
    m_q : list
        frequency of query terms in the document d
    D_len : int
        size of the passage
    V_len : int
        size of vocabulary
    Q_len : int
        size of the query

    Returns
    -------
    float
        score in log
    """
    # terms occur in the query
    score_occurrences = np.sum(np.log((m_q + 1) / (D_len + V_len)))
    # terms absent in the query
    score_absent = np.log(1 / (D_len + V_len)) * (V_len - Q_len)
    return score_absent + score_occurrences


def Lidstone_correction(m_q: list, D_len: int, V_len: int, eps: float) -> float:
    """calculate score by Lidstone correction algorithm

    Parameters
    ----------
    m_q : list
        frequency of query terms in the document d
    D_len : int
        size of the passage
    V_len : int
        size of vocabulary
    eps : float
        parameter

    Returns
    -------
    float
        score in log
    """
    score = np.sum(np.log((m_q + eps) / (D_len + eps * V_len)))
    return score


def Dirichlet_smooth(m_q: list, c_q: list, D_len: int, DD_len: int, mu: float):
    """calculate score by Dirichlet Smooth

    Parameters
    ----------
    m_q : list
        frequency of query terms in the document d
    c_q : list
        frequency of query terms in the whole document collection
    D_len : int
        size of the passage
    DD_len : int
        size of the whole document collection
    mu : float
        parameter

    Returns
    -------
    _type_
        score in log
    """
    lam = D_len / (D_len + mu)
    score = np.sum(np.log(lam * m_q / D_len + (1 - lam) * c_q / DD_len))
    return score


if __name__ == '__main__':
    nltk.download('punkt')
    candidates = load_document(
        'candidate-passages-top1000.tsv')  # document collection D
    # terms in document collection D without stop words
    terms_removed = load_terms(file_path='terms_removed.txt')
    queries = load_document('test-queries.tsv',
                            names=['qid', 'query'])  # query collection Q

    query_llh_model(document=candidates, terms=terms_removed,
                    query=queries, model='laplace')
    query_llh_model(document=candidates, terms=terms_removed,
                    query=queries, model='lidstone')
    query_llh_model(document=candidates, terms=terms_removed,
                    query=queries, model='dirichlet')
