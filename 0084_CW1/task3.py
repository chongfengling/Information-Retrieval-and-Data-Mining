from task1 import tokenisation, occurrence_counter
from task2 import load_document, load_terms, II_counts, II_simple
import pandas as pd
import numpy as np
from tqdm import tqdm
import nltk
from collections import Counter


def IDF(document: pd.DataFrame, terms: list):
    """Inverse Document Frequency of terms in a document collection
    IDF_t = log (N / n_t)
        t: term t
        N: number of documents in collection (what's collection?)
        n_t: number of documents in which term t appears

    Parameters
    ----------
    document : pd.DataFrame
        candidate-passages-top1000.tsv in pd.DataFrame, header = ['qid', 'pid', 'query', 'passage']
    terms : list
        saved terms (stopwords removed or kept), unique      

    Returns
    -------
    np.array
        IDF of corresponding terms
    """
    N = len(document['pid'].unique()) # should be unique?
    II_counts_dict = II_counts(terms=terms, document=document)
    n_terms = []
    for _, counts in II_counts_dict.items():
        n_terms.append(len(counts))  # occurrence, not frequency
    n_terms = np.asarray(n_terms)  # temrs - n_terms, one to one
    IDF_ts = np.log10(N / n_terms)
    return IDF_ts


def TF_IDF(document: pd.DataFrame, query: pd.DataFrame, terms: list, IDF_ts: np.array, save_raw: bool = True, save_top: bool = True):
    """TF_t,d = Term frequency of term t in document d
    cosine_score = cos (TF_IDF)

    Parameters
    ----------
    document : pd.DataFrame
        candidate-passages-top1000.tsv in pd.DataFrame, header = ['qid', 'pid', 'query', 'passage']
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
    # length of idf
    IDF_ts_length = np.linalg.norm(IDF_ts)
    # df_score = pd.DataFrame(names=['qid', 'pid', 'score'])
    for (qid, q) in tqdm(zip(query['qid'], query['query']), desc="Computing TF-IDF for 200 queries"):
        qid_terms = set(tokenisation(q, remove=False)) # unique terms in a query
        passages = document.loc[document['qid'] == qid] # passages answering to the qid. less than 1000
        common_term = [term for term in qid_terms if term in terms] # query terms in the whole (stopwords removed) terms
        for (pid, passage) in zip(passages['pid'], passages['passage']):
            score = 0.0
            tf_ts = [] # list of IDF of terms in the common_term
            for term in common_term: # t in (query q and document d)
                IDF_t = IDF_ts[terms.index(term)] # get the IDF of the term by index
                tf_t = passage.count(term)
                score += IDF_t * tf_t
                tf_ts.append(tf_t)
            tf_ts_length = np.linalg.norm(tf_ts)
            if tf_ts_length == 0:
                cosine_score = score
            else:
                cosine_score = score / (IDF_ts_length * tf_ts_length)
            tmp.append([qid, pid, cosine_score])
    df_score = pd.DataFrame(tmp, columns=['qid', 'pid', 'score'], dtype=float).astype(
        dtype={'qid': int, 'pid': int, 'score': float})
    select_top_passages(df_score, save_raw=save_raw, save_top=save_top, filename='TF_IDF')


def select_top_passages(df_raw: pd.DataFrame, save_raw: bool = True, save_top: bool = True, filename: str = 'None'):
    # name the output files by current time
    import datetime
    now = datetime.datetime.now()
    H_M = now.strftime('%H_%M')

    df_top100 = df_raw.sort_values(by='score', ascending=False).groupby(
        'qid').apply(lambda x: x.nlargest(100, columns='score')).reset_index(drop=True)

    if save_raw:
        df_raw.to_csv(f'{filename}_{H_M}.csv', header=False, index=False)
    if save_top:
        df_top100.to_csv(f'{filename}_TOP100_{H_M}.csv', header=False, index=False)


def BM25_Score(document: pd.DataFrame, query: pd.DataFrame, terms: list, Q, D, ri, R, ni, N, k1, k2, fi, qfi, K):
    """calculate relative score of a query q and a document d in a document collection D based on BM25 model
    score = sum_(term i in q ) (log(((ri + 0.5) / (R - ri + 0.5)) / ((ni - ri + 0.5) / (N - ni - R + ri + 0.5))) * (((k1 + 1) * fi ) / (K + fi)) * (((k2 + 1) * qfi) / (k2 + qfi)))

    Parameters
    ----------
    Q : _type_
        a single query contains terms i
    D : _type_
        document collection D
    ri : _type_
        number of relevant documents that contain term i, ri=0
    R : _type_
        number of relevant documents, R=0
    ni : _type_
        number of documents that contains term i
    N : _type_
        number of documents in the document collection D
    k1 : _type_
        empirical parameter
    k2 : _type_
        empirical parameter
    fi : _type_
        frequency of term i in the document d
    qfi : _type_
        frequency of term i in the query q
    K : _type_
        empirical parameter
    """
    pass

def BM25(document: pd.DataFrame, query: pd.DataFrame, terms: list, k1, k2, b):
    # BIM score
    # document term weight in d
    # query term weight in q
    N = len(document['pid'])  # should be unique?
    R = 0
    ri = 0
    tmp_all = []  # store the result

    term_occurrence = Counter(pd.DataFrame(II_counts(
        terms=terms, document=document, returnList=True), columns=['terms', 'qid', 'pid', 'freq'])['terms'].to_list()) # Counter object, key: terms, values: number of occurrences

    tqdm.pandas(desc='Tokenisation(1/2) ......')
    document['query_token'] = document['query'].progress_apply(
        tokenisation, remove=True)  # stop words removed
    tqdm.pandas(desc='Tokenisation(2/2) ......')
    document['passage_token'] = document['passage'].progress_apply(
        tokenisation, remove=True)  # stop words removed

    avdl = np.average(document['passage_token'].apply(len))
    # avdl = 58 (kept) or 34 (removed)

    for (qid, pid, query_token, passage_token) in tqdm(zip(document['qid'], document['pid'], document['query_token'], document['passage_token']), desc="Iterating passages"):
        query_token_set = set(query_token)
        
        BM25_td = [] # score of one term and one document
        for term in query_token_set:

            # all documents contains term i
            # df_i_occurs = II_counts_df[II_counts_df['terms'] == term]
            # ni = len(df_i_occurs)  # number of documents contain term i
            ni = term_occurrence[term]

            # 1. BIM score
            BIM_score = np.log10(
                ((ri + 0.5) / (R - ri + 0.5)) / ((ni - ri + 0.5) / (N - ni - R + ri + 0.5)))
            
            # 2. document term weight in d
            dl = len(passage_token) #length of the document d
            K = K_value(k1=k1, b=b, dl=dl, avdl=avdl) #consider the length of the document d
            fi = passage_token.count(term) # frequency of term i in the document d
            d_weight = (k1 + 1) * fi / (K + fi)

            # 3. query term weight in q
            qfi = query_token.count(term) # frequency of term i in the query q
            q_weight = (k2 + 1) * qfi / (k2 + qfi)

            # combine three parts
            BM25_td.append(BIM_score * d_weight * q_weight)
        tmp_all.append([qid, pid, np.sum(BM25_td)])
    df_score = pd.DataFrame(tmp_all, columns=['qid', 'pid', 'score'], dtype=float).astype(
        dtype={'qid': int, 'pid': int, 'score': float})
    select_top_passages(df_score, filename='BM25')


def K_value(k1, b, dl, avdl):
    """consider the length of the document d,  K = k1 * ((1 - b) + b * dl / avdl)

    Parameters
    ----------
    k1 : float
        empirical parameter
    b : float
        empirical parameter
    dl : float or np.array
        length of the document d
    avdl : float
        average document length in the document collection D

    Returns
    -------
    float or np.array, depends on the type of dl
        K_value(s) for document(s) d
    """
    return k1 * ((1 - b) * b * (dl / avdl))


def D5(candidates, terms, query):
    print('Calculating IDF ...')
    IDF_ts = IDF(document=candidates, terms=terms)
    print('Calculating TFIDF ...')
    TF_IDF(document=candidates, query=query, terms=terms, IDF_ts=IDF_ts)

if __name__ == '__main__':
    nltk.download('punkt')
    candidates = load_document('0084_CW1/candidate-passages-top1000.tsv') # document collection D
    queries = load_document('0084_CW1/test-queries.tsv',
                            names=['qid', 'query']) # query collection Q
    terms_kept = load_terms(file_path='0084_CW1/terms_kept.txt') # terms in D
    terms_removed = load_terms(file_path='0084_CW1/terms_removed.txt')

    # D5(candidates=candidates, terms=terms_removed, query=queries)
