import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from collections import Counter
from tqdm import tqdm
import nltk
import re


def load_document(
    file_path='candidate-passages-top1000.tsv',
    names=['qid', 'pid', 'query', 'passage'],
):
    df = pd.read_csv(file_path, sep='\t', names=names)
    return df


def tokenisation(astr, remove=True):
    """use nltk to
    1. lower the character
    2. substitute all non-alphanumeric characters (excluding whitespace)
    3. Return a tokenized (English) copy of *text* using NLTK's recommended word tokenizer

    Parameters
    ----------
    astr : string
        all characters in the collection

    Returns
    -------
    list
        tokens
    """
    # remove url
    astr_no_url = re.sub(r"http\S+", " ", astr)
    # lower character
    astr_lower = astr_no_url.lower()
    # remove non alpha characters
    astr_lower_nonalpha = re.sub(r'[^a-z0-9\s]', ' ', astr_lower)
    # nltk.download('punkt')
    tokens_list = nltk.word_tokenize(astr_lower_nonalpha)
    if remove:
        tokens_list_removed = []
        stop_words = nltk.corpus.stopwords.words('english')
        for token in tokens_list:
            if token not in stop_words:
                tokens_list_removed.append(token)
        return tokens_list_removed
    else:
        return tokens_list


def text_preprocess(astr, remove=False, save_txt=False):
    """text preprocess

    Parameters
    ----------
    astr : string
        document
    remove_stop_words : bool, optional
        _description_, by default False

    Returns
    -------
    set
        unique terms for a string astr
    """
    tokens_list = tokenisation(astr, remove=remove)
    unique_words = set(tokens_list)
    if remove:
        if save_txt:
            np.savetxt(
                '0084_CW2/terms_removed.txt',
                np.array(list(unique_words)),
                delimiter='\n',
                fmt="%s",
            )
        return unique_words
    else:
        if save_txt:
            np.savetxt(
                'terms_kept.txt', np.array(list(unique_words)), delimiter='\n', fmt="%s"
            )
        return unique_words


def load_terms(file_path='terms_kept.txt'):
    """load terms generated in the task 1

    Parameters
    ----------
    file_path : str, file path
        _description_, by default 'terms_kept.txt'

    Returns
    -------
    list
        each elements is a term (string type)
    """
    with open(file_path, 'r') as f:
        lines = f.readlines()
    terms = [line.strip() for line in lines]
    return terms


def BM25(document: pd.DataFrame, query: pd.DataFrame, terms: list, k1, k2, b):
    """calculate relative score of a query q and a document d in a document collection D based on BM25 model
    score = sum_(term i in q ) (log(((ri + 0.5) / (R - ri + 0.5)) / ((ni - ri + 0.5) / (N - ni - R + ri + 0.5))) * (((k1 + 1) * fi ) / (K + fi)) * (((k2 + 1) * qfi) / (k2 + qfi)))

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
    # BIM score
    # document term weight in d
    # query term weight in q
    N = len(document['pid'].unique())  # should be unique?
    R = 0
    ri = 0
    tmp_all = []  # store the result

    term_occurrence = Counter(
        pd.DataFrame(
            II_counts(terms=terms, document=document, returnList=True),
            columns=['terms', 'qid', 'pid', 'freq'],
        )['terms'].to_list()
    )  # Counter object, key: terms, values: number of occurrences

    tqdm.pandas(desc='Tokenisation(1/2) ......')
    document['query_token'] = document['queries'].progress_apply(
        tokenisation, remove=True
    )  # stop words removed
    tqdm.pandas(desc='Tokenisation(2/2) ......')
    document['passage_token'] = document['passage'].progress_apply(
        tokenisation, remove=True
    )  # stop words removed

    avdl = np.average(document['passage_token'].apply(len))
    # avdl = 58 (kept) or 34 (removed)

    for (qid, pid, query_token, passage_token, relevancy) in tqdm(
        zip(
            document['qid'],
            document['pid'],
            document['query_token'],
            document['passage_token'],
            document['relevancy'],
        ),
        desc="Iterating passages",
    ):
        query_token_set = set(query_token)

        BM25_td = []  # score of one term and one document
        for term in query_token_set:
            # all documents contains term i
            # df_i_occurs = II_counts_df[II_counts_df['terms'] == term]
            # ni = len(df_i_occurs)  # number of documents contain term i
            ni = term_occurrence[term]

            # 1. BIM score
            BIM_score = np.log10(
                ((ri + 0.5) / (R - ri + 0.5))
                / ((ni - ri + 0.5) / (N - ni - R + ri + 0.5))
            )

            # 2. document term weight in d
            dl = len(passage_token)  # length of the document d
            # consider the length of the document d
            K = K_value(k1=k1, b=b, dl=dl, avdl=avdl)
            # frequency of term i in the document d
            fi = passage_token.count(term)
            d_weight = (k1 + 1) * fi / (K + fi)

            # 3. query term weight in q
            qfi = query_token.count(term)  # frequency of term i in the query q
            q_weight = (k2 + 1) * qfi / (k2 + qfi)

            # combine three parts
            BM25_td.append(BIM_score * d_weight * q_weight)
        tmp_all.append([qid, pid, np.sum(BM25_td), relevancy])
    df_score = pd.DataFrame(
        tmp_all, columns=['qid', 'pid', 'score', 'relevancy'], dtype=float
    ).astype(dtype={'qid': int, 'pid': int, 'score': float, 'relevancy': float})
    select_top_passages(df_raw=df_score, filename='bm25', save_raw=True, save_top=True)


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


def II_counts(terms, document, returnList=False):
    """inverted index with counts

    Parameters
    ----------
    terms : list
        loaded terms
    document : pd.DataFrame
        candidate-passages-top1000.tsv in pd.DataFrame, header = ['qid', 'pid', 'query', 'passage']
    returnList : bool
        return as List type. [[term, qid, pid, frequency], [term, qid, pid, frequency], ...]

    Returns
    -------
    dict
        (key, value) = (term, [[(qid, pid), frequency], [(qid, pid), frequency], ...])
    """
    II_counts_dict = {key: [] for key in terms}
    II_simple_list = []
    for (qid, pid, passage) in tqdm(
        zip(document['qid'], document['pid'], document['passage']), desc='II_counts'
    ):
        passage_list = tokenisation(passage, remove=False)
        passage_set = set(passage_list)
        for word in passage_set:
            if word in II_counts_dict.keys():
                counts = passage_list.count(word)
                II_counts_dict[word].append([(qid, pid), counts])
                II_simple_list.append([word, qid, pid, counts])
    if returnList:  # easy to be converted to pd.DataFrame
        return II_simple_list
    else:
        return II_counts_dict


def select_top_passages(
    df_raw: pd.DataFrame,
    save_raw: bool = True,
    save_top: bool = True,
    filename: str = 'None',
    top_n: int = 100,
    rank_col: str = 'score',
    group_col: str = 'qid',
):
    # name the output files by current time
    import datetime

    now = datetime.datetime.now()
    H_M = now.strftime('%H_%M')

    df_new = (
        df_raw.groupby(group_col)
        .apply(lambda x: x.nlargest(top_n, columns=rank_col))
        .reset_index(drop=True)
    )

    if save_raw:
        df_raw.to_csv(
            f'0084_CW2/{filename}_raw_top{top_n}_{H_M}.csv', header=False, index=False
        )
    if save_top:
        df_new.to_csv(
            f'0084_CW2/{filename}_ordered_top{top_n}_{H_M}.csv',
            header=False,
            index=False,
        )
    return df_new


if __name__ == '__main__':
    ''' BM25 model
    nltk.download('punkt')
    # run BM25 model
    with open(
        '/Users/ling/MyDocuments/COMP0084/0084_CW1/passage-collection.txt', 'r'
    ) as f:
        astr = '.'.join([line.rstrip() for line in f])

    terms = text_preprocess(astr, remove=True, save_txt=True)
    candidates = load_document(
        '/Users/ling/MyDocuments/COMP0084/0084_CW2/candidate_passages_top1000.tsv'
    )  # document collection
    candidates = pd.read_csv(
        '/Users/ling/MyDocuments/COMP0084/0084_CW2/validation_data.tsv',
        sep='\t'
    )
    '''
    # apply the previous result of BM25 model
    queries = load_document(
        '/Users/ling/MyDocuments/COMP0084/0084_CW2/test-queries.tsv',
        names=['qid', 'queries'],
    )  # query    collection Q
    df_score = pd.read_csv(
        '0084_CW2/bm25_raw.csv', names=['qid', 'pid', 'score', 'relevancy']
    )
    select_top_passages(
        df_raw=df_score, filename='bm25', save_raw=False, top_n=100, save_top=True
    )
