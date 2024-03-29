import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from collections import Counter
from tqdm import tqdm
from task1 import tokenisation


def load_document(file_path='candidate-passages-top1000.tsv', names=['qid', 'pid', 'query', 'passage']):
    df = pd.read_csv(file_path, sep='\t', names=names)
    return df


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


def II_simple(terms, document):
    """simple inverted index

    Parameters
    ----------
    terms : list
        saved terms (stopwords removed or kept), unique  
    document : pd.DataFrame
        candidate files, header = ['qid', 'pid', 'query', 'passage']

    Returns
    -------
    dict
        (key, value) = (term, [(qid, pid), ..., ])
    """
    II_simple_dict = {key: [] for key in terms}
    for (qid, pid, passage) in tqdm(zip(document['qid'], document['pid'], document['passage']), desc='II_simple'):
        passage_set = set(tokenisation(passage, remove=False))
        for word in passage_set:
            if word in II_simple_dict.keys():
                II_simple_dict[word].append((qid, pid))
    return II_simple_dict


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
    for (qid, pid, passage) in tqdm(zip(document['qid'], document['pid'], document['passage']), desc='II_counts'):
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


def II_positions(terms, document):
    """inverted index with positions

    Parameters
    ----------
    terms : list
        _description_
    document : pd.Dataframe
        candidate-passages-top1000.tsv in pd.DataFrame, header = ['qid', 'pid', 'query', 'passage']

    Returns
    -------
    dict
        _description_
    """
    II_positions_dict = {key: [] for key in terms}
    for (qid, pid, passage) in tqdm(zip(document['qid'], document['pid'], document['passage']), desc='II_positions'):
        passage_list = tokenisation(passage, remove=False)
        passage_set = set(passage_list)
        for word in passage_set:
            if word in II_positions_dict.keys():
                positions = [index for index, tmp in enumerate(
                    passage_list) if tmp == word]
                II_positions_dict[word].append([(qid, pid), positions])
    return II_positions_dict


if __name__ == '__main__':
    document = load_document(
        file_path='candidate-passages-top1000.tsv')
    terms = load_terms(file_path='terms_kept.txt')
    # Three methods in Inverted Index
    # res = II_simple(terms=terms, document=document)
    res = II_counts(terms=terms, document=document) # used in task 3 and task 4
    # res = II_positions(terms=terms, document=document)
