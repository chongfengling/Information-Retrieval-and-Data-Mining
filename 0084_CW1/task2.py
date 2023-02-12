import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from collections import Counter
from tqdm import tqdm

def load_document(file_path='0084_CW1/candidate-passages-top1000.tsv'):
    df = pd.read_csv(file_path, sep='\t', names=['qid', 'pid', 'query', 'passage'])
    return df

def load_terms(file_path='0084_CW1/terms_kept.txt'):
    """load terms generated in the task 1

    Parameters
    ----------
    file_path : str, file path
        _description_, by default '0084_CW1/terms_kept.txt'

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
        loaded terms
    document : pd.DataFrame
        candidate files, header = ['qid', 'pid', 'query', 'passage']

    Returns
    -------
    dict
        (key, value) = (term, [(qid, pid), ..., ])
    """
    II_simple_dict = {key: [] for key in terms}
    for (qid, pid, passage) in tqdm(zip(document['qid'], document['pid'], document['passage'])):
        passage_list = set(passage.split())
        for word in passage_list:
            if word in II_simple_dict.keys():
                II_simple_dict[word].append((qid, pid))
    return II_simple_dict

if __name__=='__main__':
    document = load_document()
    terms = load_terms()
    res = II_simple(terms=terms, document=document)
    print(res)
    pass
