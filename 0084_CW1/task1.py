import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from collections import Counter

# text processing methods
def tokenisation(astr):
    """return the terms in a string

    Parameters
    ----------
    astr : _type_
        _description_

    Returns
    -------
    set
        _description_
    """
    tokens = astr.split()
    unique_words = set(tokens)
    return unique_words

def remove_stop_words(aset, stop_words=[]):
    """remove stop words in the set aset

    Parameters
    ----------
    aset : aset
        _description_
    stop_words : list, optional
        _description_, by default []

    Returns
    -------
    aset
        _description_
    """
    if stop_words:
        pass
    else:
        stop_words = ['the', 'an', 'a', 'to', 'so', 'then', 'of', 'and', 'is', 'in', 'for', 'or', 'that', 'are', 'The', 'on']
    for word in stop_words:
        aset.discard(word)
    return aset

def text_preprocess(astr, remove=False):
    """text preprocess

    Parameters
    ----------
    astr : string
        document?!
    remove_stop_words : bool, optional
        _description_, by default False

    Returns
    -------
    set
        _description_
    """
    unique_words = tokenisation(astr)
    if remove:
        return remove_stop_words(unique_words)
    else:
        return unique_words

def occurrence_counter(astr, terms=None):
    """counter the occurrence frequency of terms in the astr

    Parameters
    ----------
    astr : string
        _description_
    terms : set, optional
        occurrence frequency of termm should be kept, by default None

    Returns
    -------
    Counter
        _description_
    """
    tokens = astr.split()
    tokens_frequency = Counter(tokens)
    if terms:
        tokens_set = set(tokens)
        removed_terms = tokens_set - terms
        for removed_term in removed_terms:
            del tokens_frequency[removed_term]

    return tokens_frequency

def frequency_normalization(counter: Counter):
    """normalised ranked frequency

    Parameters
    ----------
    counter : Counter
        _description_

    Returns
    -------
    list
        ranked occurrence probability
    """
    ranked_counter = sorted(counter.items(), key=lambda i: i[1], reverse=True)
    values = [tmp[1] for tmp in ranked_counter]
    normalized_prob = values / np.linalg.norm(values, 1)
    return normalized_prob

def Zipf_func(s, N):
    x_values = np.linspace(1, N, N)
    y_values = x_values ** (-s) / np.sum(x_values ** (-s))
    return y_values

def plot_distributions(x_axis, y_empircal, y_zipf, loglog=True, title='Untitled'):
    if loglog:
        plt.loglog(x_axis, y_empircal, label='prob', linestyle='dotted')
        plt.loglog(x_axis, y_zipf, label='zipf')
    else:
        plt.plot(x_axis, y_empircal, label='prob', linestyle='dotted')
        plt.plot(x_axis, y_zipf, label='zipf')
    plt.legend()
    plt.title(title)
    plt.show()

def ex_1(astr):
    # Experiemnt 1: keep stop words
    terms = text_preprocess(astr, remove=False)
    size_of_terms = len(terms)
    print(f'Size of the terms (keep stop words): {size_of_terms}')

    terms_frequency = occurrence_counter(astr)
    print(f'Top 10 terms with the highest occurrences: {terms_frequency.most_common(10)}')

    x_axis = np.linspace(1, size_of_terms, size_of_terms)
    normalized_prob = frequency_normalization(terms_frequency)
    zipf = Zipf_func(1, size_of_terms)
    plot_distributions(x_axis, y_empircal=normalized_prob, y_zipf=zipf, title='Keep stop words')

def ex_2(astr):
    # Experiment 2: remove stop words
    terms = text_preprocess(astr, remove=True)
    size_of_terms = len(terms)
    print(f'Size of the terms (remove stop words): {size_of_terms}')

    terms_frequency = occurrence_counter(astr, terms=terms)
    print(f'Top 10 terms with the highest occurrences: {terms_frequency.most_common(10)}')

    x_axis = np.linspace(1, size_of_terms, size_of_terms)
    normalized_prob = frequency_normalization(terms_frequency)
    zipf = Zipf_func(1, size_of_terms)
    plot_distributions(x_axis, y_empircal=normalized_prob, y_zipf=zipf, title='remove stop words')

if __name__=='__main__':
    # reading data from the file and convert it to string
    file_path = '0084_CW1/task1_small.txt'
    file_path = '0084_CW1/passage-collection.txt'
    with open(file_path, 'r') as f:
        astr = '.'.join([line.rstrip() for line in f])

    ex_1(astr)
    ex_2(astr)

