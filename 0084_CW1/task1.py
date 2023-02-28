import numpy as np
import matplotlib.pyplot as plt
from collections import Counter
import nltk
import re

# text processing methods
def tokenisation(astr, remove=True):
    """use nltk to
    1. lower the character
    2. substitute all non-alphanumeric characters (excluding whitespace)
    3. Return a tokenized (English) copy of *text* using NLTK's recommended word tokenizer
    To Do:
    1. remove url
    

    Parameters
    ----------
    astr : _type_
        _description_

    Returns
    -------
    list
        tokens
    """
    astr_lower = astr.lower()
    astr_lower_nonalpha = re.sub(r'[^a-z0-9\s]', ' ', astr_lower)
    # nltk.download('punkt')
    tokens_list = nltk.word_tokenize(astr_lower_nonalpha)
    if remove:
        tmp = []
        stop_words = nltk.corpus.stopwords.words('english')
        for token in tokens_list:
            if token not in stop_words:
                tmp.append(token)
        return tmp
    else:
        return tokens_list

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
        nltk.download('stopwords')
        stop_words = nltk.corpus.stopwords.words('english')
        # stop_words = ['the', 'an', 'a', 'to', 'so', 'then', 'of', 'and', 'is', 'in', 'for', 'or', 'that', 'are', 'The', 'on']
    for word in stop_words:
        aset.discard(word)
    return aset

def text_preprocess(astr, remove=False, save_txt = False):
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
        unique terms for a string astr
    """
    tokens_list = tokenisation(astr, remove=False)
    unique_words = set(tokens_list)
    if remove:
        unique_words_removed = remove_stop_words(unique_words)
        if save_txt: np.savetxt('0084_CW1/terms_removed.txt', np.array(list(unique_words_removed)), delimiter='\n', fmt="%s")
        return unique_words_removed
    else:
        if save_txt: np.savetxt('0084_CW1/terms_kept.txt', np.array(list(unique_words)), delimiter='\n', fmt="%s")
        return unique_words

def occurrence_counter(astr, kept_terms=None):
    """counter the occurrence frequency of terms in the astr

    Parameters
    ----------
    astr : string
        _description_
    kept_terms : set, optional
        occurrence frequency of terms should be kept, by default None

    Returns
    -------
    Counter
        _description_
    """
    tokens_list = tokenisation(astr, remove=False)
    tokens_frequency = Counter(tokens_list)
    if kept_terms:
        tokens_set = set(tokens_list)
        removed_terms = tokens_set - kept_terms
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
    sorted_counter = sorted(counter.items(), key=lambda i: i[1], reverse=True)
    values = [tmp[1] for tmp in sorted_counter]
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
    filename = title.replace(' ', '_')
    plt.savefig(f'0084_CW1/assets/{filename}.png')

def ex_1(astr):
    # Experiemnt 1: keep stop words
    terms = text_preprocess(astr, remove=False, save_txt=True)
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
    terms = text_preprocess(astr, remove=True, save_txt=True)
    size_of_terms = len(terms)
    print(f'Size of the terms (remove stop words): {size_of_terms}')

    terms_frequency = occurrence_counter(astr, kept_terms=terms)
    print(f'Top 10 terms with the highest occurrences: {terms_frequency.most_common(10)}') # Top 10 terms with the highest occurrences: [('1', 43992), ('2', 33919), ('one', 27300), ('name', 25163), ('3', 22602), ('also', 21757), ('number', 21367), ('may', 20556), ('cost', 17128), ('used', 16542)]

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
    nltk.download('punkt')

    ex_1(astr)
    ex_2(astr)

