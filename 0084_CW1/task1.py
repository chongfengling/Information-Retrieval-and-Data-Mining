import numpy as np
import matplotlib.pyplot as plt
from collections import Counter
import nltk
import re


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
            np.savetxt('terms_removed.txt',
                       np.array(list(unique_words)), delimiter='\n', fmt="%s")
        return unique_words
    else:
        if save_txt:
            np.savetxt('terms_kept.txt',
                       np.array(list(unique_words)), delimiter='\n', fmt="%s")
        return unique_words


def occurrence_counter(astr, kept_terms=None):
    """counter the occurrence frequency of terms in the astr

    Parameters
    ----------
    astr : string
        _description_
    kept_terms : set, optional
        occurrence frequency of terms, by default None, that is, keep the frequency of stopwords

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


def plot_distributions(x_axis, y_empirical, y_zipf, loglog=True, title='Untitled'):
    fig, ax = plt.subplots()
    if loglog:
        ax.loglog(x_axis, y_empirical, label='prob', linestyle='dotted')
        ax.loglog(x_axis, y_zipf, label='zipf')
    else:
        ax.plot(x_axis, y_empirical, label='prob', linestyle='dotted')
        ax.plot(x_axis, y_zipf, label='zipf')
    ax.set_xlabel('Rank')
    ax.set_ylabel('Probability')
    ax.set_title(title)
    ax.legend()
    # ax.title(title)
    # plt.show()
    filename = title.replace(' ', '_')
    fig.savefig(f'assets/{filename}.png')


def ex_1(astr):
    # Experiemnt 1: keep stop words
    terms = text_preprocess(astr, remove=False, save_txt=True)
    size_of_terms = len(terms)
    print(f'Size of the terms (keep stop words): {size_of_terms}')

    terms_frequency = occurrence_counter(astr)
    print(
        f'Top 10 terms with the highest occurrences: {terms_frequency.most_common(10)}')

    x_axis = np.linspace(1, size_of_terms, size_of_terms)
    normalized_prob = frequency_normalization(terms_frequency)
    zipf = Zipf_func(1, size_of_terms)
    plot_distributions(x_axis, y_empircal=normalized_prob,
                       y_zipf=zipf, loglog=False, title='Keep stop words')
    plot_distributions(x_axis, y_empircal=normalized_prob,
                       y_zipf=zipf, title='Keep stop words (loglog)')


def ex_2(astr):
    # Experiment 2: remove stop words
    terms = text_preprocess(astr, remove=True, save_txt=True)
    size_of_terms = len(terms)
    print(f'Size of the terms (remove stop words): {size_of_terms}')

    terms_frequency = occurrence_counter(astr, kept_terms=terms)
    # Top 10 terms with the highest occurrences: [('1', 43992), ('2', 33919), ('one', 27300), ('name', 25163), ('3', 22602), ('also', 21757), ('number', 21367), ('may', 20556), ('cost', 17128), ('used', 16542)]
    print(
        f'Top 10 terms with the highest occurrences: {terms_frequency.most_common(10)}')

    x_axis = np.linspace(1, size_of_terms, size_of_terms)
    normalized_prob = frequency_normalization(terms_frequency)
    zipf = Zipf_func(1, size_of_terms)
    plot_distributions(x_axis, y_empircal=normalized_prob,
                       y_zipf=zipf, loglog=False, title='Remove stop words')
    plot_distributions(x_axis, y_empircal=normalized_prob,
                       y_zipf=zipf, title='Remove stop words (loglog)')


if __name__ == '__main__':
    # reading data from the file and convert it to string
    # file_path = 'task1_small.txt'
    file_path = 'passage-collection.txt'
    with open(file_path, 'r') as f:
        astr = '.'.join([line.rstrip() for line in f])
    nltk.download('punkt')
    # keep stopwords
    ex_1(astr)
    # remove stopwords
    ex_2(astr)