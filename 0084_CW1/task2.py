import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from collections import Counter
from tqdm import tqdm

def load_document(file_path='0084_CW1/candidate-passages-top1000.tsv'):
    df = pd.read_csv(file_path, sep='\t', names=['qid', 'pid', 'query', 'passage'])
    return df

def load_terms(file_path='0084_CW1/terms_kept.txt'):
    with open(file_path, 'r') as f:
        lines = f.readlines()
    terms = [line.strip() for line in lines]
    return terms
