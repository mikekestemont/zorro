"""
- Select n random sentences from m random novels per genre
- Embed all sentences using a skipthoughts model
1 - Find out the correlation for each genre (1 vs 0; other genres) with each individual neuron
    - at the sentence level
    - at the book level (some form of aggregation, e.g. mean)
2 - Routine genre classificatio experiment
"""

import argparse
import os
import glob

import random
random.seed(1001)

import torch
try:
    torch.manual_seed(1001)
    torch.cuda.manual_seed(1001)
except:
    print('no NVIDIA driver found')

import seqmod.utils as u

import pandas as pd
import numpy as np

from sklearn.neighbors import KNeighborsClassifier

from nltk.tokenize.moses import MosesTokenizer

from zorro.utils import make_dataframe

def main():
    # parse params:
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', default='./Skipthoughts-2018_01_13-13_11_19-13.726/model.pt', type=str)
    parser.add_argument('--books_path', default='/Users/mike/GitRepos/potter/data/other/books_txt_full', type=str)
    parser.add_argument('--tokenize', action='store_true')
    parser.add_argument('--dict_path', default='./Skipthoughts-2018_01_13-13_11_19-13.726/model.dict.pt', type=str)
    parser.add_argument('--gpu', action='store_true')

    parser.add_argument('--max_sent_len', default=25, type=int)
    parser.add_argument('--min_sent_len', default=5, type=int)
    parser.add_argument('--sents_per_book', default=100, type=int)
    parser.add_argument('--books_per_genre', default=20, type=int)

    parser.add_argument('--nb_top_features', default=5, type=int)

    args = parser.parse_args()

    # load model and dict
    #model = u.load_model(args.model_path)
    #vocab_dict = u.load_model(args.dict_path)

    data = make_dataframe(args, model=None, vocab=None)

    X = data.filter(regex='neur').as_matrix()
    X = np.array(X, dtype=np.float32)

    import faiss
    index = faiss.IndexFlatL2(X.shape[1])
    index.add(X)
    print(index.ntotal)

    n_examples = 100
    n_neighbors = 5
    D, I = index.search(X[:n_examples], n_neighbors)

    for i in range(n_examples):
        print('-> src sent:', data['sentences'][i][:120])
        for cnt, (dist, idx) in enumerate(zip(D[i, :], I[i, :])):
            print(f'  {cnt + 1} @{dist:.3f}: {data["sentences"][idx][:120]}')


if __name__ == '__main__':
    main()
