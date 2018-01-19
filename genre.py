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

from nltk.tokenize.moses import MosesTokenizer
from zorro.utils import make_dataframe

from sklearn.feature_selection import f_classif
from sklearn.model_selection import cross_validate
from sklearn import svm

import numpy as np

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

    X = data.filter(regex='neur')

    print('=======\n single-neuron binary classification (one vs rest):')
    for genre in set(data['genre']):
        print(f'  Testing genre {genre}:')
        y = [int(i) for i in data.genre == genre]
        # univariate feature selection with F-test for feature scoring
        F, pval = f_classif(X, y)
        max_idxs = np.argsort(F)[::-1][:args.nb_top_features]
        neuron_names, neuron_f_scores = np.array(X.columns)[max_idxs], F[max_idxs]

        for name, score in zip(neuron_names, neuron_f_scores):
            print(f'      {name} -> {score:.2f} F-score')

    # categorical case:
    print('=======\n single-neuron binary genre classification:')
    F, pval = f_classif(X, data['genre'])
    max_idxs = np.argsort(F)[::-1][:args.nb_top_features]
    neuron_names, neuron_f_scores = np.array(X.columns)[max_idxs], F[max_idxs]

    for name, score in zip(neuron_names, neuron_f_scores):
        print(f'      {name} -> {score:.2f} F-score')

    print('=======\n all-neuron genre classification (5-fold CV):')
    #clf = svm.SVC(kernel='linear', random_state=15653)
    #scores = cross_validate(clf, X, data['genre'],
    #                        cv=5, return_train_score=False)
    #print(scores)


if __name__ == '__main__':
    main()
