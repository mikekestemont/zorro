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

from sklearn.preprocessing import normalize
from sklearn.feature_selection import f_classif
from sklearn.model_selection import cross_validate
from sklearn import svm

from nltk.tokenize.moses import MosesTokenizer



def make_dataframe(args, model, vocab):
    if args.tokenize:
        tokenizer = MosesTokenizer()

    genres = [p for p in os.listdir(args.books_path)
              if os.path.isdir(os.sep.join((args.books_path, p)))]
    print(genres)

    data = []
    titles_covered = set()
    for genre in genres:
        print(genre)
        filenames = glob.glob(args.books_path + '/' + genre + '/*.txt')
        random.shuffle(filenames)
        genre_cnt = 0
        
        for filename in filenames:
            # some books reappear across categories! only include it once
            title = os.path.basename(filename).replace('.txt', '')
            if title in titles_covered:
                continue
            else:
                titles_covered.add(title)

            print('   '+filename)
            try:
                lines = [l.strip() for l in open(filename, 'r')]
                lines = [l for l in lines if l]
            except UnicodeDecodeError:
                continue

            random.shuffle(lines)
            vectors = []
            sentence_cnt = 0
            for line in lines:
                if args.tokenize:
                    tokens = tokenizer.tokenize(line)
                else:
                    tokens = line.split()
                tokens = [t.lower() for t in tokens]
                if len(tokens) >= args.min_sent_len and len(tokens) <= args.max_sent_len:
                    # vectorize...
                    vector = np.random.uniform(size=2400)
                    vectors.append(vector)
                    sentence_cnt += 1
                    if sentence_cnt >= args.sents_per_book:
                        break

            if len(vectors) == args.sents_per_book:
                for v, l in zip(vectors, lines):
                    data.append((genre, title, v, l))
                genre_cnt += 1
                if genre_cnt >= args.books_per_genre:
                    break

    genres, titles, vectors, sentences = zip(*data)
    vectors = np.array(vectors)

    # normalize vector to unit vorm (cf. Tang et al.):
    vectors = normalize(vectors, norm='l2', axis=1)

    df = pd.DataFrame(vectors, columns=['neur'+str(i) for i in range(vectors.shape[1])])
    df['title'] = titles
    df['genre'] = genres
    df['sentences'] = sentences

    return df


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
    clf = svm.SVC(kernel='linear', random_state=15653)
    scores = cross_validate(clf, X, data['genre'],
                            cv=5, return_train_score=False)
    print(scores)




if __name__ == '__main__':
    main()
