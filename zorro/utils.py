import random
random.seed(1001)

import torch
try:
    torch.manual_seed(1001)
    torch.cuda.manual_seed(1001)
except:
    print('no NVIDIA driver found')

import pandas as pd
import numpy as np

from sklearn.preprocessing import normalize

from seqmod import utils as u
from .data import *
from seqmod.misc.dataset import PairedDataset, Dict


def translate(model, target, gpu, beam=True, max_len=4):
    src_dict = model.encoder.embeddings.d
    inp = torch.LongTensor(list(src_dict.transform([target]))).transpose(0, 1)
    length = torch.LongTensor([len(target)]) + 2
    inp, length = u.wrap_variables((inp, length), volatile=True, gpu=gpu)
    if beam:
        scores, hyps, _ = model.translate_beam(
            inp, length, beam_width=5, max_decode_len=max_len)
    else:
        scores, hyps, _ = model.translate(inp, length, max_decode_len=max_len)

    return scores, hyps


def embed_single(model, target):
    model.train()
    src_dict = model.encoder.embeddings.d
    inp = torch.LongTensor(list(src_dict.transform([target]))).transpose(0, 1)
    length = torch.LongTensor([len(target)]) + 2
    inp, length = u.wrap_variables((inp, length), volatile=True, gpu=False)
    _, embedding = model.encoder.forward(inp, lengths=None)
    return embedding.data.numpy()[0].flatten()

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
