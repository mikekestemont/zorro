import os
import glob
import random
from collections import deque

import torch
from nltk.tokenize.moses import MosesTokenizer

from seqmod import utils as u
from .data import *
from seqmod.misc.dataset import PairedDataset, Dict


class SentenceCouples(object):
    """
    Pairs of sentences tokenized at the word-level.
    """
    def __init__(self, input_, max_items=None, max_len=30, tokenized=False):
        if os.path.isdir(input_):
            if not input_.endswith('/'):
                input_ += '/'
            self.filenames = sorted(list(glob.glob(input_ + '*.txt')))
        else:
            self.filenames = [input_]

        self.max_items = max_items
        self.max_item_len = max_len
        self.processed = 0
        self.tokenized, self.tokenizer = tokenized, None
        if not self.tokenized:
            self.tokenizer = MosesTokenizer()

    def __iter__(self):
        for filename in self.filenames:
            couple = deque(maxlen=2)
            for line in open(filename, 'r'):
                line = ' '.join(line.strip().split())
                if not line:
                    continue
                if self.tokenized:
                    tokens = line.split()
                else:
                    try:
                        tokens = tuple(self.tokenizer.tokenize(line))
                    except IndexError:
                        tokens = None
                if len(tokens) > 0 and len(tokens) <= self.max_item_len:
                    couple.append(tokens)
                    if len(couple) == 2:
                        self.processed += 1
                        yield tuple(couple)
                if self.max_items and self.processed >= self.max_items:
                    return

    def __len__(self):
        # number of triples yielded so far
        return self.processed


class SnippetCouples(object):
    """
    Pairs of freeform character snippters as strings (no tokenization).
    """
    def __init__(self, input_, max_items=None,
                 focus_size=0, right_size=0):
        """
        size is the number of characters in a single when `shingling=characters.
        """
        if os.path.isdir(input_):
            if not input_.endswith('/'):
                input_ += '/'
            self.filenames = sorted(list(glob.glob(input_ + '*.txt')))
        else:
            self.filenames = [input_]

        if not focus_size:
            raise ValueError(f'`focus_size` should be >=1')
        if not right_size:
            print('Setting `right_size` to that of `focus_size`')
            right_size = focus_size

        self.focus_size = focus_size
        self.right_size = right_size
        self.max_items = max_items
        self.processed = 0

    def __iter__(self):
        chunk_size = self.focus_size + self.right_size
        for filename in self.filenames:
            chunk = ''
            for line in open(filename, 'r'):
                line = line.strip()
                if line:
                    chunk += ' ' + line
                while len(chunk) >= chunk_size:
                    yield tuple([chunk[:self.focus_size],
                                 chunk[self.focus_size : -(len(chunk) - chunk_size)]])
                    self.processed += 1
                    if self.max_items and self.processed >= self.max_items:
                        return
                    chunk = chunk[self.focus_size:]

    def __len__(self):
        # number of triples yielded so far
        return self.processed


def shingle_dataset(args, vocab_dict=None, focus_size=None, right_size=None):
    if focus_size:
        args.focus_size = focus_size
    if right_size:
        args.right_size = right_size

    # load the data:
    if args.task == 'sentences':
        dataset = list(SentenceCouples(args.input,
                        max_items=args.max_items))
        print(f'* loaded {len(dataset)} sentences')
    elif args.task == 'snippets':
        dataset = list(SnippetCouples(args.input,
                         focus_size=args.focus_size,
                         right_size=args.right_size,
                         max_items=args.max_items))
        print(f'* loaded {len(dataset)} snippets')
    else:
        raise ValueError("`Task` should be one of ('sentences', 'snippets')")

    # random shuffle:
    if args.shuffle:
        print('* shuffling batches...')
        random.seed(args.rnd_seed)
        random.shuffle(dataset)

    for c in dataset[:10]:
        print('\t'.join(' '.join(s[:10]) for s in c))

    if vocab_dict is None:
        vocab_dict = Dict(pad_token=u.PAD, bos_token=u.BOS, eos_token=u.EOS,
                      min_freq=args.min_item_freq, sequential=True, force_unk=True,
                      max_size=args.max_vocab_size)

    focus, right = zip(*dataset)
    del dataset
    if not vocab_dict.fitted:
        vocab_dict.fit(focus, right) # sometimes inefficient? # do a partial fit in the triple store?
    
    train, valid = PairedDataset(
        src=(focus,), trg=(right,),
        d={'src': (vocab_dict,), 'trg': (vocab_dict,)},
        batch_size=args.batch_size, gpu=args.gpu,
        align_right=args.reverse, fitted=False).splits(sort_by='src', dev=args.dev, test=None, sort=True)

    return train, valid, vocab_dict
