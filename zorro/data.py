import os
import glob
import random
from collections import deque

import torch
from seqmod import utils as u
from .data import *
from seqmod.misc.dataset import PairedDataset, Dict


#class TripleStore(object):
#
#    def __init__(self, input_, shingling='sentences',
#                 allow_overlap=False, max_triples=None,
#                 left_size=0, focus_size=0, right_size=0, shingle_stride=0):
#        """
#        size is the number of characters in a single when `shingling=characters.
#        """
#        if os.path.isdir(input_):
#            if not input_.endswith('/'):
#                input_ += '/'
#            self.filenames = sorted(list(glob.glob(input_ + '*.txt')))
#        else:
#            self.filenames = [input_]
#
#        if shingling not in ('sentences', 'characters'):
#            raise ValueError(f'Unsupported shingling option {shingling}')
#        if shingling == 'characters':
#            if not focus_size:
#                raise ValueError(f'With character shingling option, focus_size should be >=1')
#            if not left_size and not right_size:
#                raise ValueError(f'At least, left_size OR right_size should be specified.')
#        if not shingle_stride:
#            shingle_stride = focus_size
#        if shingle_stride > left_size + focus_size + right_size:
#            raise ValueError(f'shingle_stride too large: you are missing out on data.')
#
#        self.shingling = shingling
#        self.focus_size = focus_size
#        self.left_size = left_size
#        self.right_size = right_size
#        self.shingle_stride = shingle_stride
#        self.allow_overlap = allow_overlap
#        self.max_triples = max_triples
#        self.processed = 0
#
#    def __iter__(self):
#        if self.shingling == 'sentences':
#            if self.allow_overlap:
#                for filename in self.filenames:
#                    triple = deque(maxlen=3)
#                    for line in open(filename, 'r'):
#                        line = ' '.join(line.strip().split())
#                        if line:
#                            triple.append(line)
#                            if len(triple) == 3:
#                                self.processed += 1
#                                yield tuple(triple)
#                        if self.max_triples and self.processed >= self.max_triples:
#                            return
#            else:
#                for filename in self.filenames:
#                    triple = deque(maxlen=3)
#                    for line in open(filename, 'r'):
#                        line = ' '.join(line.strip().split())
#                        if line:
#                            triple.append(line)
#                            if len(triple) == 3:
#                                self.processed += 1
#                                yield tuple(triple)
#                                triple = deque()
#                        if self.max_triples and self.processed >= self.max_triples:
#                            return
#
#        elif self.shingling == 'characters':
#            chunk_size = self.left_size + self.focus_size + self.right_size
#            for filename in self.filenames:
#                chunk = ''
#                for line in open(filename, 'r'):
#                    line = line.strip()
#                    if line:
#                        chunk += ' ' + line
#                    while len(chunk) >= chunk_size:
#                        yield tuple([chunk[:self.left_size],
#                                     chunk[self.left_size : self.left_size + self.focus_size],
#                                     chunk[self.left_size + self.focus_size : -(len(chunk) - chunk_size)]])
#                        self.processed += 1
#                        if self.max_triples and self.processed >= self.max_triples:
#                            return
#                        if self.allow_overlap:
#                            chunk = chunk[self.shingle_stride:]
#                        else:
#                            chunk = chunk[self.focus_size:]
#
#    def __len__(self):
#        # number of triples yielded so far
#        return self.processed

class CoupleStore(object):

    def __init__(self, input_, shingling='sentences',
                 allow_overlap=False, max_couples=None,
                 focus_size=0, right_size=0, shingle_stride=0):
        """
        size is the number of characters in a single when `shingling=characters.
        """
        if os.path.isdir(input_):
            if not input_.endswith('/'):
                input_ += '/'
            self.filenames = sorted(list(glob.glob(input_ + '*.txt')))
        else:
            self.filenames = [input_]

        if shingling not in ('sentences', 'characters'):
            raise ValueError(f'Unsupported `shingling` option: {shingling}')
        if shingling == 'characters':
            if not focus_size:
                raise ValueError(f'With "character" shingling, `focus_size` should be >=1')
            if not right_size:
                print('Setting `right_size` to that of `focus_size`')
                right_size = focus_size
        if not shingle_stride:
            print('Setting `shingle_stride` to `focus_size`')
            shingle_stride = focus_size
        if shingle_stride > focus_size:
            print(f'`shingle_stride` larger than `focus_size`: you are missing out on data.')

        self.shingling = shingling
        self.focus_size = focus_size
        self.right_size = right_size
        self.shingle_stride = shingle_stride
        self.allow_overlap = allow_overlap
        self.max_couples = max_couples
        self.processed = 0

    def __iter__(self):
        if self.shingling == 'sentences':
            if self.allow_overlap:
                for filename in self.filenames:
                    couple = deque(maxlen=2)
                    for line in open(filename, 'r'):
                        line = ' '.join(line.strip().split())
                        if line:
                            couple.append(line)
                            if len(couple) == 2:
                                self.processed += 1
                                yield tuple(couple)
                        if self.max_couples and self.processed >= self.max_couples:
                            return
            else:
                for filename in self.filenames:
                    couple = deque(maxlen=2)
                    for line in open(filename, 'r'):
                        line = ' '.join(line.strip().split())
                        if line:
                            couple.append(line)
                            if len(couple) == 2:
                                self.processed += 1
                                yield tuple(couple)
                                couple = deque()
                        if self.max_couples and self.processed >= self.max_couples:
                            return

        elif self.shingling == 'characters':
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
                        if self.max_couples and self.processed >= self.max_couples:
                            return
                        if self.allow_overlap:
                            chunk = chunk[self.shingle_stride:]
                        else:
                            chunk = chunk[self.focus_size + self.right_size:]

    def __len__(self):
        # number of triples yielded so far
        return self.processed


def shingle_dataset(args, vocab_dict=None, focus_size=None, right_size=None):
    if focus_size:
        args.focus_size = focus_size
    if right_size:
        args.right_size = right_size
    # load the data:
    if args.task == 'triples':
        dataset = list(TripleStore(args.input,
                         shingling=args.shingling,
                         focus_size=args.focus_size,
                         left_size=args.left_size,
                         right_size=args.right_size,
                         shingle_stride=args.shingle_stride,
                         allow_overlap=args.allow_overlap,
                         max_triples=args.max_items))
        print(f'* loaded {len(dataset)} triples')
    elif args.task == 'couples':
        dataset = list(CoupleStore(args.input,
                         shingling=args.shingling,
                         focus_size=args.focus_size,
                         right_size=args.right_size,
                         shingle_stride=args.shingle_stride,
                         allow_overlap=args.allow_overlap,
                         max_couples=args.max_items))
        print(f'* loaded {len(dataset)} couples')
    else:
        raise ValueError("`Task` should be one of ('couples', 'triples')")

    # random shuffle:
    if args.shuffle:
        print('* shuffling batches...')
        random.seed(args.rnd_seed)
        random.shuffle(dataset)

    for c in dataset[:10]:
        print('\t'.join(c))

    if vocab_dict is None:
        vocab_dict = Dict(pad_token=u.PAD, bos_token=u.BOS, eos_token=u.EOS,
                      min_freq=args.min_char_freq, sequential=True, force_unk=True)
    if args.task == 'triples':
        left, focus, right = zip(*dataset)
        del dataset
        if vocab_dict is None:
            vocab_dict.fit(left, focus, right) # sometimes inefficient? # do a partial fit in the triple store?
        
        train, valid = PairedDataset(
            src=(focus,), trg=(left, right),
            d={'src': (vocab_dict,), 'trg': (vocab_dict, vocab_dict)},
            batch_size=args.batch_size, gpu=args.gpu,
            align_right=args.reverse, fitted=False).splits(sort_by='src', dev=args.dev, test=None, sort=True)
    elif args.task == 'couples':
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


#def main():
#    triple_store = TripleStore(input_='../data', shingling='characters',
#                 size=100, allow_overlap=True, max_triples=10,
#                 focus_size=21, left_size=21, right_size=21)
#    for triple in triple_store:
#        print('\t'.join([t for t in triple]))
#
#if __name__ == '__main__':
#    main()
