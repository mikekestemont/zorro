import argparse

import random
random.seed(1001)

import torch
try:
    torch.manual_seed(1001)
    torch.cuda.manual_seed(1001)
except:
    print('no NVIDIA driver found')

import seqmod.utils as u
from seqmod.modules.encoder_decoder import make_rnn_encoder_decoder
from seqmod.misc.dataset import PairedDataset, Dict

from zorro.data import TripleStore


def main():
    parser = argparse.ArgumentParser()
    # dataset
    parser.add_argument('--path', type=str, default=None)
    parser.add_argument('--input', type=str, default='data')
    parser.add_argument('--min_char_freq', type=int, default=10)
    parser.add_argument('--min_len', default=1, type=int)
    parser.add_argument('--max_len', default=15, type=int)
    parser.add_argument('--dev', default=0.05, type=float)
    parser.add_argument('--test', default=0.05, type=float)
    parser.add_argument('--rnd_seed', default=12345, type=int)
    parser.add_argument('--max_triples', default=1000, type=int)
    parser.add_argument('--allow_overlap', action='store_true', default=False)
    parser.add_argument('--gpu', action='store_true', default=False)
    parser.add_argument('--reverse', action='store_true', default=False)

    # training
    parser.add_argument('--batch_size', default=15, type=int)

    args = parser.parse_args()

    # load the triples:
    triple_store = TripleStore(args.input,
                               allow_overlap=args.allow_overlap,
                               max_triples=args.max_triples)
    triples = list(triple_store)

    # random shuffle:
    random.seed(args.rnd_seed)
    random.shuffle(triples)

    left, focus, right = zip(*triples)
    del triples

    vocab_dict = Dict(pad_token='<PAD>',
                      min_freq=args.min_char_freq, sequential=True)
    vocab_dict.fit(left, focus, right) # inefficient?
    
    train, dev, test = PairedDataset(
        src=(left, focus, right, ), trg=None,
        d={'src': (vocab_dict, vocab_dict, vocab_dict, )},
        batch_size=args.batch_size, gpu=args.gpu,
        align_right=args.reverse, fitted=False).splits(sort_by='src', dev=args.dev, test=args.test, sort=True)

    print(f' * vocabulary size {len(vocab_dict)}')
    print(f' * number of train batches {len(train)}')
    print(f' * maximum batch size {args.batch_size}')

    model = make_rnn_encoder_decoder(2, 64, 150, vocab_dict, cell='GRU', bidi=True, att_type='general')

    u.initialize_model(model, rnn={'type': 'orthogonal', 'args': {'gain': 1.0}})

if __name__ == '__main__':
    main()