import argparse
import random

from seqmod.misc.dataset import PairedDataset, Dict
from seqmod import utils as u

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
    parser.add_argument('--max_triples', default=100, type=int)
    parser.add_argument('--allow_overlap', action='store_true', default=False)
    parser.add_argument('--gpu', action='store_true', default=False)
    parser.add_argument('--reverse', action='store_true', default=False)

    # training
    parser.add_argument('--batch_size', default=2, type=int)

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

    if args.path is not None:
        with open(args.path, 'rb+') as f:
            dataset = PairedDataset.from_disk(f)
        dataset.set_batch_size(args.batch_size)
        dataset.set_gpu(args.gpu)
        train, valid = dataset.splits(sort_by='src', dev=args.dev, test=None)
        src_dict = dataset.dicts['src']
    else:
        vocab_dict = Dict(pad_token=u.PAD, eos_token=u.EOS, bos_token=u.BOS,
                          min_freq=args.min_char_freq)
        vocab_dict.fit(left, focus, right) # inefficient 
        print(vocab_dict.vocab)
        train, valid, test = PairedDataset(
            src=(list(focus), list(left), list(right)), trg=None,
            d=(vocab_dict, vocab_dict, vocab_dict),
            batch_size=args.batch_size, gpu=args.gpu,
            align_right=args.reverse, fitted=True
        ).splits(dev=args.dev, test=args.test, sort=True)

    print(f' * vocabulary size. {len(vocab_dict)}')
    print(f' * number of train batches. {len(train)}')
    print(f' * maximum batch size. {batch_size}')

if __name__ == '__main__':
    main()