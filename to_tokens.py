import argparse
import os
import glob

from nltk.tokenize.moses import MosesTokenizer

"""
Usage:
python to_tokens.py --inp='data' --outp='tokenized.txt' --min_len=1 --max_len=15 --max_items=100

"""


def main():
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--inp', type=str, default='data')
    parser.add_argument('--outp', type=str, default='tokenized.txt')
    parser.add_argument('--min_len', default=1, type=int)
    parser.add_argument('--max_len', default=15, type=int)
    parser.add_argument('--max_items', default=None, type=int)

    args = parser.parse_args()

    if os.path.isdir(args.inp):
        if not args.inp.endswith('/'):
            args.inp += '/'
            filenames = sorted(list(glob.glob(args.inp + '*.txt')))
        else:
            filenames = [args.inp]

    processed = 0
    tokenizer = MosesTokenizer()

    with open(args.outp, 'w') as f:
        for filename in filenames:
            for line in open(filename, 'r'):
                line = ' '.join(line.strip().split())
                if line:
                    try:
                        tokens = tokenizer.tokenize(line)
                    except IndexError:
                        tokens = None
                        continue
                    if tokens and len(tokens) <= args.max_len and \
                             len(tokens) >= args.min_len:
                        f.write(' '.join(tokens) + '\n')
                        processed += 1
                    if args.max_items and processed >= args.max_items:
                        return

if __name__ == '__main__':
    main()