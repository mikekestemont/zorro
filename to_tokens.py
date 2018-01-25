import argparse
import os
import glob

from nltk import sent_tokenize, word_tokenize

"""
Usage:
python to_tokens.py --inp="/home/mike/GitRepos/zorro/data/EN" --outp="data/en_tokenized_10M.txt" --min_len=1 --max_len=25 --max_items=100000 --sent_tokenize

"""


def main():
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--inp', type=str, default='data')
    parser.add_argument('--outp', type=str, default='tokenized.txt')
    parser.add_argument('--min_len', default=1, type=int)
    parser.add_argument('--max_len', default=15, type=int)
    parser.add_argument('--max_items', default=None, type=int)
    parser.add_argument('--sent_tokenize', action='store_true')

    args = parser.parse_args()

    if os.path.isdir(args.inp):
        if not args.inp.endswith('/'):
            args.inp += '/'
        filenames = sorted(list(glob.glob(args.inp + '*/*.txt')))
    else:
        filenames = [args.inp]

    processed = 0

    with open(args.outp, 'w') as f:
        for filename in filenames:
            print(filename)
            if not args.sent_tokenize:
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
            else:
                try:
                    with open(filename, 'r') as nf:
                        text = ' '.join(nf.read().strip().split())
                except UnicodeDecodeError:
                    continue
                
                for sentence in sent_tokenize(text):
                    if sentence:
                        try:
                            tokens = word_tokenize(sentence)
                        except IndexError:
                            tokens = None
                            continue
                        if tokens and len(tokens) <= args.max_len and \
                                 len(tokens) >= args.min_len:
                            tokens = [t.lower() for t in tokens]
                            f.write(' '.join(tokens) + '\n')
                            processed += 1
                        if args.max_items and processed >= args.max_items:
                            return

if __name__ == '__main__':
    main()