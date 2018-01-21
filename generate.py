"""
Usage:
CUDA_VISIBLE_DEVICES=0 \
python generate.py --model_path="./Skipthoughts-2018_01_21-12_22_44-40.014-final/model.pt" \
  --file_path="tokenized.txt" --beam --gpu --max_len=25 \
  --dict_path="./Skipthoughts-2018_01_21-12_22_44-40.014-final/model.dict.pt" \
  --target="Ze hield er veel van hem."


"""

import argparse

import random
random.seed(1001)

import torch
try:
    torch.manual_seed(1001)
    torch.cuda.manual_seed(1001)
except:
    print('no NVIDIA driver found')

import numpy as np
import scipy.spatial.distance as sd

import seqmod.utils as u
import zorro.utils

from nltk.tokenizer import MosesTokenizer

def main():
    # parse params:
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', default='./Skipthoughts-2018_01_21-12_22_44-40.014-final/model.pt', type=str)
    parser.add_argument('--file_path', default='big.txt', type=str)
    parser.add_argument('--dict_path', default='./Skipthoughts-2018_01_21-12_22_44-40.014-final/model.dict.pt', type=str)
    parser.add_argument('--beam', action='store_true')
    parser.add_argument('--gpu', action='store_true')
    parser.add_argument('--max_len', default=4, type=int)
    parser.add_argument('--target', default=None, type=str)
    args = parser.parse_args()

    # load model and dict
    model = u.load_model(args.model_path)
    vocab_dict = u.load_model(args.dict_path)

    # translate the target:
    if args.target:
        tokenizer = MosesTokenizer()
        tokens = tokenizer.tokenize(args.target)
        tokens = [t.lower() for t in tokens]

        scores, hyps = zorro.utils.translate(model, tokens,
                                             beam=args.beam,
                                             max_len=args.max_len)
        hyps = [u.format_hyp(score, hyp, num + 1, vocab_dict)
                for num, (score, hyp) in enumerate(zip(scores, hyps))]
        print(f'Translation for "{args.target}":\n',
              '\n***' + ''.join(hyps) + '\n***')

    """
    # embed sentences/lines from a single document:
    lines = [' '.join(line.strip().split())
             for line in open(args.file_path, 'r')]
    lines = [l for l in lines if l]
    encodings = np.array([zorro.utils.embed_single(model, l)
                          for l in lines])

    # find n nearest neighbors for each line:
    for idx, line in enumerate(lines):
        encoding = encodings[idx, :]
        scores = sd.cdist([encoding], encodings, "cosine")[0]
        sorted_ids = np.argsort(scores)
        print('\nSentence:')
        print('', lines[idx][:75])
        print('\nNearest neighbors:')
        for i in range(1, 5 + 1):
            print(' %d. %s (%.3f)' %
                 (i, lines[sorted_ids[i]][:75], scores[sorted_ids[i]]))
    """


if __name__ == '__main__':
    main()
