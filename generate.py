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
import zorro.utils


def main():
    parser = argparse.ArgumentParser()

    # model
    parser.add_argument('--model_path', default='./Skipthoughts-2018_01_12-11_19_59-6.275/model.pt', type=str)
    parser.add_argument('--dict_path', default='./Skipthoughts-2018_01_12-11_19_59-6.275/model.dict.pt', type=str)
    parser.add_argument('--beam', action='store_true')
    parser.add_argument('--gpu', action='store_true')
    parser.add_argument('--max_len', default=4, type=int)
    parser.add_argument('--target', default='Ze was', type=str)

    args = parser.parse_args()

    model = u.load_model(args.model_path)
    vocab_dict = u.load_model(args.dict_path)
    scores, hyps = zorro.utils.translate(model, args.target, args.gpu,
                                 beam=args.beam, max_len=args.max_len)
    hyps = [u.format_hyp(score, hyp, num + 1, vocab_dict)
                for num, (score, hyp) in enumerate(zip(scores, hyps))]
    print(f"Translation for '{args.target}':\n", '\n***' + ''.join(hyps) + '\n***')

if __name__ == '__main__':
    main()