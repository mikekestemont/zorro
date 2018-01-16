import argparse

import random
random.seed(1001)

import torch
try:
    torch.manual_seed(1001)
    torch.cuda.manual_seed(1001)
except:
    print('no NVIDIA driver found')

from torch import nn, optim

from seqmod.misc.dataset import PairedDataset, Dict
from seqmod.misc import EarlyStopping
from seqmod.misc import StdLogger, VisdomLogger, TensorboardLogger
from seqmod.misc import PairedDataset, Dict, inflection_sigmoid
import seqmod.utils as u

import zorro.utils as uz
from zorro.data import CoupleStore
from zorro.logging import JsonLogger, StdLogger
from zorro.skipthoughts import make_skipthoughts_model, SkipthoughtsTrainer


def make_translation_hook(target, gpu, beam=True, max_len=4):

    def hook(trainer, epoch, batch_num, checkpoint):
        trainer.log("info", "Translating {}".format(target))
        trg_dict = trainer.model.decoder.embeddings.d
        scores, hyps = uz.translate(trainer.model, target, gpu,
                                    beam=beam, max_len=max_len)
        hyps = [u.format_hyp(score, hyp, num + 1, trg_dict)
                for num, (score, hyp) in enumerate(zip(scores, hyps))]
        trainer.log("info", '\n***' + ''.join(hyps) + '\n***')

    return hook


def main():
    parser = argparse.ArgumentParser()
    # dataset
    parser.add_argument('--path', type=str, default=None)
    parser.add_argument('--input', type=str, default='data')
    parser.add_argument('--min_char_freq', type=int, default=10)
    parser.add_argument('--min_len', default=1, type=int)
    parser.add_argument('--max_len', default=15, type=int)
    parser.add_argument('--dev', default=0.1, type=float)
    parser.add_argument('--rnd_seed', default=12345, type=int)
    parser.add_argument('--max_items', default=1000, type=int)
    parser.add_argument('--task', default='couples', type=str)
    parser.add_argument('--focus_size', default=15, type=int)
    parser.add_argument('--left_size', default=15, type=int)
    parser.add_argument('--right_size', default=15, type=int)
    parser.add_argument('--shingle_stride', default=None, type=int)
    parser.add_argument('--shingling', default='characters', type=str)
    parser.add_argument('--allow_overlap', action='store_true', default=False)
    parser.add_argument('--shuffle', action='store_true')
    parser.add_argument('--grow', action='store_true')
    parser.add_argument('--grow_n_epochs', default=3, type=int)

    # training
    parser.add_argument('--epochs', default=5, type=int)
    parser.add_argument('--gpu', action='store_true')
    parser.add_argument('--batch_size', default=30, type=int)
    parser.add_argument('--optim', default='Adam', type=str)
    parser.add_argument('--lr', default=0.01, type=float)
    parser.add_argument('--max_norm', default=10., type=float)
    parser.add_argument('--dropout', default=0.25, type=float)
    parser.add_argument('--word_dropout', default=0.0, type=float)
    parser.add_argument('--use_schedule', action='store_true')
    parser.add_argument('--patience', default=10, type=int)
    parser.add_argument('--reverse', action='store_true')
    parser.add_argument('--checkpoint', default=5, type=int)
    parser.add_argument('--hooks_per_epoch', default=None, type=int)
    parser.add_argument('--target', default='Ze was', type=str)
    parser.add_argument('--bidi', action='store_true')
    parser.add_argument('--beam', action='store_true')
    parser.add_argument('--plot', action='store_true')
    parser.add_argument('--json', type=str, default='history.json')

    # model
    parser.add_argument('--model_path', default='./model_storage', type=str)
    parser.add_argument('--num_layers', default=1, type=int)
    parser.add_argument('--emb_dim', default=64, type=int)
    parser.add_argument('--hid_dim', default=150, type=int)
    parser.add_argument('--cell', default='GRU')
    parser.add_argument('--tie_weights', action='store_true')
    parser.add_argument('--train_init', action='store_true')
    parser.add_argument('--add_init_jitter', action='store_true')
    parser.add_argument('--encoder-summary', default='inner-attention')
    parser.add_argument('--deepout_layers', type=int, default=0)

    args = parser.parse_args()

    train, valid, vocab_dict = uz.shingle_dataset(args,
                                                           vocab_dict=None)

    print(f' * vocabulary size {len(vocab_dict)}')
    print(f' * number of train batches {len(train)}')
    print(f' * number of dev batches {len(valid)}')
    print(f' * maximum batch size {args.batch_size}')

    args.checkpoint = min(len(train), args.checkpoint)

    model = make_skipthoughts_model(num_layers=args.num_layers,
                                    emb_dim=args.emb_dim,
                                    hid_dim=args.hid_dim,
                                    src_dict=vocab_dict,
                                    cell=args.cell,
                                    bidi=args.bidi,
                                    att_type='general',
                                    task=args.task)

    u.initialize_model(model, rnn={'type': 'orthogonal',
                                   'args': {'gain': 1.0}})

    optimizer = getattr(optim, args.optim)(model.parameters(), lr=args.lr)

    print(model)
    print('* number of parameters: {}'.format(model.n_params()))

    if args.gpu:
        model.cuda()

    early_stopping = EarlyStopping(patience=args.patience, maxsize=5)
    trainer = SkipthoughtsTrainer(
        model, {'train': train, 'valid': valid}, optimizer,
        early_stopping=early_stopping, max_norm=args.max_norm)

    if args.json:
        logger = JsonLogger(json_file=args.json)
    else:
        logger = StdLogger()
    trainer.add_loggers(logger)

    trainer.set_additional_params(args, vocab_dict)

    hook = make_translation_hook(args.target, args.gpu,
                                 beam=args.beam, max_len=args.right_size)
    trainer.add_hook(hook, num_checkpoints=3)

    hook = u.make_schedule_hook(
        inflection_sigmoid(len(train) * 2, 1.75, inverse=True))
    trainer.add_hook(hook, hooks_per_epoch=200)

    (best_model, valid_loss), test_loss = trainer.train(
        args.epochs, args.checkpoint, shuffle=True,
        use_schedule=args.use_schedule)

    u.save_checkpoint(args.model_path, best_model, vars(args),
                      d=vocab_dict, ppl=valid_loss, suffix='final')


if __name__ == '__main__':
    main()
