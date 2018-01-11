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

from skipthoughts import make_skipthoughts_model

from seqmod.misc.dataset import PairedDataset, Dict
from seqmod.misc import EarlyStopping, Trainer
from seqmod.misc import StdLogger, VisdomLogger, TensorboardLogger
from seqmod.misc import PairedDataset, Dict, inflection_sigmoid
import seqmod.utils as u

from zorro.data import TripleStore


def translate(model, target, gpu, beam=True):
    src_dict = model.encoder.embeddings.d
    inp = torch.LongTensor(list(src_dict.transform([target]))).transpose(0, 1)
    length = torch.LongTensor([len(target)]) + 2
    inp, length = u.wrap_variables((inp, length), volatile=True, gpu=gpu)
    if beam:
        scores, hyps, _ = model.translate_beam(
            inp, length, beam_width=5, max_decode_len=4)
    else:
        scores, hyps, _ = model.translate(inp, length, max_decode_len=4)

    return scores, hyps


def make_encdec_hook(target, gpu, beam=True):

    def hook(trainer, epoch, batch_num, checkpoint):
        trainer.log("info", "Translating {}".format(target))
        trg_dict = trainer.model.decoder.embeddings.d
        scores, hyps = translate(trainer.model, target, gpu, beam=beam)
        hyps = [u.format_hyp(score, hyp, num + 1, trg_dict)
                for num, (score, hyp) in enumerate(zip(scores, hyps))]
        trainer.log("info", '\n***' + ''.join(hyps) + '\n***')

    return hook


def make_att_hook(target, gpu, beam=False):
    assert not beam, "beam doesn't output attention yet"

    def hook(trainer, epoch, batch_num, checkpoint):
        d = train.decoder.embedding.d
        scores, hyps, atts = translate(trainer.model, target, gpu, beam=beam)
        trainer.log("attention",
                    {"att": atts[0],
                     "score": sum(scores[0]) / len(hyps[0]),
                     "target": [d.bos_token] + list(target),
                     "hyp": ' '.join([d.vocab[i] for i in hyps[0]]).split(),
                     "epoch": epoch,
                     "batch_num": batch_num})

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
    parser.add_argument('--max_triples', default=1000, type=int)
    parser.add_argument('--allow_overlap', action='store_true', default=False)

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
    parser.add_argument('--patience', default=2, type=int)
    parser.add_argument('--reverse', action='store_true')
    parser.add_argument('--checkpoint', default=5, type=int)
    parser.add_argument('--hooks_per_epoch', default=None, type=int)
    parser.add_argument('--target', default='redrum', type=str)
    parser.add_argument('--beam', action='store_true')
    parser.add_argument('--plot', action='store_true')

    # saving
    parser.add_argument('--model_path', default='./', type=str)

    args = parser.parse_args()

    # load the triples:
    triple_store = TripleStore(args.input,
                               allow_overlap=args.allow_overlap,
                               max_triples=args.max_triples)
    triples = list(triple_store)
    print(f'loaded {len(triple_store)} triples')


    # random shuffle:
    print('shuffling batches...')
    random.seed(args.rnd_seed)
    random.shuffle(triples)

    left, focus, right = zip(*triples)
    del triples

    vocab_dict = Dict(pad_token='<PAD>', bos_token='<EOS>', eos_token='<EOS>',
                      min_freq=args.min_char_freq, sequential=True, force_unk=True)
    vocab_dict.fit(left, focus, right) # sometimes inefficient? # do a partial fit in the triple store?
    
    train, valid = PairedDataset(
        src=(focus,), trg=(left, right),
        d={'src': (vocab_dict,), 'trg': (vocab_dict, vocab_dict)},
        batch_size=args.batch_size, gpu=args.gpu,
        align_right=args.reverse, fitted=False).splits(sort_by='src', dev=args.dev, test=None, sort=True)

    print(f' * vocabulary size {len(vocab_dict)}')
    print(f' * number of train batches {len(train)}')
    print(f' * number of dev batches {len(valid)}')
    print(f' * maximum batch size {args.batch_size}')

    args.checkpoint = min(len(train), args.checkpoint)

    model = make_skipthoughts_model(1, 64, 50, vocab_dict, cell='GRU', bidi=True, att_type='general')

    u.initialize_model(model, rnn={'type': 'orthogonal', 'args': {'gain': 1.0}})

    optimizer = getattr(optim, args.optim)(model.parameters(), lr=args.lr)

    print(model)
    print('* number of parameters: {}'.format(model.n_params()))

    if args.gpu:
        model.cuda()

    early_stopping = EarlyStopping(patience=args.patience, maxsize=3)
    trainer = Trainer(
        model, {'train': train, 'valid': valid}, optimizer,
        early_stopping=early_stopping, max_norm=args.max_norm)
    trainer.add_loggers(StdLogger())
    #trainer.add_loggers(TensorboardLogger(comment='encdec'))

    hook = make_encdec_hook(args.target, args.gpu, beam=args.beam)
    trainer.add_hook(hook, num_checkpoints=3)

    hook = u.make_schedule_hook(
        inflection_sigmoid(len(train) * 2, 1.75, inverse=True))
    trainer.add_hook(hook, hooks_per_epoch=200)

    (best_model, valid_loss), test_loss = trainer.train(
        args.epochs, args.checkpoint, shuffle=True,
        use_schedule=args.use_schedule)

    u.save_checkpoint(args.model_path, best_model, vars(args), d=vocab_dict, ppl=valid_loss)



if __name__ == '__main__':
    main()