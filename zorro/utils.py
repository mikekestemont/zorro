import torch
from seqmod import utils as u
from .data import *
from seqmod.misc.dataset import PairedDataset, Dict

def translate(model, target, gpu, beam=True, max_len=4):
    src_dict = model.encoder.embeddings.d
    inp = torch.LongTensor(list(src_dict.transform([target]))).transpose(0, 1)
    length = torch.LongTensor([len(target)]) + 2
    inp, length = u.wrap_variables((inp, length), volatile=True, gpu=gpu)
    if beam:
        scores, hyps, _ = model.translate_beam(
            inp, length, beam_width=5, max_decode_len=max_len)
    else:
        scores, hyps, _ = model.translate(inp, length, max_decode_len=max_len)

    return scores, hyps

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