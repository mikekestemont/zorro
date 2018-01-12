import torch
from seqmod import utils as u

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