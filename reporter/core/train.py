from logging import Logger
from typing import Dict, List

import numpy
import torch
from nltk.translate.bleu_score import SmoothingFunction, sentence_bleu
try:
    from torchtext.data import Iterator
except ImportError:
    from torchtext.legacy.data import Iterator

from torchtext.vocab import Vocab

from reporter.core.network import Attention, EncoderDecoder
from reporter.core.operation import (
    get_latest_closing_vals,
    replace_tags_with_vals
)
from reporter.postprocessing.text import remove_bos
from reporter.util.constant import SEED, Code, Phase, SeqType, SpecialToken, summarizer_map
from reporter.util.conversion import stringify_ric_seqtype
from reporter.util.tool import takeuntil
from proto_lib import getTruthValue
import gc
import time

class RunResult:
    def __init__(self,
                 loss: float,
                 summary_ids: List[str],
                 gold_sents: List[List[str]],
                 pred_sents: List[List[str]]):

        self.loss = loss
        self.summary_ids = summary_ids
        self.gold_sents = gold_sents
        self.pred_sents = pred_sents

def run(X: Iterator,
        num_batches: int,
        vocab: Vocab,
        model,
        optimizer: Dict[SeqType, torch.optim.Optimizer],
        criterion: torch.nn.modules.Module,
        phase: Phase,
        device: torch.device,
        learn_templates: bool = False,
        transformer: bool = False) -> RunResult:

    if phase in [Phase.Valid, Phase.Test]:
        with torch.no_grad():
            model.eval()
    else:
        model.train()

    numpy.random.seed(SEED)
    #max_sentence_length = 20
    accum_loss = 0.0
    all_summary_ids = []
    all_gold_sents = []
    all_pred_sents = []
    all_gold_sents_with_number = []
    all_pred_sents_with_number = []
    attn_weights = []
    truth_values = []

    import numpy as np
    batch_idx = 0

    check_memory = False

    for batch in X:
        optimizer.zero_grad()
        model.to(device)

        if check_memory:
            #print(torch.__version__)
            print("\n\nBatch",batch_idx,str(batch_idx)+"/"+str(num_batches),str(int((float(batch_idx)/num_batches)*100))+'%')

            print("\nbatch")
            t = torch.cuda.get_device_properties(0).total_memory
            r = torch.cuda.memory_reserved(0)
            #c = torch.cuda.memory_cached(0)
            a = torch.cuda.memory_allocated(0)
            f = r-a  # free inside reserved
            print("Total memory:", t)
            print("Memory reserved:", r)
            #print("Memory cached:", c)
            print("Memory allocated:", a)
            print("Memory free:", f)

        batch_idx+=1
        tokens = batch.token.to(device)
        summary_ids = batch.summary_id

        max_n_tokens, _ = tokens.size()

        latest_vals = batch.passengers
        truth_values = batch.truth

        sax_reps = batch.sax
        if learn_templates:
            templates = batch.template
            from collections import Counter
            word_dict = dict()
            for template in templates:
                for word in template:
                    if word in word_dict:
                        word_dict[word] += 1
                    else:
                        word_dict[word] = 1
            template_cnt = Counter(word_dict)
            template_vocab = Vocab(template_cnt)


            max_len = max([len(x) for x in templates])
            template_tokens = []
            for template in templates:
                sublist = []

                for word in template:
                    sublist.append(int(template_vocab.stoi[word]))
                if len(template) < max_len:
                    for i in range(max_len-len(template)):
                        sublist.append(template_vocab.stoi[SpecialToken.Padding.value])
                template_tokens.append(np.array(sublist))

            del templates
            template_tokens = torch.tensor(template_tokens).t()
            template_tokens = template_tokens.long()
        else:
            template_tokens = None
            template_vocab = None

        #if transformer:
            #model.zero_grad()
            #loss = 0.0
            #pred = []

            #x = torch.rand((10, 32, 512))
            #y = torch.rand((20, 32, 512))
            #out = model(x,y)
            #input(x)

            #out = out.view(out.shape[1],out.shape[2])

            #for i in range(max_n_tokens):
                #tokens_ = tokens[i].tolist()

                #n = len(tokens_)
                #tokens_ += [vocab.stoi[SpecialToken.Padding.value]] * (out.shape[0] - n)
                #tokens_ = torch.tensor(tokens_)
                #try:
                    #loss += criterion(out,tokens_)
                #except:
                    #print(out.shape,tokens_.shape)
                    #loss += criterion(out,tokens_)
                #topv, topi = out.data.topk(1)
                #pred.append([t[0] for t in topi.cpu().numpy()])

        #else:

        if check_memory:
            print("\nBefore forward pass")
            t = torch.cuda.get_device_properties(0).total_memory
            r = torch.cuda.memory_reserved(0)
            #c = torch.cuda.memory_cached(0)
            a = torch.cuda.memory_allocated(0)
            f = r-a  # free inside reserved
            print("Total memory:", t)
            print("Memory reserved:", r)
            #print("Memory cached:", c)
            print("Memory allocated:", a)
            print("Memory free:", f)


        loss, pred, attn_weight = model(batch, batch.batch_size, tokens, template_tokens, vocab, criterion, phase, template_vocab)
        #input(dict(model.named_parameters()))
        #from torchviz import make_dot
        #dot = make_dot(torch.tensor(pred).float().mean(),params=dict(model.named_parameters()),show_attrs=True,show_saved=True)
        #input(dot.source)

        template_tokens.detach()
        if check_memory:
            print("\nAfter forward pass")
            t = torch.cuda.get_device_properties(0).total_memory
            r = torch.cuda.memory_reserved(0)
            #c = torch.cuda.memory_cached(0)
            a = torch.cuda.memory_allocated(0)
            f = r-a  # free inside reserved
            print("Total memory:", t)
            print("Memory reserved:", r)
            #print("Memory cached:", c)
            print("Memory allocated:", a)
            print("Memory free:", f)

        if phase == Phase.Train:

            if check_memory:
                print("\nzero_grad")
                t = torch.cuda.get_device_properties(0).total_memory
                r = torch.cuda.memory_reserved(0)
                #c = torch.cuda.memory_cached(0)
                a = torch.cuda.memory_allocated(0)
                f = r-a  # free inside reserved
                print("Total memory:", t)
                print("Memory reserved:", r)
                #print("Memory cached:", c)
                print("Memory allocated:", a)
                print("Memory free:", f)

            loss.backward()

            if check_memory:

                print("\nbackpropagation")
                t = torch.cuda.get_device_properties(0).total_memory
                r = torch.cuda.memory_reserved(0)
                #c = torch.cuda.memory_cached(0)
                a = torch.cuda.memory_allocated(0)
                f = r-a  # free inside reserved
                print("Total memory:", t)
                print("Memory reserved:", r)
                #print("Memory cached:", c)
                print("Memory allocated:", a)
                print("Memory free:", f)

            optimizer.step()

            if check_memory:
                print("\nstep")
                t = torch.cuda.get_device_properties(0).total_memory
                r = torch.cuda.memory_reserved(0)
                #c = torch.cuda.memory_cached(0)
                a = torch.cuda.memory_allocated(0)
                f = r-a  # free inside reserved
                print("Total memory:", t)
                print("Memory reserved:", r)
                #print("Memory cached:", c)
                print("Memory allocated:", a)
                print("Memory free:", f)

        loss.detach()


        if not transformer and isinstance(model.decoder.attn, Attention):
            attn_weight = numpy.array(list(zip(*attn_weight)))
            attn_weights.extend(attn_weight)

        all_summary_ids.extend(summary_ids)

        i_eos = vocab.stoi[SpecialToken.EOS.value]
        # Recover words from ids removing BOS and EOS from gold sentences for evaluation
        gold_sents = [remove_bos([vocab.itos[i] for i in takeuntil(i_eos, sent)])
                      for sent in zip(*tokens.cpu().numpy())]
        tokens.detach()

        all_gold_sents.extend(gold_sents)
        pred_sents = []

        try:
            pred_sents = [remove_bos([vocab.itos[i] for i in takeuntil(i_eos, sent)]) for sent in zip(*pred)]
        except:
            vocab.itos = np.array(vocab.itos)
            index_list = []
            for sent in zip(*pred):
                for i in takeuntil(i_eos, sent):
                    index_list.append(i)
            #index_list = [i for i in takeuntil(i_eos, sent) for sent in zip(*pred)]
            print(index_list)
            pred_sents = [remove_bos([vocab.itos[i] for i in takeuntil(i_eos, sent)]) for sent in zip(*pred)]
        all_pred_sents.extend(pred_sents)

        if phase == Phase.Test:
            z_iter = zip(gold_sents, pred_sents, truth_values, sax_reps, latest_vals)
            for (gold_sent, pred_sent, gold_truth, sax_rep, latest_val) in z_iter:
                bleu = sentence_bleu([gold_sent],
                                     pred_sent,
                                     smoothing_function=SmoothingFunction().method1)

                predicted = pred_sent[:-1]
                pred_truth = getTruthValue(predicted,sax_rep)

                description = \
                    '\n'.join(['=== {} ==='.format(phase.value.upper()),
                               'Gold (tag): {}'.format(' '.join(gold_sent)),
                               'Gold (truth): {}'.format(gold_truth),
                               'Pred (tag): {}'.format(' '.join(pred_sent)),
                               'Pred (truth): {}'.format(pred_truth),
                               'Data sequence: {}'.format(latest_val),
                               'Corresponding SAX: {}'.format(sax_rep),
                               'BLEU: {:.5f}'.format(bleu),
                               'Loss: {:.5f}'.format(loss.item() / max_n_tokens)])
                print(description)  # TODO: info → debug in release

        try:
            accum_loss += float(loss.item()) / max_n_tokens
        except:
            accum_loss += float(loss.item()) / max_n_tokens[0]

        if batch_idx >= num_batches:
            break

        #torch.cuda.empty_cache()
        gc.collect()
        #time.sleep(20)

        if check_memory:


            print("\naccum_loss")
            t = torch.cuda.get_device_properties(0).total_memory
            r = torch.cuda.memory_reserved(0)
            #c = torch.cuda.memory_cached(0)
            a = torch.cuda.memory_allocated(0)
            f = r-a  # free inside reserved
            print("Total memory:", t)
            print("Memory reserved:", r)
            #print("Memory cached:", c)
            print("Memory allocated:", a)
            print("Memory free:", f)


    model.cpu()

    if check_memory:


        print("\nMove model back to CPU")
        t = torch.cuda.get_device_properties(0).total_memory
        r = torch.cuda.memory_reserved(0)
        #c = torch.cuda.memory_cached(0)
        a = torch.cuda.memory_allocated(0)
        f = r-a  # free inside reserved
        print("Total memory:", t)
        print("Memory reserved:", r)
        #print("Memory cached:", c)
        print("Memory allocated:", a)
        print("Memory free:", f)

    return RunResult(accum_loss,
                     all_summary_ids,
                     all_gold_sents,
                     all_pred_sents)