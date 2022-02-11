import datetime

import numpy as np
from matplotlib import pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split

try:
    from torchtext.data import RawField, Field, Example, Dataset, Iterator, BucketIterator
except ImportError:
    from torchtext.legacy.data import RawField, Field, Example, Dataset, Iterator, BucketIterator
from tqdm import tqdm
import seaborn as sns

from tst import Transformer
from tst.loss import OZELoss

from src.visualization import map_plot_function, plot_values_distribution, plot_error_distribution, plot_errors_threshold, plot_visual_sample

from ED import *
from reporter.postprocessing.bleu import calc_bleu

from bisect import bisect_left
import copy

UNK_IDX = 0
PAD_IDX = 1
START_IDX = 2
END_IDX = 3
MAX_SUMM_LEN = None
MAX_SEQ_LEN = None
MAX_LEN = None

#config = None

#config = {
    #'use_dropout' : True,
    #'learning_rate' : 1e-4, # 0.1 with lr_scheduler
    #'n_epochs' : 30, # 30
    #'num_workers' : 0,
    #'batch_size' : 8, # 8
    #'use_preprocessing' : False,
    #'use_series' : True,
    #'use_decoder' : False,
    #'sax_bool' : False,
    #'sax_bool2' : False,
    #'learn_templates' : True, # True
    #'cross_entropy' : True,
    #'truth_loss' : False,
    #'concat_sax' : False,
    #'usr_divide' : True,
    #'usr_shuffle' : True,
    #'summarizer_loss' : True, # True
    #'quantifier_loss' : True, # True
    #'summarizer_training_loss' : True, # True
    #'weekday_loss' : True,
    #'epoch_select': 'bleu',
    #'summary_types' : ["DB"],
    #'time_windows' : [7],
    #'test_percentage' : 0.15,
    #'start_end_tokens' : True,
    #'dropout' : 0.2,
    #'d_model' : 64, # TODO: Try 128 again (original is 64)
    #'query_size' : 8,
    #'value_size' : 8,
    #'num_heads' : 4,
    #'n_stacks' : 4, # TODO: Try 5 again (original is 4)
    #'attention_size' : 12, # TODO: Try 24 again (original is 12)
    #'positional_encoding' : 'original',
    #'chunk_mode' : None,
    #'test_analysis' : False,
    #'squish_tokens' : True,
    #'ge_norm' : False,
    #'schedule_lr' : False,
    #'step_lr' : False,
    #'reduce_lr' : False,
    #'include_templates' : True,
    #'short_patience' : 5,
    #'long_patience' : 20,
    #'remove_unk' : False,
#}

def add_start_end_tokens(inout_seq,vocab,START_IDX,END_IDX):
    for i in range(len(inout_seq)):
        inout_seq[i] = list(inout_seq[i])
        try:
            inout_seq[i][1] = vocab.itos[START_IDX] + ' ' + inout_seq[i][1] + ' ' + vocab.itos[END_IDX]
        except:
            inout_seq[i][2] = vocab.itos[START_IDX] + ' ' + inout_seq[i][2] + ' ' + vocab.itos[END_IDX]
        inout_seq[i] = tuple(inout_seq[i])
    return inout_seq

def summary_tokens(summary,vocab_stoi,max_len):
    if type(summary) is str:
        summary = summary.split(" ")
    tokens = [vocab_stoi[token] for token in summary]
    for i in range(max_len-len(tokens)):
        tokens.append(PAD_IDX)
    return tokens

def add_temp_loss(netout_,y_,BATCH_SIZE,MAX_SUMM_LEN,MAX_LEN,vocab,template_vocab_stoi,special_chars,ntoken,device,loss_function,summarizer_loss=True,quantifier_loss=True,summarizer_training_loss=True,tw_loss=True,weekday_loss=True,squish_tokens=False):
    preds = []
    pred_strs = []
    trgs = []
    for i in range(BATCH_SIZE):
        pred = netout_[i]
        pred_sent = []
        pred_str = ""
        trg_sent = ""
        for j in range(MAX_SUMM_LEN):
            topv, topi = netout_[i][j][:].data.topk(1)
            token_idx = int(topi[0].item())
            pred_token = vocab.itos[token_idx]

            pred_sent.append(combine_special_tokens(pred_token.strip(),reverse=True))
            pred_str += pred_token.strip()

            if token_idx == END_IDX:
                break

            if j != MAX_LEN-1:
                pred_str += " "

        for j in range(len(y_[i])):
            trg_token = vocab.itos[y_[i][j]]
            if trg_token not in special_chars: # For template
                trg_sent += trg_token

            if trg_token == vocab.itos[END_IDX]:
                break

            if j != len(y_[i])-1:
                trg_sent += " "

        preds.append(pred_sent)
        pred_strs.append(combine_special_tokens(pred_str.strip(),reverse=True))
        trgs.append(combine_special_tokens(trg_sent.strip(),reverse=True))

    # Template loss
    template_tokens = []
    for summary in trgs:
        #summary_ = summary.split(" ")
        #summary_ = [summary_[i].strip() for i in range(len(summary_))]
        if squish_tokens:
            summary = combine_special_tokens(summary,reverse=True)
        template_tokens.append(build_template(summary,single=True))
        #template_tokens.append(summary_)

    multiplier = 1
    template_chars = set(["A","S","Q","TW"])
    temp_loss = 0

    pred_output = []
    trg_output = []
    for i in range(BATCH_SIZE):
        for j in range(len(template_tokens[i])):
            token = template_tokens[i][j]
            try:
                pred_token = preds[i][j]
            except:
                pred_token = None
            if pred_token != token and token in template_chars:
                multiplier += 1

                if token == 'S' and (summarizer_loss or summarizer_training_loss):
                    multiplier += 1

                if token == 'Q' and quantifier_loss:
                    multiplier += 1

                if token == 'TW' and tw_loss:
                    multiplier += 1

                if token == 'D' and weekday_loss:
                    multiplier += 1

        max_len = max([len(pred_strs[i]),len(template_tokens[i])])
        #print(template_vocab_stoi)
        #print(build_template(pred_strs[i]))
        preds = [template_vocab_stoi[elem] for elem in build_template(pred_strs[i])]
        preds += [(template_vocab_stoi[vocab.itos[PAD_IDX]])] * max([0,max_len - len(preds)])
        for elem in preds:
            zeros = [0.0]*ntoken
            zeros[elem] = 1.0
            pred_output.append(zeros)

        trg_output += [template_vocab_stoi[elem] for elem in template_tokens[i]] + [(template_vocab_stoi[vocab.itos[PAD_IDX]])] * max([0,max_len - len(template_tokens[i])])

    pred_output = torch.tensor(pred_output).float()
    trg_output = torch.tensor(trg_output).long()

    return multiplier*loss_function(pred_output.to(device), trg_output.to(device))

def compute_accuracy(netout,y,model_sent,y_sent,model_sent_,y_sent_,num_correct,num_correct_,num_total,BATCH_SIZE,MAX_LEN,ntoken,vocab,special_chars,dataset_size,print_preds=False):
    netout = netout.view(BATCH_SIZE,MAX_LEN,ntoken)
    y = y.view(BATCH_SIZE,MAX_LEN)

    for i in range(BATCH_SIZE):
        pred = pred_ = ""
        trg = trg_ = ""
        for j in range(MAX_LEN):
            topv, topi = netout[i][j][:].data.topk(1)
            try:
                token_idx = int(topi[0].item())
                pred_token = vocab.itos[token_idx]
            except:
                print("Error retrieving pred token in test set")
                input([topi,netout[i][j][:].data])

            if pred_token not in special_chars:
                pred_ += pred_token
            pred += pred_token

            if token_idx == END_IDX:
                break

            if j != MAX_LEN-1:
                pred += " "
                pred_ += " "

        for j in range(len(y[i])):
            trg_token = vocab.itos[y[i][j]]

            if trg_token not in special_chars:
                trg_ += trg_token
            trg += trg_token

            if trg_token == vocab.itos[END_IDX]:
                break

            if j != len(y[i])-1:
                trg += " "
                trg_ += " "

        model_sent.append(pred)
        y_sent.append(trg)
        model_sent_.append(pred_)
        y_sent_.append(trg_)

        if pred == trg:
            num_correct += 1
        if pred_ == trg_:
            num_correct_ += 1

        num_total += 1

        if print_preds:
            print("Prediction:", pred)
            print("Actual:", trg)
            print("Prediction (no special):", pred_)
            print("Actual (no special):", trg_)
            print()

    return num_correct, num_correct_, num_total, model_sent, y_sent, model_sent_, y_sent_

def compute_loss(net: torch.nn.Module,
                 dataloader: torch.utils.data.DataLoader,
                 loss_function: torch.nn.Module,
                 ntoken: int,
                 MAX_SUMM_LEN: int,
                 vocab: Vocab,
                 template_vocab_stoi: dict,
                 special_chars,
                 device: torch.device = 'cpu',
                 batch_size: int = 8,
                 d_input: int = 1,
                 squish_tokens: bool = False,
                 output_seq_len: int = 12,
                 shift_right: bool = False) -> torch.Tensor:
    """Compute the loss of a network on a given dataset.

    Does not compute gradient.

    Parameters
    ----------
    net:
        Network to evaluate.
    dataloader:
        Iterator on the dataset.
    loss_function:
        Loss function to compute.
    device:
        Torch device, or :py:class:`str`.

    Returns
    -------
    Loss as a tensor with no grad.
    """
    num_correct = num_correct_ = num_total = 0
    model_sent = y_sent = model_sent_ = y_sent_ = []
    running_loss = 0
    with torch.no_grad():
        for x, y, z in dataloader:
            x = x.view(batch_size,-1,d_input)
            if shift_right:
                netout, netout_temp = net(x.to(device),device,copy.deepcopy(y).to(device))
            else:
                netout, netout_temp = net(x.to(device),device)
            y = y.view(-1).long()
            if config['learn_templates']:
                z = z.view(-1).long()
            num_correct, num_correct_, num_total, model_sent, y_sent, model_sent_, y_sent_ = compute_accuracy(netout,y,model_sent,y_sent,model_sent_,y_sent_,num_correct,num_correct_,num_total,batch_size,MAX_LEN,ntoken,vocab,special_chars,len(dataloader))
            running_loss += loss_function(netout.to(device), y.to(device))
            if config['learn_templates']:
                running_loss += loss_function(netout_temp.to(device), z.to(device))
                netout_ = netout.view(batch_size,-1,ntoken)
                y_ = y.view(batch_size,-1)
                running_loss += add_temp_loss(netout_,y_,batch_size,MAX_SUMM_LEN,MAX_LEN,vocab,template_vocab_stoi,special_chars,ntoken,device,loss_function,squish_tokens=squish_tokens)

    return running_loss / len(dataloader), float(num_correct)/num_total, calc_bleu(y_sent,model_sent)


def main():
    global MAX_LEN

    # Model parameters
    d_input = 1 # From dataset

    # Config
    sns.set()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu") #<- "cuda:0"

    global config
    with open('transformer_config.json','r') as jsonfile:
        config = json.load(jsonfile)
    #print(f"Using device {device}")

    # Model flags
    usr_divide = config['usr_divide']
    concat_sax = config['concat_sax']
    #full_data = config['full_data']
    full_data = (str(device) != "cpu")
    usr_shuffle = config['usr_shuffle']
    sax_bool = config['sax_bool']
    sax_bool2 = config['sax_bool2']
    summary_types = config['summary_types']
    time_windows = config['time_windows']
    use_series = config['use_series']
    BATCH_SIZE = config['batch_size']
    #one_epoch = (str(device) == "cpu")
    one_epoch = False
    EPOCHS = config['n_epochs'] if not one_epoch else 1
    NUM_WORKERS = config['num_workers']
    LR = config['learning_rate']
    start_end_tokens = config['start_end_tokens']
    dropout = config['dropout'] # 0.2
    summarizer_loss = config['summarizer_loss']
    quantifier_loss = config['quantifier_loss' ]
    summarizer_training_loss = config['summarizer_training_loss']
    learn_templates = config['learn_templates']
    d_model = config['d_model'] # Lattent dim
    q = config['query_size'] # Query size
    v = config['value_size'] # Value size
    h = config['num_heads'] # Number of heads
    N = config['n_stacks'] # Number of encoder and decoder to stack
    attention_size = config['attention_size'] # Attention window size
    pe = config['positional_encoding'] # Positional encoding
    chunk_mode = config['chunk_mode']
    squish_tokens = config['squish_tokens']
    ge_norm = config['ge_norm']
    schedule_lr = config['schedule_lr']
    steplr = config['schedule_lr'] and config['step_lr']
    reducelr = config['schedule_lr'] and config['reduce_lr']
    include_templates = config['include_templates']
    remove_unk = config['remove_unk']
    include_dates = ("DB" in summary_types or "GA" in summary_types)
    shift_right = config['shift_right']
    continue_train = config['continue_train']

    for summary_type in summary_types:

        log_cnt = len(os.listdir("tst/log"))+1
        dest_log = "tst/log/log_" + summary_types[0] + "_" + str(log_cnt) + ".csv"

        # Get data
        #all_data, train_dates, valid_dates, test_dates, train_data, train_norm, train_sax, valid_data, valid_norm, valid_sax, test_data, test_norm, test_sax, max_size, test_data_size = get_data(usr_divide,concat_sax,full_data,config,usr_shuffle=usr_shuffle,include_dates=include_dates,return_dataset=True)

        # Create Fields
        passenger_field = RawField()
        series_field = RawField()
        if remove_unk:
            global PAD_IDX, START_IDX, END_IDX
            PAD_IDX -= 1
            START_IDX -= 1
            END_IDX -= 1
            token_field = Field(use_vocab=True,
                                init_token=SpecialToken.BOS.value,
                      eos_token=SpecialToken.EOS.value,
                      pad_token=SpecialToken.Padding.value,
                                    unk_token=None)
        else:
            token_field = Field(use_vocab=True,
                                init_token=SpecialToken.BOS.value,
                      eos_token=SpecialToken.EOS.value,
                      pad_token=SpecialToken.Padding.value,
                      unk_token=SpecialToken.Unknown.value)
        token_field.is_target = True
        truth_field = RawField()
        truth_field.is_target = False
        sax_field = RawField()
        sax_field.is_target = False
        id_field = RawField()
        id_field.is_target = False
        if remove_unk:
            template_field = Field(use_vocab=True,
                                    init_token=SpecialToken.BOS.value,
                                    eos_token=SpecialToken.EOS.value,
                                    pad_token=SpecialToken.Padding.value,
                                    unk_token=None)
        else:
            template_field = Field(use_vocab=True,
                            init_token=SpecialToken.BOS.value,
                  eos_token=SpecialToken.EOS.value,
                  pad_token=SpecialToken.Padding.value,
                  unk_token=SpecialToken.Unknown.value)
        template_field.is_target = True
        fields = [('passengers',passenger_field),('token',token_field)]
        template_fields = [('passengers',passenger_field),('template',template_field)]

        # Get sequences
        if summary_type in seq_dict.keys():
            filename = seq_dict[summary_type]
        else:
            filename = summary_type + "_" + str(device) + ".csv"

        train_inout_seq, valid_inout_seq, test_inout_seq, max_series_size, max_seq_size, test_data_size = load_seq_data(filename,config,transformer=True,quick=True,squish_tokens=squish_tokens)
        config['max_series_size'] = max_series_size
        train_valid_inout_seq = train_inout_seq + valid_inout_seq

        # Build summary and template vocab
        train_examples = []
        train_valid_examples = []
        for seq in train_inout_seq:
            train_examples.append(Example.fromlist(seq,fields))

        for seq in train_valid_inout_seq:
            train_valid_examples.append(Example.fromlist(seq,fields))

        train = Dataset(train_examples,fields)
        train_valid = Dataset(train_valid_examples,fields)
        token_field.build_vocab(train_valid,min_freq=1)

        vocab_path = 'tst/vocab/tst_' + summary_type + '.txt'

        if continue_train:
            # Load vocabulary
            vocab = load_vocab(vocab_path)
        else:
            vocab = token_field.vocab

        special_chars = vocab.itos[0:4]
        ntoken = len(vocab)
        ntemptoken = ntoken
        template_vocab_stoi = template_vocab_itos = None
        if learn_templates:

            from collections import Counter
            word_dict = dict()
            for seq in train_inout_seq:
                summary = seq[1]
                if squish_tokens:
                    summary = combine_special_tokens(summary,reverse=True)
                template = build_template(summary,single=squish_tokens)

                for word in template:
                    if word in word_dict:
                        word_dict[word] += 1
                    else:
                        word_dict[word] = 1

            template_cnt = Counter(word_dict)
            template_vocab_stoi = dict()
            if remove_unk:
                template_vocab_itos = [vocab.itos[PAD_IDX]] + list(template_cnt.keys())
                #template_vocab_stoi = {vocab.itos[PAD_IDX] : vocab.stoi[vocab.itos[PAD_IDX]],}
            else:
                template_vocab_itos = [vocab.itos[UNK_IDX],vocab.itos[PAD_IDX]] + list(template_cnt.keys())
                #template_vocab_stoi = {vocab.itos[UNK_IDX] : vocab.stoi[vocab.itos[UNK_IDX]],
                                   #vocab.itos[PAD_IDX] : vocab.stoi[vocab.itos[PAD_IDX]],}
            for i in range(len(template_vocab_itos)):
                token = template_vocab_itos[i]
                template_vocab_stoi[token] = len(template_vocab_stoi.keys())

            ntemptoken = len(template_vocab_stoi.keys())

        # Add start and end tokens to summaries
        if start_end_tokens:
            train_inout_seq = add_start_end_tokens(train_inout_seq,vocab,START_IDX,END_IDX)
            valid_inout_seq = add_start_end_tokens(valid_inout_seq,vocab,START_IDX,END_IDX)
            test_inout_seq = add_start_end_tokens(test_inout_seq,vocab,START_IDX,END_IDX)


        # Check for inclusion of series in inout_seq
        #if config["summary_types"][0] in seq_dict.keys():
            #summary_index = 2
            #temp_index = -1
        #else:
        summary_index = 1
        temp_index = -1

        # Find max summary and sequence length
        train_summ_len = max([len(seq[summary_index].split(" ")) for seq in train_inout_seq])
        valid_summ_len = max([len(seq[summary_index].split(" ")) for seq in valid_inout_seq])
        test_summ_len = max([len(seq[summary_index].split(" ")) for seq in test_inout_seq])

        train_temp_len = max([len(seq[temp_index]) for seq in train_inout_seq])
        valid_temp_len = max([len(seq[temp_index]) for seq in valid_inout_seq])
        test_temp_len = max([len(seq[temp_index]) for seq in test_inout_seq])

        train_seq_len = max([len(seq[0]) for seq in train_inout_seq])
        valid_seq_len = max([len(seq[0]) for seq in valid_inout_seq])
        test_seq_len = max([len(seq[0]) for seq in test_inout_seq])

        MAX_SUMM_LEN = max([train_summ_len,valid_summ_len,test_summ_len])
        MAX_SEQ_LEN = max([train_seq_len,valid_seq_len,test_seq_len])

        MAX_LEN = max([MAX_SUMM_LEN,MAX_SEQ_LEN])

        # Turn summaries/templates into tokens and pad time series
        for i in range(len(train_inout_seq)):
            train_inout_seq[i] = list(train_inout_seq[i])
            train_inout_seq[i][0] = torch.tensor(list(train_inout_seq[i][0]) + [torch.tensor(0).float()]*(MAX_LEN - len(train_inout_seq[i][0])))

            train_inout_seq[i][summary_index] = torch.tensor(summary_tokens(train_inout_seq[i][summary_index],vocab.stoi,MAX_LEN)).type(torch.FloatTensor)
            if learn_templates:
                #print(train_inout_seq[i][temp_index],template_vocab_stoi)
                train_inout_seq[i][temp_index] = torch.tensor(summary_tokens(train_inout_seq[i][temp_index],template_vocab_stoi,MAX_LEN)).type(torch.FloatTensor)
            else:
                train_inout_seq[i][temp_index] = train_inout_seq[i][summary_index]
            train_inout_seq[i] = tuple(train_inout_seq[i])

        for i in range(len(valid_inout_seq)):
            valid_inout_seq[i] = list(valid_inout_seq[i])
            valid_inout_seq[i][0] = torch.tensor(list(valid_inout_seq[i][0]) + [torch.tensor(0).float()]*(MAX_LEN - len(valid_inout_seq[i][0])))

            valid_inout_seq[i][summary_index] = torch.tensor(summary_tokens(valid_inout_seq[i][summary_index],vocab.stoi,MAX_LEN)).type(torch.FloatTensor)
            if learn_templates:
                valid_inout_seq[i][temp_index] = torch.tensor(summary_tokens(valid_inout_seq[i][temp_index],template_vocab_stoi,MAX_LEN)).type(torch.FloatTensor)
            else:
                valid_inout_seq[i][temp_index] = valid_inout_seq[i][summary_index]
            valid_inout_seq[i] = tuple(valid_inout_seq[i])

        for i in range(len(test_inout_seq)):
            test_inout_seq[i] = list(test_inout_seq[i])
            test_inout_seq[i][0] = torch.tensor(list(test_inout_seq[i][0]) + [torch.tensor(0).float()]*(MAX_LEN - len(test_inout_seq[i][0])))

            test_inout_seq[i][summary_index] = torch.tensor(summary_tokens(test_inout_seq[i][summary_index],vocab.stoi,MAX_LEN)).type(torch.FloatTensor)
            if learn_templates:
                test_inout_seq[i][temp_index] = torch.tensor(summary_tokens(test_inout_seq[i][temp_index],template_vocab_stoi,MAX_LEN)).type(torch.FloatTensor)
            else:
                test_inout_seq[i][temp_index] = test_inout_seq[i][summary_index]
            test_inout_seq[i] = tuple(test_inout_seq[i])

        # Remove remainder sequences
        train_remainder = (len(train_inout_seq) % BATCH_SIZE)*-1
        if train_remainder != 0:
            train_inout_seq = train_inout_seq[:train_remainder]

        valid_remainder = (len(valid_inout_seq) % BATCH_SIZE)*-1
        if valid_remainder != 0:
            valid_inout_seq = valid_inout_seq[:valid_remainder]

        test_remainder = (len(test_inout_seq) % BATCH_SIZE)*-1
        if test_remainder != 0:
            test_inout_seq = test_inout_seq[:test_remainder]

        # DataLoaders
        dataloader_train = DataLoader(train_inout_seq,
                                      batch_size=BATCH_SIZE,
                                      shuffle=False,
                                      num_workers=NUM_WORKERS,
                                      pin_memory=False)

        dataloader_valid = DataLoader(valid_inout_seq,
                                      batch_size=BATCH_SIZE,
                                      shuffle=False,
                                      num_workers=NUM_WORKERS,
                                      pin_memory=False)

        dataloader_test = DataLoader(test_inout_seq,
                                      batch_size=BATCH_SIZE,
                                      shuffle=False,
                                      num_workers=NUM_WORKERS,
                                      pin_memory=False)

        # Load Transformer
        net = Transformer(d_input, d_model, ntoken, ntemptoken, q, v, h, N, attention_size=attention_size, dropout=dropout, chunk_mode=chunk_mode, pe=pe, use_decoder=config['use_decoder']).to(device)

        if continue_train:
            model_path = 'tst/model/tst_' + summary_type + '.model'
            try:
                net.load_state_dict(torch.load(model_path,map_location=torch.device('cpu')))
            except RuntimeError:
                net.load_state_dict(torch.jit.load(model_path,map_location=torch.device('cpu')))

        optimizer = optim.Adam(net.parameters(), lr=LR)
        #optimizer = optim.SGD(net.parameters(), lr=LR)

        if schedule_lr:
            from torch.optim.lr_scheduler import StepLR, ReduceLROnPlateau
            if steplr:
                scheduler = StepLR(optimizer, step_size=2, gamma=0.1)
            elif reducelr:
                scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=0, verbose=True)

        #loss_function = OZELoss(alpha=0.3)
        loss_function = torch.nn.CrossEntropyLoss(ignore_index=PAD_IDX)
        #loss_function = torch.nn.NLLLoss(reduction='mean',
                                     #ignore_index=PAD_IDX)

        # Train model
        #model_save_path = f'transformer_models/model_{datetime.datetime.now().strftime("%Y_%m_%d__%H%M%S")}.pth'
        model_save_path = f'tst/model/tst_' + summary_type + '.model'
        val_loss_best = np.inf

        # Prepare loss history
        hist_loss = np.zeros(EPOCHS)
        hist_loss_val = np.zeros(EPOCHS)
        best_epoch = 0
        prev_valid_bleu = 0
        early_stop_counter = 0
        early_stop_counter_ = 0

        epoch_path = 'tst/epoch/tst_' + summary_type + '.txt'

        # Most recent epoch
        current_epoch = 0
        if continue_train:
            epoch_path = 'tst/epoch/tst_' + summary_type + '.txt'
            print("Loading epoch from",epoch_path)
            current_epoch = load_epoch(epoch_path)
            print(current_epoch)

        if current_epoch == EPOCHS+1:
            input("ERROR: This model is already done training")

        for idx_epoch in range(current_epoch,EPOCHS):

            running_loss = 0
            with tqdm(total=len(dataloader_train.dataset), desc=f"[Epoch {idx_epoch+1:3d}/{EPOCHS}]") as pbar:
                for idx_batch, (x,y,z) in enumerate(dataloader_train):
                    if len(x) < BATCH_SIZE:
                        continue

                    x = x.view(BATCH_SIZE,-1,d_input)
                    if learn_templates:
                        z = z.view(-1).long()

                    optimizer.zero_grad()

                    # Propagate input
                    if shift_right:
                        netout, netout_temp = net(x.to(device),device,copy.deepcopy(y).to(device))
                    else:
                        netout, netout_temp = net(x.to(device),device)
                    y = y.view(-1).long()
                    #input([netout.shape,y.shape,ntoken])
                    loss = loss_function(netout,y.to(device))
                    #input([netout.shape,netout_temp.shape,y.shape,x.shape,z.shape,ntemptoken,ntoken])
                    #input(template_vocab_stoi)
                    if learn_templates:
                        loss += loss_function(netout_temp,z.to(device))
                        netout_ = netout_temp.view(BATCH_SIZE,-1,ntemptoken)
                        y_ = y.view(BATCH_SIZE,-1)
                        #print(vocab.stoi,vocab.itos)
                        loss += add_temp_loss(netout_,y_,BATCH_SIZE,MAX_SUMM_LEN,MAX_LEN,vocab,template_vocab_stoi,special_chars,ntoken,device,loss_function,squish_tokens=squish_tokens)

                    # Backpropagate loss
                    loss.backward()

                    # Update weights
                    optimizer.step()

                    running_loss += loss.item()
                    pbar.set_postfix({'loss': running_loss/(idx_batch+1)})
                    pbar.update(x.shape[0])

                train_loss = running_loss/len(dataloader_train)
                val_loss, val_accuracy, val_bleu = compute_loss(net, dataloader_valid, loss_function, ntoken, MAX_SUMM_LEN, vocab, template_vocab_stoi, special_chars, device, BATCH_SIZE, d_input, shift_right=shift_right)
                val_loss = val_loss.item()
                pbar.set_postfix({'loss': train_loss, 'val_loss': val_loss})

                hist_loss[idx_epoch] = train_loss
                hist_loss_val[idx_epoch] = val_loss

                if val_loss < val_loss_best:
                    val_loss_best = val_loss
                    best_epoch = idx_epoch
                    save_epoch(str(epoch_path), idx_epoch+1)
                    save_vocab(vocab,vocab_path)
                    print("Saved epoch",idx_epoch+1,"at",str(epoch_path))
                    torch.save(net.state_dict(), model_save_path)

                early_stop_counter = early_stop_counter + 1 \
                        if prev_valid_bleu > val_bleu else 0

                early_stop_counter_= early_stop_counter_ + 1 \
                        if (prev_valid_bleu == val_bleu and val_bleu == 1) else 0
                #input([prev_valid_bleu,val_bleu,val_accuracy,val_loss])

                #if early_stop_counter_ == config['short_patience']:
                    #print('EARLY STOPPING (short)',early_stop_counter_,config['short_patience'])
                    #break
                if early_stop_counter == config['long_patience']:
                    print('EARLY STOPPING (long)',early_stop_counter,config['long_patience'])
                    break
                prev_valid_bleu = val_bleu

            if schedule_lr:
                if steplr:
                    scheduler.step()
                elif reducelr:
                    scheduler.step(val_accuracy)

        if not one_epoch:
            plt.plot(hist_loss, 'o-', label='train')
            plt.plot(hist_loss_val, 'o-', label='val')
            plt.legend()
            plt.show()
        print(f"model exported to {model_save_path} with loss {val_loss_best:5f}")

        # Validation
        _ = net.eval()

        # Testing
        #predictions = np.empty(shape=(len(dataloader_test.dataset), 168, 8))

        num_correct = 0
        num_correct_ = 0
        num_total = 0
        running_loss = 0
        model_sent = []
        y_sent = []
        model_sent_ = []
        y_sent_ = []
        with torch.no_grad():
            for x, y, z in tqdm(dataloader_test, total=len(dataloader_test)):
                x = x.view(BATCH_SIZE,-1,d_input)

                if learn_templates:
                    z = z.view(-1).long()
                if shift_right:
                    netout, netout_temp = net(x.to(device),device,copy.deepcopy(y).to(device))
                else:
                    netout, netout_temp = net(x.to(device),device)
                y = y.view(-1).long()

                loss = loss_function(netout, y.to(device))
                running_loss += loss.item()
                if learn_templates:
                    loss = loss_function(netout_temp, z.to(device))
                    running_loss += loss.item()

                    netout_ = netout.view(BATCH_SIZE,-1,ntoken)
                    y_ = y.view(BATCH_SIZE,-1)

                    running_loss += add_temp_loss(netout_,y_,BATCH_SIZE,MAX_SUMM_LEN,MAX_LEN,vocab,template_vocab_stoi,special_chars,ntoken,device,loss_function,squish_tokens=squish_tokens).item()


                num_correct, num_correct_, num_total, model_sent, y_sent, model_sent_, y_sent_ = compute_accuracy(netout,y,model_sent,y_sent,model_sent_,y_sent_,num_correct,num_correct_,num_total,BATCH_SIZE,MAX_LEN,ntoken,vocab,special_chars,len(dataloader_test),print_preds=True)

        accuracy = float(num_correct)/num_total
        print("Accuracy:", accuracy)

        accuracy_ = float(num_correct_)/num_total
        print("Accuracy (no special):", accuracy_)

        test_loss = running_loss/len(dataloader_test)
        print("Test Loss:", test_loss)

        print("Epoch used:", best_epoch)

        test_bleu = calc_bleu(y_sent, model_sent)
        print("Test BLEU:", test_bleu)

        test_bleu_ = calc_bleu(y_sent_, model_sent_)
        print("Test BLEU (no special):", test_bleu_)

        log_run(config,accuracy,test_loss,test_bleu,best_epoch,dest_log,accuracy_,test_bleu_)

        if config['test_analysis']:
            test_analysis(model_sent_,y_sent_)

if __name__ == "__main__":
    main()