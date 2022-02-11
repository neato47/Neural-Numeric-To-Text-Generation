legacy = True

import torch
from torch import Tensor, nn
try:
    from torchtext.data import Batch
except ImportError:
    from torchtext.legacy.data import Batch
import numpy as np
import pandas as pd
import seaborn as sns
from proto_lib import *
from saxpy.paa import paa
from saxpy.hotsax import find_discords_hotsax
import string
from network import *
from collections import Counter
from reporter.util.constant import Phase, SpecialToken
from reporter.util.logging import create_logger
from reporter.postprocessing.bleu import calc_bleu
from reporter.postprocessing.export import export_results_to_csv
from reporter.core.train import run
from pathlib import Path
import torchtext
try:
    from torchtext.data import get_tokenizer
except ImportError:
    from torchtext.legacy.data import get_tokenizer
import matplotlib.pyplot as plt
#from torch.nn import Transformer
from tst.transformer import Transformer

config = None

torch.backends.cudnn.benchmark = True
torch.backends.cudnn.enabled = True

"""Constants"""
#config = {
    #'enc_hidden_size' : 256,
    #'base_ric_embed_size' : 256,
    #'base_ric_hidden_size' : 256,
    #'ric_embed_size' : 32,
    #'enc_n_layers' : 3,
    #'word_embed_size' : 128,
    #'time_embed_size' : 64,
    #'dec_hidden_size' : 180,
    #'attn_type' : '',
    #'ric_hidden_size' : 64,
    #'use_dropout' : True,
    #'learning_rate' : 1e-2,
    #'n_epochs' : 60,
    #'batch_size' : 7,
    #'short_patience' : 5,
    #'long_patience' : 20,
    #'max_series_size' : 360,
    #'seq_length' : 7,
    #'enc_output_size' : 64,
    #'use_preprocessing' : True,
    #'use_series' : True
#}

#config = {
    #'enc_hidden_size' : 180,
    #'base_ric_embed_size' : 1024,
    #'base_ric_hidden_size' : 1024,
    #'ric_embed_size' : 1024,
    #'enc_n_layers' : 1,
    #'word_embed_size' : 64,
    #'time_embed_size' : 180,
    #'dec_hidden_size' : 180,
    #'attn_type' : '',
    #'ric_hidden_size' : 1024,
    #'use_dropout' : True,
    #'learning_rate' : 1e-4,
    #'n_epochs' : 78,
    #'batch_size' : 180,
    #'short_patience' : 5,
    #'long_patience' : 20,
    #'max_series_size' : 187,
    #'seq_length' : 7,
    #'enc_output_size' : 1024,
    #'use_attn' : True,
    #'use_preprocessing' : False,
    #'use_series' : True,
    #'sax_bool' : False,
    #'sax_bool2' : False,
    #'learn_templates' : True,
    #'cross_entropy' : False,
    #'truth_loss' : False,
    #'concat_sax' : False,
    #'usr_divide' : True,
    #'usr_shuffle' : True,
    #'summarizer_loss' : True,
    #'quantifier_loss' : True,
    #'summarizer_training_loss' : True,
    #'weekday_loss' : True,
    #'tw_loss' : True,
    #'epoch_select': 'bleu',
    #'summary_types' : ["GC"],
    #'time_windows' : [7],
    #'test_percentage': 0.15,
    #'test_analysis' : False,
    #'squish_tokens' : False,
    #'ge_norm' : False,
    #'schedule_lr' : False,
    #'step_lr' : False,
    #'reduce_lr' : False,
    #'remove_unk' : False,
#}

type_dict = dict()
summarizers = ["very","low","moderate","high","lower","about","the","same","higher","not","do","as","well","better","worse","stayed","rose","dropped"]
squished_summarizers = ["verylow","veryhigh","aboutthesame","notdoaswell","didnotreach","stayedthesame"]
quantifiers = ["all","most","more","than","half","some","almost","none","of","the"]

weekday_dict = { 1 : "Monday",
                 2 : "Tuesday",
                 3 : "Wednesday",
                 4 : "Thursday",
                 5 : "Friday",
                 6 : "Saturday",
                 7 : "Sunday" }

seq_dict = {
    "IT" : "IT_sequences_concat_large.csv",
    "WIT" : "WIT_sequences_concat.csv",
    "CB_description" : "CB_sequences_description_split_.csv",
    "CB_cluster" : "CB_sequences_cluster_split_.csv",
    "CB" : "CB_sequences_concat.csv",
    "SP" : "SP_sequences_cluster_input_.csv",
    "DB" : "DB_sequences_concat.csv",
}

def test_analysis(model_sent,y_sent):
    print()
    diff_list = []
    diff_dict = dict()

    for i in range(len(model_sent)):
        if model_sent[i] != y_sent[i]:
            pair = [model_sent[i],y_sent[i]]
            if pair not in diff_list:
                diff_list.append(pair)
                diff_dict[len(diff_list)-1] = 1
            else:
                idx = diff_list.index(pair)
                diff_dict[idx] += 1

    for i in range(len(diff_list)):
        prediction = diff_list[i][0]
        actual = diff_list[i][1]
        percentage = float(diff_dict[i])/sum(list(diff_dict.values()))
        try:
            print('Prediction: ' + prediction + '\nActual: ' + actual + '\nPercentage: ' + str(percentage))
        except TypeError:
            print('Prediction: ' + ' '.join(prediction) + '\nActual: ' + ' '.join(actual) + '\nPercentage: ' + str(percentage))
        print()
    input()

def construct_string(sequence):
    result = ""
    for j in range(len(sequence)):
        try:
            result += str(sequence[j].item())
        except:
            result += str(sequence[j])
        result += "|"

    return result[:-1]

def save_vocab(vocab, path):
    with open(path, 'w') as f:
        for token, freq in vocab.freqs.items():
            f.write(f'{token}\t{freq}\n')
        for token, index in vocab.stoi.items():
            if token not in vocab.freqs.keys():
                index_ = str(-1)
                f.write(f'{token}\t{index_}\n')

def save_epoch(path, epoch):
    with open(path, 'w') as f:
        f.write(f'{epoch}')

def log_run(config, performance, test_loss, bleu_score, epoch_used, path, second_accuracy=None, second_bleu=None):
    with open(path, 'w', newline='') as csvfile:
        import csv, datetime

        csvwriter = csv.writer(csvfile)

        header = ["Key","Value"]
        csvwriter.writerow(header)

        empty = ["",""]
        csvwriter.writerow(empty)

        now = ["Date",datetime.datetime.now().strftime("%d/%m/%Y %H:%M:%S")]
        csvwriter.writerow(now)

        if second_accuracy:
            performance = '%s (%s)' % (str(performance),str(second_accuracy))
        perf_line = ["Performance",str(performance)]
        csvwriter.writerow(perf_line)

        loss_line = ["Test Loss",str(test_loss)]
        csvwriter.writerow(loss_line)

        if second_bleu:
            performance = '%s (%s)' % (str(performance),str(second_accuracy))
        bleu_line = ["BLEU Score",str(bleu_score)]
        csvwriter.writerow(bleu_line)

        epoch_line = ["Epoch Used",str(epoch_used)]
        csvwriter.writerow(epoch_line)

        csvwriter.writerow(empty)

        for key in config:
            csvwriter.writerow([key,config[key]])

def get_summary_counts(train_inout_seq,valid_inout_seq,test_inout_seq,summary_index=2):
    train_dict = dict()
    valid_dict = dict()
    test_dict = dict()

    for i in range(len(train_inout_seq)):
        summary = train_inout_seq[i][summary_index]
        if summary in train_dict.keys():
            train_dict[summary] += 1
        else:
            train_dict[summary] = 1

    for i in range(len(valid_inout_seq)):
        summary = valid_inout_seq[i][summary_index]
        if summary in valid_dict.keys():
            valid_dict[summary] += 1
        else:
            valid_dict[summary] = 1

    for i in range(len(test_inout_seq)):
        summary = test_inout_seq[i][summary_index]
        if summary in test_dict.keys():
            test_dict[summary] += 1
        else:
            test_dict[summary] = 1

    summary_dict = dict()
    for (key,val) in train_dict.items():
        summary_dict[key] = [float(val)/len(train_inout_seq)]

    for (key,val) in valid_dict.items():
        if key in summary_dict.keys():
            summary_dict[key].append(float(val)/len(valid_inout_seq))
        else:
            summary_dict[key] = [0,float(val)/len(valid_inout_seq)]

    for (key,val) in test_dict.items():
        if key in summary_dict.keys():
            summary_dict[key].append(float(val)/len(test_inout_seq))
        else:
            summary_dict[key] = [0,0,float(val)/len(test_inout_seq)]

    for key in summary_dict.keys():
        n = len(summary_dict[key])
        if n < 3:
            summary_dict[key] += [0]*(3-n)

    for key in summary_dict.keys():
        print(key,summary_dict[key])

    print()
    print(train_dict)
    print()
    print(valid_dict)
    print()
    print(test_dict)
    input()

def build_template(summary,single=False):
    global type_dict
    #word_types = ["N","Q","TW","A","S"]


    time_windows = ["days","weeks","months","years"]
    attributes = ["calorie","intake","calorieintake"]

    if summary[-1] == '.':
        summary = summary[:-1]

    summary = summary.split(" ")

    # Single word corner case
    if len(summary) == 1:
        if summary[0] == "<s>":
            return ["<s>"]
        elif summary[0] == "</s>":
            return ["</s>"]
        template = []
    else:
        template = ["<s>"]

    last = ''
    prev_summ = ''
    for word in summary:
        #if word not in type_dict:
            #type_dict[word] = []
        word = word.rstrip(",")
        word = word.rstrip(".")

        if word in summarizers or word in squished_summarizers:
            #if "S" not in type_dict[word]:
                #type_dict[word].append("S")

            if (word == "the" or word == "was" or word == "as") and last != "S":
                last = word
                template.append(word)
                continue

            if single and last == 'S':
                query = prev_summ + word
                found = False
                for summ in squished_summarizers:
                    if query in summ:
                        found = True
                        break

                if found:
                    continue

                template.append(word)
                last = word
                continue

            last = 'S'
            template.append("S")
            prev_summ = word
        elif word in quantifiers or "ofthe" in word:
            if len(template) > 0 and (word == "the" or word == "of" or word == "than") and template[-1] == "Q":
                #if "Q" not in type_dict[word]:
                    #type_dict[word].append("Q")
                if single and last == 'Q':
                    continue
                last = 'Q'
                template.append("Q")
            elif word != "the" and word != "of" and word != "than":
                #if "Q" not in type_dict[word]:
                    #type_dict[word].append("Q")
                if single and last == 'Q':
                    continue
                last = 'Q'
                template.append("Q")
            else:
                last = word
                template.append(word)
        elif word in time_windows or word+"s" in time_windows or word.isnumeric():
            #if "TW" not in type_dict[word]:
                #type_dict[word].append("TW")
            if single and last == 'TW':
                continue
            last = 'TW'
            template.append("TW")
        elif word in attributes:
            #if "A" not in type_dict[word]:
                #type_dict[word].append("A")
            if single and last == 'A':
                continue
            last = 'A'
            template.append("A")
        elif word in weekday_dict.values():
            #if "D" not in type_dict[word]:
            #type_dict[word].append("D")

            if single and last == 'D':
                continue
            last = 'D'
            template.append("D")
        else:
            last = word
            template.append(word)

    template.append("</s>") #</s>

    return template


def combine_special_tokens(summary,reverse=False):

    replace_dict = { "all of the" : "allofthe",
                     "most of the" : "mostofthe",
                     "more than half of the" : "morethanhalfofthe",
                     "half of the" : "halfofthe",
                     "some of the" : "someofthe",
                     "almost none of the" : "almostnoneofthe",
                     "none of the" : "noneofthe",
                     "very high" : "veryhigh",
                     "very low" : "verylow",
                     "abnormally low" : "abnormallylow",
                     "abnormally high" : "abnormallyhigh",
                     "extremely low" : "extremelylow",
                     "extremely high" : "extremelyhigh",
                     "did not reach" : "didnotreach",
                     "calorie intake" : "calorieintake",
                     "about the same" : "aboutthesame",
                     "not do as well" : "notdoaswell",
                     "stayed the same" : "stayedthesame"}

    # Order matters for keys to iterate through this
    replace_list = ["all of the","most of the","more than half of the",
                     "half of the","some of the","almost none of the",
                     "none of the","very high","very low","abnormally low",
                     "abnormally high","extremely low","extremely high",
                     "did not reach","calorie intake","about the same",
                     "not do as well","stayed the same"]

    for elem in replace_list:
        if reverse:
            elem = replace_dict[elem]

        if type(summary) is torch.Tensor:
            summary = summary.item()
        if elem in summary or elem.capitalize() in summary:
            if reverse:
                for key in replace_dict.keys():
                    if replace_dict[key] == elem:
                        replacement = key
            elif elem.capitalize() in summary:
                replacement = replace_dict[elem].capitalize()
                elem = elem.capitalize()
            else:
                replacement = replace_dict[elem]
            summary = summary.replace(elem,replacement)

    return summary

def getTW(tw):
    # Choose string for time window based on tw
    if tw == 365:
        TW = "years"
    elif tw == 30:
        TW = "months"
    elif tw == 7:
        TW = "weeks"
    elif tw == 1:
        TW = "days"
    elif tw == 0.04:
        TW = "hours"
    else:
        TW = None

    return TW

#def generateGE(key_list,sax,letter_map_list,alpha_sizes,age,activity_level,TW,alpha):
    #guideline_summarizers = ["reached","did not reach"]

    #summarizers_list = []
    #for i in range(len(key_list)):
        #summarizers_list.append(guideline_summarizers)

    #summarizer_type = "Goal Evaluation - "
    #for i in range(len(key_list)):
        #summarizer_type += key_list[i]
        #if i != len(key_list)-1:
            #summarizer_type += " and "

    #past_tw_list = [sax]
    #input_goals = [] # TODO: fix this
    #_, t1_list, quantifier_list, summary_list, _ = generate_summaries(summarizers_list,
                                                                      #summarizer_type,
                                                                      #key_list,
                                                                      #past_tw_list,
                                                                      #letter_map_list,
                                                                      #alpha_sizes,
                                                                      #alpha,
                                                                      #age=age,
                                                                      #activity_level=activity_level,
                                                                      #TW=TW,
                                                                      #goals=input_goals)

    #input(summary_list)
    #index = best_quantifier_index(quantifier_list,t1_list)
    #return [[summary_list[index]],[t1_list[index]]]

#def generateEC(key_list,sax,alphabet_list,letter_map_list,alpha_sizes,tw):
    #pass

def generateIT(key_list,sax,alphabet_list,letter_map_list,alpha_sizes,tw,weekday=False,provenance=False):
    summarizer_type = "If-then pattern "
    for i in range(len(key_list)):
        if key_list[i] == "step count":
            key_list[i] = "Step Count"

        summarizer_type += key_list[i]
        if i != len(key_list)-1:
            summarizer_type += " and "
    sax_list = [sax]
    db_fn_prefix = "series_db"
    path = "C:/Users/harrij15/Documents/GitHub/RPI-HEALS//" # Path for pattern data
    cygwin_path = r"C:\Apps\cygwin64\bin" # Path to Cygwin

    min_conf = 0.8
    min_sup = 0.2
    if weekday:
        min_conf = 0
        min_sup = 0

    proto_cnt = 0

    weekday_dict = { 1 : "Monday",
                     2 : "Tuesday",
                     3 : "Wednesday",
                     4 : "Thursday",
                     5 : "Friday",
                     6 : "Saturday",
                     7 : "Sunday" }

    weekdays = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    date_column = None
    if weekday:
        date_column = []
        index = 0
        for letter in sax:
            date_column.append(weekdays[index])
            index += 1
            if index == 7:
                index = 0

    summary_list, supports, proto_cnt, numsum_list, summarizers_list = analyze_patterns(key_list,
                                                                                        sax_list,
                                                                                        alphabet_list,
                                                                                        letter_map_list,
                                                                                        weekday_dict,
                                                                                        tw,
                                                                                        alpha_sizes,
                                                                                        db_fn_prefix,
                                                                                        path,
                                                                                        cygwin_path,
                                                                                        min_conf,
                                                                                        min_sup,
                                                                                        proto_cnt,
                                                                                        weekdays=date_column)
    if provenance:
        truths = []
        for i in range(len(supports)):
            _, t1 = getQForS(supports[i],None,None)
            truths.append(t1)
        return summary_list, summarizers_list, truths
    return summary_list

def split_datasets(all_data,config):
    # Split data into train, valid, and test
    test_data_size = int(config["test_percentage"]*len(all_data))
    train_data = all_data[:-test_data_size*2] # 0.7
    valid_data = all_data[-test_data_size*2:-test_data_size] # 0.15
    test_data = all_data[-test_data_size:] # 0.15

    return train_data, valid_data, test_data, test_data_size

def load_epoch(epoch_path):
    with open(epoch_path,'r') as f:
        lines = f.readlines()
        for line in lines:
            line = line.strip('\n')
            return int(line)

def load_vocab(vocab_path):
    freq_tuples = dict()
    with open(vocab_path,'r') as f:
        lines = f.readlines()
        for line in lines:
            line = line.strip('\n').split('\t')
            word = line[0]
            count = int(line[1])
            freq_tuples[word] = count

    counter = Counter(freq_tuples)
    vocab = Vocab(counter,min_freq=1,specials=('<unk/>', '<pad/>', '<s>', '</s>'))

    return vocab

def load_seq_data(filename,config,transformer=False,quick=False,squish_tokens=False,include_tw_sax=False):

    # Retrieve user data
    attr = 'Calories'
    data_path = 'combined_data_' + attr + '.csv'
    df = pd.read_csv(data_path)
    all_data = list(df[attr])
    dates = list(df["date"])
    user_data = get_user_data(all_data,config)

    # Load sequences
    data_path = filename
    inout_seq = []
    import csv
    max_series_size = max_seq_size = 0
    with open(data_path, newline='') as csvfile:
        csvreader = csv.reader(csvfile)
        header = True
        seq_lengths = set([])
        rows = []

        row_limit = float("Inf")
        if "DB" in config["summary_types"]:
            row_limit = 64601
            if not transformer:
                #config["n_epochs"] = 500
                #config["n_epochs"] = 78
                #row_limit = 1000
                #row_limit = -1
                config["batch_size"] = 128
                #config["enc_output_size"] = 256
                #config["learn_templates"] = False
                #config["use_attn"] = False
                #config["use_series"] = False
        #elif "SETW" in config["summary_types"]:
            #config["n_epochs"] = 500
            #row_limit = 1000
            #row_limit = -1
            #config["batch_size"] = 1
            #config["learn_templates"] = False
            #config["use_attn"] = False
            #config["use_series"] = False
        elif "IT" in config["summary_types"] or "WIT" in config["summary_types"]:
            #row_limit = 4500
            #if "WIT" in config["summary_types"]:
                #row_limit = 1500

            if transformer:
                row_limit = -1
            else:
                #config["n_epochs"] = 150
                config["batch_size"] = 128
        #elif "CB_description" in config["summary_types"] or "CB_cluster" in config["summary_types"]:
            #config["n_epochs"] = 250

        for row in csvreader:
            if header:
                header = False
                continue

            seq = row[0].split("|")
            if len(seq) < 100:
                if len(seq) not in seq_lengths:
                    seq_lengths.add(len(seq))
                rows.append(row)
                #input(row)

            if len(rows) == row_limit:
                break

        config['time_windows'] = list(seq_lengths)
        config['seq_length'] = sum(config['time_windows'])
        header = True

        #if "DB" in config["summary_types"]: input(len(rows))

        for row in rows:
            if header:
                header = False
                #seq_length = len([x for x in row if "Day" in x])
                continue

            seq_norm = row[0].split("|")

            #seq_len = len(seq_norm)
            try:
                seq_norm = [float(x) for x in seq_norm]
            except ValueError:
                pass
            #seq_norm += [0.0] * (seq_length - seq_len)
            try:
                seq_norm = torch.FloatTensor(seq_norm)
            except ValueError:
                for i in range(len(seq_norm)):
                    seq_norm[i] = float(seq_norm[i][7:-1])
                seq_norm = torch.FloatTensor(seq_norm)

            #seq_norm = torch.FloatTensor([float(x) for x in row[:seq_length-1]])
            idx = 1
            try:
                series_norm = row[idx].split("|")
                try:
                    series_norm = [float(x) for x in series_norm]
                except ValueError:
                    pass

                try:
                    series_norm = torch.FloatTensor(series_norm)
                except ValueError:
                    for i in range(len(series_norm)):
                        series_norm[i] = float(series_norm[i][7:-1])
                    series_norm = torch.FloatTensor(series_norm)
            except ValueError:
                idx = 0

            #print(len(seq_norm))
            #input(len(series_norm))

            summary = row[idx+1]
            truth = float(row[idx+2])
            sax = row[idx+3]
            sid = int(row[idx+4])
            uid = int(row[idx+5])

            if squish_tokens:
                summary = combine_special_tokens(summary)

            template = build_template(summary)

            if quick:
                if transformer:
                    inout_seq.append((torch.tensor(list(seq_norm) + list(series_norm)), summary, template))
                else:
                    inout_seq.append((seq_norm, summary))
            else:
                if transformer:
                    inout_seq.append((torch.tensor(list(seq_norm) + list(series_norm)), summary, truth, sax, sid, template))
                else:
                    seq_ = (seq_norm, series_norm, summary, truth, sax, sid, template)
                    if include_tw_sax:
                        seq_ = (seq_norm, series_norm, summary, truth, sax, sid, template, row[-1])
                    inout_seq.append(seq_)

            max_seq_size = max(len(seq_norm),max_seq_size)
            max_series_size = max(len(series_norm),max_series_size)
            #max_series_size = max(len(user_data[uid]),max_series_size)

    # Shuffle examples
    import random
    random.shuffle(inout_seq)

    train, valid, test, test_data_size = split_datasets(inout_seq,config)
    return train, valid, test, max_seq_size, max_series_size, test_data_size

def get_user_data(data,config):
    user_data = []
    sublist = []
    for i in range(len(data)):
        if not type(data[i]) is int and (data[i] == " " or " " in data[i]):
            if config['use_preprocessing']:
                user_data.append(preprocess(sublist))
            else:
                user_data.append(sublist)
            sublist = []
        else:
            try:
                sublist.append(int(data[i]))
            except:
                sublist.append(data[i])
    return user_data

#def get_data(usr_divide,concat_sax,full_data,config,usr_shuffle=False,return_dataset=False,include_dates=False):
    #data_path = 'combined_data.csv'

    ###additional_data_path = 'data/MyFitnessPal/FoodLogs/mfpfoodlog3277_fixed.csv'
    #if concat_sax:
        #df = pd.read_csv(data_path)
        #all_data = list(df["Calories"])
        #sax = list(df["SAX"])
        #dates = None
    #else:
        #df = pd.read_csv(data_path)
        #all_data = list(df["Calories"])
        #dates = list(df["date"])

    #if usr_divide:
        #user_data = get_user_data(all_data,config)
        #dates = get_user_data(dates,config)
        #user_data = [[dates[i],user_data[i]] for i in range(len(user_data))]

    #if usr_shuffle:
        #import random
        #random.shuffle(user_data)

    #if usr_divide:
        ## Train model on a subset of the data
        #if not full_data:
            #user_data = user_data[:7]
        #dates = [x[0] for x in user_data]
        #all_data = [x[1] for x in user_data]
    #else:
        #all_data = all_data[:1000]
        #import random
        #random.shuffle(all_data)

    ## Split data into train, valid, and test
    #train_data, valid_data, test_data, test_data_size = split_datasets(all_data,config)
    #if include_dates:
        #train_dates, valid_dates, test_dates, test_data_size = split_datasets(dates,config)
    #else:
        #train_dates = None
        #valid_dates = None
        #test_dates = None

    #if not concat_sax and not usr_divide:
        #sax = ts_to_string(znorm(np.array(all_data)), cuts_for_asize(5))
    #elif usr_divide:
        #sax = []
        #sax_dates = []
        #max_size = 0
        #for sublist in all_data:
            #if len(sublist) > max_size:
                #max_size = len(sublist)
            #sax.append(ts_to_string(znorm(np.array(sublist)), cuts_for_asize(5)))

    ## Split SAX into train, valid, and test
    #train_sax, valid_sax, test_sax, test_data_size = split_datasets(sax)

    #if usr_divide:
        #train_norm = [torch.FloatTensor(znorm(np.array(x))).view(-1) for x in train_data]
        #valid_norm = [torch.FloatTensor(znorm(np.array(x))).view(-1) for x in valid_data]
        #test_norm = [torch.FloatTensor(znorm(np.array(x))).view(-1) for x in test_data]
    #else:
        #train_norm = [torch.FloatTensor(znorm(np.array(train_data))).view(-1)]
        #valid_norm = [torch.FloatTensor(znorm(np.array(valid_data))).view(-1)]
        #test_norm = [torch.FloatTensor(znorm(np.array(test_data))).view(-1)]

    #if return_dataset:
        #return all_data, train_dates, valid_dates, test_dates, train_data, train_norm, train_sax, valid_data, valid_norm, valid_sax, test_data, test_norm, test_sax, max_size, test_data_size
    #return train_dates, valid_dates, test_dates, train_data, train_norm, train_sax, valid_data, valid_norm, valid_sax, test_data, test_norm, test_sax, max_size, test_data_size

def preprocess(data_list):
    # Standardization
    data = np.array(data_list)
    mean = np.mean(data)
    std = np.std(data)

    output = []
    for x in data_list:
        x_std = (x - mean) / std
        output.append(x_std)

    return output

def main():

    import sys

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load config
    global config
    with open('ED_config.json','r') as jsonfile:
        config = json.load(jsonfile)

    # Flags
    full_data = (str(device) != "cpu")
    sax_bool = config['sax_bool']
    sax_bool2 = config['sax_bool2']
    learn_templates = config['learn_templates']
    cross_entropy = config['cross_entropy']
    truth_loss = config['truth_loss']
    concat_sax = config['concat_sax']
    usr_divide = config['usr_divide']
    usr_shuffle = config['usr_shuffle']
    summary_types = config['summary_types']
    time_windows = config['time_windows']
    one_epoch = (str(device) == "cpu")
    squish_tokens = config['squish_tokens']
    ge_norm = config['ge_norm']
    schedule_lr = config['schedule_lr']
    steplr = config['schedule_lr'] and config['step_lr']
    reducelr = config['schedule_lr'] and config['reduce_lr']
    remove_unk = config['remove_unk']
    continue_train = config['continue_train']
    include_dates = ("DB" in summary_types or "GA" in summary_types)
    include_tw_sax = False

    use_transformer = config['use_transformer']
    if use_transformer:
        with open('transformer_config.json','r') as jsonfile:
            transformer_config = json.load(jsonfile)

    config['seq_length'] = max(time_windows)

    summary_type = config["summary_types"][0]
    if summary_type in seq_dict.keys():
        data_filename = seq_dict[summary_type]
    else:
        data_filename = summary_type + "_" + str(device) + ".csv"

    train_inout_seq, valid_inout_seq, test_inout_seq, max_seq_size, max_series_size, test_data_size = load_seq_data(data_filename,config,squish_tokens=squish_tokens,include_tw_sax=include_tw_sax)
    config['max_series_size'] = max_series_size
    #input([config['max_series_size'],config['seq_length'],config['time_windows']])

    train_valid_inout_seq = train_inout_seq + valid_inout_seq

    try:
        from torchtext.data import RawField, Field, Example, Dataset, Iterator, BucketIterator
    except ImportError:
        from torchtext.legacy.data import RawField, Field, Example, Dataset, Iterator, BucketIterator

    # Create Fields
    passenger_field = RawField()
    series_field = RawField()
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
    template_field = RawField()
    template_field.is_target = False

    fields = [('passengers',passenger_field),('series',series_field),('token',token_field),('truth',truth_field),('sax',sax_field),('summary_id',id_field),('template',template_field)]

    # Create Examples
    from torch.utils.data import random_split
    shuffle_seqs = []
    train_examples = []
    valid_examples = []
    test_examples = []
    train_valid_examples = []
    seq_size = 12

    all_seqs = train_valid_inout_seq + test_inout_seq
    random.shuffle(all_seqs)

    for seq in all_seqs:
        shuffle_seqs.append([construct_string(seq[0]),construct_string(seq[1]),seq[2],seq[4],seq[-1]])
        if len(shuffle_seqs) == test_data_size:
            break

    #import csv
    #with open("first_batch_Carbohydrates.csv","w",newline="") as csvfile:
        #csvwriter = csv.writer(csvfile)
        #csvwriter.writerow(["Sequence","Series","Summary","SAX","TWSAX"])
        #for sublist in shuffle_seqs:
            #csvwriter.writerow(sublist)

    #input("Done")

    for seq in train_inout_seq:
        #input([seq[0].is_cuda,seq[1].is_cuda])
        train_examples.append(Example.fromlist(seq,fields))

    for seq in valid_inout_seq:
        valid_examples.append(Example.fromlist(seq,fields))

    for seq in train_valid_inout_seq:
        train_valid_examples.append(Example.fromlist(seq,fields))

    for seq in test_inout_seq:
        test_examples.append(Example.fromlist(seq,fields))

    input(len(train_valid_examples)+len(test_examples))

    # Create Datasets
    train = Dataset(train_examples,fields)
    valid = Dataset(valid_examples,fields)
    train_valid = Dataset(train_valid_examples,fields)
    test = Dataset(test_examples,fields)
    dataset_list = [train,valid,train_valid,test]
    #for x in dataset_list: print(x.get_device)
    #input()

    num_train_batches = int(len(train)/config['batch_size'])
    if num_train_batches==0:
        num_train_batches = 1
    num_valid_batches = int(len(valid)/config['batch_size'])
    num_test_batches = int(len(test)/config['batch_size'])
    if num_valid_batches==0:
        num_valid_batches = 1
    if num_test_batches==0:
        num_test_batches = 1

    token_field.build_vocab(train_valid,min_freq=1)

    train = BucketIterator(train,config['batch_size'],device=device)
    valid = BucketIterator(valid,config['batch_size'],train=False,sort=False,device=device)
    test = BucketIterator(test,config['batch_size'],train=False,sort=False,device=device)

    transformer_str = '_transformerlstm' if use_transformer else ''

    if continue_train:
        # Load vocabulary
        vocab_path = 'reporter/vocab/reporter_' + summary_type + '_fd_td' + transformer_str + '.txt'
        vocab = load_vocab(vocab_path)
        vocab_size = len(vocab)
    else:
        vocab = token_field.vocab

        if "ST" in summary_types:
            for x in quantifiers:
                vocab.freqs.update([x.capitalize()])
                vocab.freqs.subtract([x.capitalize()])
        else:
            for x in quantifiers:
                vocab.freqs.update([x])
                vocab.freqs.subtract([x])

        for x in summarizers:
            vocab.freqs.update([x])
            vocab.freqs.subtract([x])

        vocab_size = len(vocab)

    # Setup
    attn = None
    if config['use_attn']:
        attn = GeneralAttention(config['dec_hidden_size'], config['dec_hidden_size'])

    if use_transformer:
        d_input = 1 # From dataset
        d_output = config['enc_output_size']
        d_model = transformer_config['d_model']
        if d_model >= 64: # Leads to calculation error when concatenating
            d_model = 32
        q = transformer_config['query_size'] # Query size
        v = transformer_config['value_size'] # Value size
        h = transformer_config['num_heads'] # Number of heads
        N = transformer_config['n_stacks'] # Number of encoder and decoder to stack
        attention_size = transformer_config['attention_size'] # Attention window size
        pe = transformer_config['positional_encoding'] # Positional encoding
        chunk_mode = transformer_config['chunk_mode']
        dropout = transformer_config['dropout']
        value_encoder = Transformer(d_input, d_model, d_output, d_output, q, v, h, N, attention_size=attention_size, dropout=dropout, chunk_mode=chunk_mode, pe=pe, use_decoder=False,enc_hidden_size=config['enc_hidden_size'],max_series_size=config['max_series_size'],seq_length=max_seq_size).to(device)
    else:
        #config['seq_length'] = 4572
        value_encoder = Encoder(config, device)

    #t = torch.cuda.get_device_properties(0).total_memory
    #r = torch.cuda.memory_reserved(0)
    #a = torch.cuda.memory_allocated(0)
    #f = r-a  # free inside reserved
    #print([t,r,a,f])

    summary_decoder = Decoder(config, vocab_size, attn, device)
    encoder_list = [value_encoder]
    decoder_list = [summary_decoder]

    if learn_templates:
        template_decoder = Decoder(config, vocab_size, attn, device, "temp")
        decoder_list = [summary_decoder,template_decoder]

    model = EncoderDecoder(encoder_list, decoder_list, device, learn_templates, cross_entropy, truth_loss, transformer_encoder=use_transformer)

    if continue_train:
        model_path = 'reporter/model/reporter_' + summary_type + '_fd_td' + transformer_str + '.model'
        try:
            model.load_state_dict(torch.load(model_path,map_location=torch.device('cpu')))
        except RuntimeError:
            model.load_state_dict(torch.jit.load(model_path,map_location=torch.device('cpu')))

    #t = torch.cuda.get_device_properties(0).total_memory
    #r = torch.cuda.memory_reserved(0)
    #a = torch.cuda.memory_allocated(0)
    #f = r-a  # free inside reserved
    #input([t,r,a,f])

    optimizer = torch.optim.Adam(model.parameters(), lr=config['learning_rate'])
    #optimizer = torch.optim.RMSprop(model.parameters(), lr=config['learning_rate'], momentum=0.9)
    criterion = torch.nn.NLLLoss(reduction='mean',
                                 ignore_index=vocab.stoi[SpecialToken.Padding.value])

    if schedule_lr:
        from torch.optim.lr_scheduler import StepLR, ReduceLROnPlateau
        if steplr:
            scheduler = StepLR(optimizer, step_size=2, gamma=0.1)
        elif reducelr:
            scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=0, verbose=True)

    # Train model
    if sax_bool:
        dest_model = 'reporter/model/reporter_sax'
        dest_vocab = 'reporter/vocab/reporter_sax'
        dest_log = 'reporter/log/reporter_sax'
        dest_epoch = 'reporter/epoch/reporter_sax'
    elif sax_bool2:
        dest_model = 'reporter/model/reporter_sax2'
        dest_vocab = 'reporter/vocab/reporter_sax2'
        dest_log = 'reporter/log/reporter_sax2'
        dest_epoch = 'reporter/epoch/reporter_sax2'
    else:
        dest_model = 'reporter/model/reporter'
        dest_vocab = 'reporter/vocab/reporter'
        dest_log = 'reporter/log/reporter'
        dest_epoch = 'reporter/epoch/reporter'

    for summary_type in summary_types:
        dest_model += "_" + summary_type
        dest_vocab += "_" + summary_type
        dest_log += "_" + summary_type
        dest_epoch += "_" + summary_type

    if full_data:
        dest_model += "_fd"
        dest_vocab += "_fd"
        dest_log += "_fd"
        dest_epoch += "_fd"
    if learn_templates:
        dest_model += "_td"
        dest_vocab += "_td"
        dest_log += "_td"
        dest_epoch += "_td"
    if cross_entropy:
        dest_model += "_ce"
        dest_vocab += "_ce"
        dest_log += "_ce"
        dest_epoch += "_ce"
    elif truth_loss:
        dest_model += "_mse"
        dest_vocab += "_mse"
        dest_log += "_mse"
        dest_epoch += "_mse"
    if use_transformer:
        dest_model += "_transformerlstm"
        dest_vocab += "_transformerlstm"
        dest_log += "_transformerlstm"
        dest_epoch += "_transformerlstm"

    import os
    log_cnt = len(os.listdir("reporter/log"))+1
    dest_log += "_" + str(log_cnt)

    dest_model += ".model"
    dest_vocab += ".txt"
    dest_log += ".csv"
    dest_epoch += ".txt"

    dest_dir = Path('reporter/output')
    prev_valid_bleu = 0.0
    max_score = -1
    best_epoch = 0
    early_stop_counter = 0
    early_stop_counter_ = 0

    train_loss = []
    valid_loss = []
    training_matches = []
    validation_matches = []
    test_loss = []

    current_epoch = 0
    if continue_train:
        epoch_path = 'reporter/epoch/reporter_' + summary_type + '_fd_td' + transformer_str + '.txt'
        print("Loading epoch from",epoch_path)
        current_epoch = load_epoch(epoch_path)
        #input(current_epoch)

    if current_epoch == config['n_epochs']+1:
        input("ERROR: This model is already done training")

    for epoch in range(current_epoch,config['n_epochs']):
        progress = float(epoch)/config['n_epochs'] * 100
        print('start epoch {}/{} ({}%)'.format(epoch+1, config['n_epochs'], int(progress)))
        train_result = run(train,
                           num_train_batches,
                           vocab,
                           model,
                           optimizer,
                           criterion,
                           Phase.Train,
                           device,
                           learn_templates,
                           remove_unk)

        print("Training prediction:",(' ').join(train_result.pred_sents[0]))
        print("Training target:",(' ').join(train_result.gold_sents[0]))
        #input(train_result.gold_sents[0])
        #input([len(train_result.gold_sents), len(train_result.pred_sents)])
        train_bleu = calc_bleu(train_result.gold_sents, train_result.pred_sents)
        train_matches = len([train_result.pred_sents[i] for i in range(len(train_result.pred_sents)) if (train_result.pred_sents[i] == train_result.gold_sents[i]) or (train_result.pred_sents[i] == train_result.gold_sents[i][:-1] and train_result.gold_sents[i][-1] == SpecialToken.EOS.value)])
        train_matches = train_matches/float(len(train_result.pred_sents))

        #print("done train")
        valid_result = run(valid,
                           num_valid_batches,
                           vocab,
                           model,
                           optimizer,
                           criterion,
                           Phase.Valid,
                           device,
                           learn_templates,
                           remove_unk)

        print("Valid prediction:",(' ').join(valid_result.pred_sents[0]))
        print("Valid target:",(' ').join(valid_result.gold_sents[0]))
        valid_bleu = calc_bleu(valid_result.gold_sents, valid_result.pred_sents)
        valid_matches = len([valid_result.pred_sents[i] for i in range(len(valid_result.pred_sents)) if (valid_result.pred_sents[i] == valid_result.gold_sents[i]) or (valid_result.pred_sents[i] == valid_result.gold_sents[i][:-1] and valid_result.gold_sents[i][-1] == SpecialToken.EOS.value)])
        valid_matches = valid_matches/float(len(valid_result.pred_sents))

        if config['epoch_select'] == 'score':
            if valid_result.loss != 0:
                valid_score = ((valid_bleu*100) * (valid_matches*100))/valid_result.loss
            else:
                valid_score = ((valid_bleu*100) * (valid_matches*100))
        elif config['epoch_select'] == 'accuracy':
            valid_score = valid_matches
        else:
            valid_score = valid_bleu

        train_loss.append(train_result.loss)
        training_matches.append(train_matches)
        valid_loss.append(valid_result.loss)
        validation_matches.append(valid_matches)

        #avg_diff = 0
        #for i in range(len(valid_result.pred_sents)):
            ##print([valid_result.pred_sents[i],valid_result.gold_sents[i]])

            #diff = 0
            #for j in range(min(len(valid_result.pred_sents[i]),len(valid_result.gold_sents[i]))):
                #if valid_result.pred_sents[i][j] != valid_result.gold_sents[i][j]:
                    #diff += 1

            #avg_diff += diff + abs(len(valid_result.pred_sents[i])-len(valid_result.gold_sents[i]))

        #input([len(valid_result.pred_sents[0]),len(valid_result.gold_sents[0]),float(avg_diff)/len(valid_result.pred_sents)])

        s = ' | '.join(['epoch: {0:4d}'.format(epoch+1),
                        'training loss: {:.2f}'.format(train_result.loss),
                        'training BLEU: {:.4f}'.format(train_bleu),
                        'training accuracy: {:.2f}'.format(train_matches),
                        'validation loss: {:.2f}'.format(valid_result.loss),
                        'validation BLEU: {:.4f}'.format(valid_bleu),
                        'validation accuracy: {:.2f}'.format(valid_matches)
                        ])
        print(s)
        if max_score < valid_score:
            torch.save(model.state_dict(), str(dest_model))
            save_epoch(str(dest_epoch), epoch+1)
            save_vocab(vocab,dest_vocab)
            max_score = valid_score
            best_epoch = epoch

        early_stop_counter = early_stop_counter + 1 \
            if prev_valid_bleu > valid_bleu else 0

        if "DB" not in config["summary_types"]:
            early_stop_counter_ = early_stop_counter_ + 1 \
                if (prev_valid_bleu == valid_bleu and valid_bleu == 1) else 0

            if early_stop_counter_ == config['short_patience']:
                print('EARLY STOPPING')
                break
        if early_stop_counter == config['long_patience']:
            print('EARLY STOPPING')
            break
        prev_valid_bleu = valid_bleu

        if one_epoch:
            break

        if schedule_lr:
            if steplr:
                scheduler.step()
            elif reducelr:
                scheduler.step(valid_matches)

    #fig, axs = plt.subplots(1)
    #axs.plot(train_loss, label='train')
    #axs.plot(valid_loss, label='valid')
    #axs.legend()
    #plt.title("Loss")
    #plt.show()
    if not one_epoch:
        f1 = plt.figure()
        ax1 = f1.add_subplot(111)
        ax1.plot(train_loss, label='train')
        ax1.plot(valid_loss, label='valid')
        plt.title("Loss")
        plt.xlabel("Epochs")
        plt.ylabel("Loss")

        f2 = plt.figure()
        ax2 = f2.add_subplot(111)
        ax2.plot(training_matches, label='train')
        ax2.plot(validation_matches, label='valid')
        plt.title("Accuracy")
        plt.xlabel("Epochs")
        plt.ylabel("Accuracy")

    plt.show()

    # === Test ===
    with open(dest_model,mode='rb') as f:
        model.load_state_dict(torch.load(f))

    test_result = run(test,
                      num_test_batches,
                      vocab,
                      model,
                      optimizer,
                      criterion,
                      Phase.Test,
                      device,
                      learn_templates,
                      remove_unk)

    exact_matches = len([test_result.pred_sents[i] for i in range(len(test_result.pred_sents)) if (test_result.pred_sents[i] == test_result.gold_sents[i]) or (test_result.pred_sents[i] == test_result.gold_sents[i][:-1] and test_result.gold_sents[i][-1] == SpecialToken.EOS.value)])
    exact_matches = exact_matches/float(len(test_result.pred_sents))

    test_bleu = calc_bleu(test_result.gold_sents, test_result.pred_sents)
    test_loss.append(test_result.loss)

    s = ' | '.join(['epoch used: {:04d}'.format(best_epoch),
                    'Test Loss: {:.2f}'.format(test_result.loss),
                    'Test BLEU: {:.10f}'.format(test_bleu),
                    'Exact matches: {:.2f}'.format(exact_matches)])
    print(s)

    export_results_to_csv(dest_dir, test_result, sax_bool, sax_bool2, full_data, learn_templates, cross_entropy, truth_loss)
    log_run(config,exact_matches,test_result.loss,test_loss,best_epoch,dest_log)

    if config['test_analysis']:
        test_analysis(test_result.pred_sents,test_result.gold_sents)

if __name__ == "__main__":
    main()
