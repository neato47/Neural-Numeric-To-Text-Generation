from proto_lib import *
from saxpy.paa import paa
from saxpy.hotsax import find_discords_hotsax
import numpy as np
import csv
import torch
from torch import Tensor, nn
from ED import *

def get_data(time_windows,usr_divide,concat_sax,full_data,usr_shuffle=False,return_dataset=False,include_dates=False,all_seqs=False):
    attr = "Calories"
    data_path = 'combined_data_' + attr + '.csv'
    tw = time_windows[0]

    ##additional_data_path = 'data/MyFitnessPal/FoodLogs/mfpfoodlog3277_fixed.csv'
    if concat_sax:
        df = pd.read_csv(data_path)
        all_data = list(df[attr])
        sax = list(df["SAX"])
        dates = None
    else:
        df = pd.read_csv(data_path)
        all_data = list(df[attr])
        dates = list(df["date"])

    if usr_divide:
        user_data = get_user_data(all_data,config)
        dates = get_user_data(dates,config)
        user_data = [[dates[i],user_data[i]] for i in range(len(user_data))]

    if usr_shuffle:
        import random
        random.shuffle(user_data)

    if usr_divide:
        # Train model on a subset of the data
        if not full_data:
            user_data = user_data[:7]
        dates = [x[0] for x in user_data]
        all_data = [x[1] for x in user_data]
    else:
        all_data = all_data[:1000]
        import random
        random.shuffle(all_data)

    for i in range(len(all_data)):

        remainder = len(all_data[i]) % tw

        if remainder:
            all_data[i] = all_data[i][:-1*remainder]

    # Split data into train, valid, and test
    if not all_seqs:
        train_data, valid_data, test_data, test_data_size = split_datasets(all_data)
    if include_dates and not all_seqs:
        train_dates, valid_dates, test_dates, test_data_size = split_datasets(dates)
    else:
        train_dates = None
        valid_dates = None
        test_dates = None

    if not concat_sax and not usr_divide:
        sax = ts_to_string(znorm(np.array(all_data)), cuts_for_asize(5))
    elif usr_divide:
        sax = []
        sax_dates = []
        max_size = 0
        for sublist in all_data:
            if len(sublist) > max_size:
                max_size = len(sublist)
            sax.append(ts_to_string(znorm(np.array(sublist)), cuts_for_asize(5)))

    # Split SAX into train, valid, and test
    if not all_seqs:
        train_sax, valid_sax, test_sax, test_data_size = split_datasets(sax)

    if usr_divide and not all_seqs:
        train_norm = [torch.FloatTensor(znorm(np.array(x))).view(-1) for x in train_data]
        valid_norm = [torch.FloatTensor(znorm(np.array(x))).view(-1) for x in valid_data]
        test_norm = [torch.FloatTensor(znorm(np.array(x))).view(-1) for x in test_data]
    elif not all_seqs:
        train_norm = [torch.FloatTensor(znorm(np.array(train_data))).view(-1)]
        valid_norm = [torch.FloatTensor(znorm(np.array(valid_data))).view(-1)]
        test_norm = [torch.FloatTensor(znorm(np.array(test_data))).view(-1)]

    if all_seqs:
        all_norm = [torch.FloatTensor(znorm(np.array(x))).view(-1) for x in all_data]


    if return_dataset:
        return all_data, train_dates, valid_dates, test_dates, train_data, train_norm, train_sax, valid_data, valid_norm, valid_sax, test_data, test_norm, test_sax, max_size, test_data_size
    if all_seqs:
        return all_data, dates, all_norm, sax, max_size
    return train_dates, valid_dates, test_dates, train_data, train_norm, train_sax, valid_data, valid_norm, valid_sax, test_data, test_norm, test_sax, max_size, test_data_size


def create_inout_sequences(input_data, norm_data, sax_rep, tw_list, sid, key_list=["Calorie Intake"], sax_bool=False, sax_bool2=False, summary_types=["SESTW"], quick=False, include_series=False, squish_tokens=False, ge_norm=False, include_templates=False, dates=None, include_tw_sax=False):
    inout_seq = []
    age = 23
    activity_level = "active"
    alpha = 0.9
    attr_list = key_list.copy()
    pid_list = [""]*len(key_list)
    attr = "MyFitnessPal"
    alphabet = string.ascii_lowercase
    letter_map = dict()
    for i in range(5):
        letter_map[alphabet[i]] = i+1
    letter_map_list = [letter_map]
    alpha_sizes = [5]
    alphabet_list = [alphabet]
    summarizer_7 = ["extremely low","very low","low","moderate","high","very high","extremely high"]

    # Retrieve weekdays
    if dates != None:
        weekdays = []
        for j in range(len(dates)):
            weekdays.append(retrieve_weekdays(attr,dates[j],weekday_dict))

    full_seqs = []
    for j in range(len(input_data)):
        uid = j

        for tw in tw_list:
            TW = getTW(tw)
            singular_TW = ""
            if TW != None:
                singular_TW = TW[:-1]

            L = len(input_data[j])
            if L==tw:
                L += 1

            # Create SAX representation at TW granularity
            tw_sax_list = []

            data = input_data[j]
            data_norm = norm_data[j].tolist()
            sax_data = sax_rep[j]

            tw_num = int(len(data)/tw)
            data = data[:tw_num*tw]
            data_norm = data_norm[:tw_num*tw]
            sax_data = sax_data[:tw_num*tw]
            import copy

            shift_data = copy.deepcopy(data)
            shift_norm = copy.deepcopy(data_norm)
            shift_sax = copy.deepcopy(sax_data)

            for summary_type in summary_types:
                tw_cnt = 0
                for i in range(L-tw):
                    tw_sax = ts_to_string(paa(znorm(np.array(shift_data)),tw_num), cuts_for_asize(alpha_sizes[0]))
                    tw_sax_list = [tw_sax]

                    prev_start_day = int(tw*(len(tw_sax)-2))
                    start_day = int(tw*(len(tw_sax)-1))
                    end_day = int(tw*len(tw_sax))

                    if summary_type == "SETW":
                        seq = shift_data[-1*tw:]
                        seq_norm = shift_norm[-1*tw:]
                        sax = tw_sax[-1]
                        if len(seq) < tw or sax == "":
                            break
                        summaries = generateSETW(attr,key_list,pid_list,singular_TW,[seq],[],letter_map_list,alpha_sizes,tw,sax,age=age,activity_level=activity_level,quick=True)
                        summaries = [summaries]
                        truths = [1]*len(summaries)
                    elif summary_type == "SESTW":
                        seq = shift_data[-1*tw:]
                        seq_norm = shift_norm[-1*tw:]
                        sax = shift_sax[-1*tw:]
                        summaries, truths = generateSESTW(attr,key_list,[sax],letter_map_list,alpha,alpha_sizes,tw,TW,quick=True)
                    elif summary_type == "SESTWQ":
                        summaries = None
                        pass
                    elif summary_type == "EC":
                        seq = torch.tensor(shift_data[prev_start_day:])
                        seq_norm = torch.tensor(shift_norm[prev_start_day:])
                        sax = shift_sax[-1*tw:]
                        if tw_cnt == len(tw_sax):
                            random.shuffle(data)
                            tw_sax = ts_to_string(paa(znorm(np.array(data)),tw_num), cuts_for_asize(alpha_sizes[0]))
                            tw_sax_list = [tw_sax]
                            tw_cnt = 0

                        summaries = generateEC(attr,key_list,[sax],tw_sax_list,alpha,alpha_sizes,letter_map_list,TW,tw,age=age,activity_level=activity_level,quick=True,relative_tw=True)
                        truths = [1]
                        tw_sax_list[0] = tw_sax_list[0][-1:] + tw_sax_list[0][:-1]
                        tw_cnt += 1
                    elif summary_type == "GC":
                        seq = torch.tensor(shift_data[prev_start_day:]).float()
                        seq_norm = torch.tensor(shift_norm[prev_start_day:])
                        sax = shift_sax[-1*tw:]
                        if tw_cnt == len(tw_sax):
                            random.shuffle(data)
                            tw_sax = ts_to_string(paa(znorm(np.array(data)),tw_num), cuts_for_asize(alpha_sizes[0]))
                            tw_sax_list = [tw_sax]
                            tw_cnt = 0

                        summaries = generateGC(attr,attr_list,key_list,[input_data[j]],[sax_rep[j]],tw_sax_list,alpha,alpha_sizes,letter_map_list,TW,tw,prev_start_day,start_day,end_day,age=age,activity_level=activity_level,quick=True,relative_tw=True)
                        truths = [1]
                        tw_sax_list[0] = tw_sax_list[0][-1:] + tw_sax_list[0][:-1]
                        tw_cnt += 1
                        if not ge_norm:
                            seq_norm = torch.FloatTensor(seq)
                            shift_norm = shift_data
                    elif summary_type == "GE":
                        seq = shift_data[-1*tw:]
                        seq_norm = shift_norm[-1*tw:]
                        sax = shift_sax[-1*tw:]
                        summaries, truths = generateGE(attr,key_list,key_list,sax,[seq],letter_map_list,alpha,alpha_sizes,TW,age=age,activity_level=activity_level,start_day=i,end_day=i+tw,quick=True)
                        if not ge_norm:
                            seq_norm = torch.FloatTensor(seq)
                    elif summary_type == "ST":
                        seq = shift_data[-1*tw:]
                        seq_norm = shift_norm[-1*tw:]
                        sax = shift_sax[-1*tw:]
                        summaries, truths = generateST(attr,key_list,[seq],letter_map_list,alpha_sizes,alpha,TW,age=age,activity_level=activity_level,quick=True)
                    elif summary_type == "DB":
                        seq = shift_data[-1*tw:]
                        seq_norm = shift_norm[-1*tw:]
                        sax = sax_rep[j]
                        summaries, truths = generateDB(attr,key_list,[sax],summarizer_7,alpha,alpha_sizes,letter_map_list,alphabet_list,tw,TW,age,activity_level,weekdays[j],quick=True)
                    elif summary_type == "GA":
                        seq = shift_data[-1*tw:]
                        seq_norm = shift_norm[-1*tw:]
                        sax = sax_rep[j]
                        summaries = generateGA(attr,{"Calories" : input_data[j]},key_list,[sax],summarizer_7,i,i+tw,alpha,alpha_sizes,letter_map_list,alphabet_list,tw,TW,age,activity_level,dates[j],quick=True)
                        truths = [1]

                    if sax_bool:
                        sax = list(sax)
                        sax = torch.tensor(np.array([float(alphabet.index(x)+1) for x in sax]))
                        seq = sax.float()
                        seq_norm = seq
                    elif sax_bool2:
                        sax = list(sax)
                        indices = dict()
                        index = 0
                        seq = [0]*max(alpha_sizes)
                        for i in range(len(alphabet)):
                            if i == max(alpha_sizes):
                                break
                            letter = alphabet[i]
                            indices[letter] = index
                            index+=1

                        for letter in sax:
                            seq[indices[letter]] += 1
                        seq = torch.tensor(np.array(seq)).float()
                        seq_norm = seq

                    if summaries != None:
                        for index in range(len(summaries)):
                            summary = summaries[index]
                            if summary == None:
                                continue
                            if squish_tokens:
                                summary = combine_special_tokens(summary)

                            truth = round(truths[index],2)

                            tokens = summary
                            template = build_template(summary)

                            import copy
                            series = copy.deepcopy(norm_data[j].tolist())
                            if summary_type in ["EC","GC"]:
                                config['seq_length'] = tw*2
                                if summary_type == "GC":
                                    shift_norm = [float(x) for x in series]

                            pad_token = 0
                            if sax_bool:
                                shift_norm = [float(alphabet.index(x)+1) for x in sax_rep[j]]
                                pad_token = -1

                            pad = [pad_token] * (config['max_series_size'] - len(shift_norm))
                            series_norm = shift_norm + pad

                            if quick:
                                if include_series:
                                    seq_norm = torch.tensor(list(seq_norm) + series_norm)
                                inout = [seq_norm, summary]
                                if include_templates:
                                    inout.append(template)
                                inout_seq.append(tuple(inout))
                            else:
                                seq_ = (seq_norm, torch.tensor(series_norm), summary, truth, sax, sid, uid)
                                if include_tw_sax:
                                    seq_ = (seq_norm, torch.tensor(series_norm), summary, truth, sax, sid, uid, tw_sax)
                                inout_seq.append(seq_)
                            sid+=1

                    shift_data = shift_data[-1*tw:] + shift_data[:-1*tw]
                    shift_norm = shift_norm[-1*tw:] + shift_norm[:-1*tw]
                    shift_sax = shift_sax[-1*tw:] + shift_sax[:-1*tw]

    return inout_seq

def write_sequences(summary_type, inout_seq, device, quick=False, include_tw_sax=False):

    with open(summary_type + '_' + str(device) + '.csv','w',newline='') as csvfile:
        if quick:
            header = ["Sequence","Summary","Template"]
        else:
            header = ["Sequence","Series","Summary","Truth","SAX","sid","uid"]

        if include_tw_sax:
            header.append("TWSAX")

        csvwriter = csv.writer(csvfile)
        csvwriter.writerow(header)

        for seq in inout_seq:
            seq = list(seq)

            # Create seq and series strings
            seq[0] = construct_string(seq[0])
            seq[1] = construct_string(seq[1])

            csvwriter.writerow(seq)

if __name__ == "__main__":

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    full_data = (str(device) != "cpu")

    transformer = False

    config_file = 'ED_config.json'
    if transformer:
        config_file = 'transformer_config.json'

    # Load config
    with open(config_file,'r') as jsonfile:
        config = json.load(jsonfile)

    usr_divide = config['usr_divide']
    concat_sax = config['concat_sax']
    usr_shuffle = config['usr_shuffle']
    summary_types = config['summary_types']
    time_windows = config['time_windows']
    use_series = config['use_series']
    squish_tokens = config['squish_tokens']
    ge_norm = config['ge_norm']
    include_tw_sax = False

    if transformer:
        include_templates = config['include_templates']
    else:
        include_templates = False

    include_dates = ("DB" in summary_types or "GA" in summary_types)

    # Load data
    all_data, dates, all_norm, all_sax, max_size = get_data(time_windows,usr_divide,concat_sax,full_data,usr_shuffle,include_dates=include_dates,all_seqs=True)

    # Create sequences
    inout_seq = create_inout_sequences(all_data, all_norm, all_sax, time_windows, 0, summary_types=summary_types, quick=transformer, include_series=use_series, squish_tokens=squish_tokens, ge_norm=ge_norm, include_templates=include_templates, dates=dates, include_tw_sax=include_tw_sax)

    # Write sequences to CSV
    write_sequences(summary_types[0], inout_seq, device, quick=transformer)

    print("Done")


