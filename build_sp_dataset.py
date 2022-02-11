import csv
import os
from proto_lib import generateCB, generateSP
from saxpy.sax import ts_to_string
from saxpy.znorm import znorm
from saxpy.alphabet import cuts_for_asize
from saxpy.paa import paa
import numpy as np

if __name__ == "__main__":
    data_path = "mfp_data"
    directory = os.listdir(data_path)

    cluster_input = True
    uid = 0
    sid = 0
    output = []
    for filename in directory:
        if ".csv" not in filename:
            continue

        # Retrieve data
        with open(data_path + '/' + filename,newline='') as csvfile:
            csvreader = csv.reader(csvfile)
            calories = []
            header = True
            for row in csvreader:
                if header:
                    header = False
                    continue
                if len(row) == 0:
                    continue
                else:
                    calories.append(int(row[1]))

        # Remove last week if it isn't full
        series_length = len(calories)
        cutoff = series_length % 7

        if cutoff:
            calories = calories[:-1*cutoff]

        # Setup
        attr = "MyFitnessPal"
        attr_list = [attr]
        key_list = ["Calorie Intake"]
        alpha_size = 5
        alpha_sizes = [alpha_size]
        tw = 7
        TW = "weeks"
        alpha = 0.9
        age = 23
        activity_level = "active"

        num_variations = int(len(calories))
        for i in range(num_variations):
            calories = calories[-1:] + calories[:-1]
            data_list = [calories]

            # Normalize data and retrieve series sequence
            calories_norm = znorm(np.array(calories))
            series_str = ""
            for j in range(len(calories_norm)):
                series_str += str(calories_norm[j])
                if j != len(calories_norm)-1:
                    series_str += "|"

            # Construct mapping from letters in alphabet to integers (starting at 1)
            import string
            alphabet = string.ascii_lowercase
            letter_map = dict()
            for j in range(alpha_size):
                letter_map[alphabet[j]] = j+1
            letter_map_list = [letter_map]

            # SAX
            full_sax_rep = ts_to_string(znorm(np.array(calories)), cuts_for_asize(alpha_size))
            sax = list(full_sax_rep)
            sax_list = [full_sax_rep]
            tw_sax_list = [ts_to_string(paa(znorm(np.array(calories)),int(len(full_sax_rep)/tw)), cuts_for_asize(alpha_size))]

            # Retrieve summary
            _, _, _, _, _, _, _, _, tw_index, cluster_data, indices_, clusters, _, summ_indices = generateCB(attr,attr_list,key_list,full_sax_rep,tw_sax_list,sax_list,data_list,letter_map_list,alpha_sizes,alpha,tw,TW,age,activity_level) # TODO: Use clusters
            if cluster_data == None: continue
            last_tw = None
            for j in range(len(clusters)):
                if tw_index in clusters[j]:
                    if len(clusters[j]) > 1:
                        last_tw = clusters[j][-2]
                    break
            if last_tw == None: continue
            sp_summary = generateSP(attr,key_list,cluster_data,tw_index,indices_,letter_map_list,alpha_sizes,age,activity_level,quick=True)
            truth = 1

            # Retrieve input sequence
            num_weeks = int(len(calories_norm)/tw)-1
            #input_sequence = list(calories_norm[tw*last_tw:tw*(last_tw+1)]) + list(calories_norm[tw*(last_tw+1):tw*(last_tw+2)]) + list(calories_norm[tw*num_weeks:tw*(num_weeks+1)])
            input_sequence = list(calories_norm[tw*last_tw:tw*(last_tw+1)]) + list(calories_norm[tw*(last_tw+1):tw*(last_tw+2)])
            #print(last_tw,input_sequence,list(calories_norm[tw*last_tw:tw*(last_tw+1)]),list(calories_norm[tw*(last_tw+1):tw*(last_tw+2)]),list(calories_norm[tw*num_weeks:tw*(num_weeks+1)]))

            input_str = ""
            for j in range(len(input_sequence)):
                input_str += str(input_sequence[j])
                if j != len(input_sequence)-1:
                    input_str += "|"

            if cluster_input and summ_indices != None:
                weeks = summ_indices[:-1]
                seq_str = ""
                for j in range(0,len(weeks),2):
                    day_index = weeks[j][0]
                    sequence = calories_norm[day_index:(day_index + tw*2)]

                    for j in range(len(sequence)):
                        seq_str += str(sequence[j])
                        seq_str += "|"

                seq_str += input_str

                zero_sequence = [0]*tw
                for j in range(len(zero_sequence)):
                    seq_str += str(zero_sequence[j])
                    seq_str += "|"

                input_str = seq_str[:-1]


            #print(tw,cluster_input,summ_indices,sp_summary)
            #input(input_str)
            output.append([input_str,series_str,sp_summary,truth,sax,sid,uid])

            sid += 1
        uid += 1

        print(str(len(directory)-uid) + " left")

    filename = 'SP_sequences.csv'
    if cluster_input:
        filename = 'SP_sequences_cluster_input_short.csv'

    with open(filename,'w',newline='') as csvfile:
        csvwriter = csv.writer(csvfile)
        header = ["Sequence","Series","Summary","Truth","SAX","sid","uid"]
        csvwriter.writerow(header)
        for i in range(len(output)):
            csvwriter.writerow(output[i])

    print("Done")