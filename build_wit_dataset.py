import csv
import os
from proto_lib import analyze_patterns, summarizer_to_SAX, retrieve_weekdays
from saxpy.sax import ts_to_string
from saxpy.znorm import znorm
from saxpy.alphabet import cuts_for_asize
from saxpy.paa import paa
import numpy as np
import torch

weekday_dict = { 1 : "Monday",
                 2 : "Tuesday",
                 3 : "Wednesday",
                 4 : "Thursday",
                 5 : "Friday",
                 6 : "Saturday",
                 7 : "Sunday" }

db_fn_prefix = "series_db_WIT_concat"

# Parameters for SPADE
min_conf = 0.8
min_sup = 0.2
path = "." # Path for pattern data
cygwin_path = r"C:\cygwin64\bin" # Path to Cygwin
output_filename = 'WIT_sequences_concat.csv'
header = ["Sequence","Series","Summary","Truth","SAX","sid","uid"]

# If stops prematurely, start from most recent sid, uid
def retrieveLastSpot():
    if output_filename not in os.listdir("."):
        return 0, 0

    with open(output_filename,newline='') as csvfile:
        csvreader = csv.reader(csvfile)
        sid = 0
        uid = 0
        for row in csvreader:
            if row == header:
                continue
            sid = int(row[-2])
            uid = int(row[-1])

    return sid, uid

if __name__ == "__main__":
    data_path = "mfp_data"
    directory = os.listdir(data_path)

    sid, uid = retrieveLastSpot()

    if uid == 0:
        with open(output_filename,'w',newline='') as csvfile:
            csvwriter = csv.writer(csvfile)
            csvwriter.writerow(header)

    file_cnt = uid
    for filename in directory:
        if ".csv" not in filename:
            continue
        elif file_cnt > 0: # Each file is a different user. Skip latest user in file since pattern output isn't the same every time (otherwise could be duplicates)
            file_cnt -= 1
            continue

        # Retrieve data
        with open(data_path + '/' + filename,newline='') as csvfile:
            csvreader = csv.reader(csvfile)
            calories = []
            dates = []
            header = True
            for row in csvreader:
                if header:
                    header = False
                    continue
                if len(row) == 0:
                    continue
                else:
                    calories.append(int(row[1]))
                    dates.append(row[0])

        # Remove last week if it isn't full
        series_length = len(calories)
        cutoff = series_length % 7

        if cutoff:
            calories = calories[:-1*cutoff]

        # Normalize data
        pre_norm = calories
        calories = znorm(np.array(calories))

        # Retrieve weekdays
        dates = retrieve_weekdays("MyFitnessPal",dates,weekday_dict)

        num_variations = int(len(calories))
        output = []
        for _ in range(num_variations):
            first = [calories[0].item()]
            second = calories[1:].tolist()
            calories = torch.tensor(second + first)
            dates = dates[-1:] + dates[:-1]

            # Retrieve input sequence
            series_str = ""
            for i in range(len(calories)):
                series_str += str(calories[i])
                if i != len(calories)-1:
                    series_str += "|"

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
            proto_cnt = 0

            # Construct mapping from letters in alphabet to integers (starting at 1)
            import string
            alphabet = string.ascii_lowercase
            alphabet_list = [alphabet]
            letter_map = dict()
            for i in range(alpha_size):
                letter_map[alphabet[i]] = i+1
            letter_map_list = [letter_map]

            # SAX
            full_sax_rep = ts_to_string(calories, cuts_for_asize(alpha_size))
            sax = list(full_sax_rep)
            sax_list = [full_sax_rep]
            tw_sax_list = [ts_to_string(paa(calories,tw), cuts_for_asize(alpha_size))]

            # Retrieve summaries
            summary_list, supports, proto_cnt, numsum_list, summarizers_list = analyze_patterns(key_list,sax_list,alphabet_list,letter_map_list,weekday_dict,tw,alpha_sizes,db_fn_prefix,path,cygwin_path,min_conf,min_sup,proto_cnt,weekdays=dates)


            for i in range(len(summary_list)):

                # Get SAX version of pattern
                input_str = ""
                pre_list = summarizers_list[i][0][0]
                suf_list = summarizers_list[i][1][0]

                pre_list = [summarizer_to_SAX(x,alpha_size) for x in pre_list if x not in weekday_dict.values()]
                suf_list = [summarizer_to_SAX(x,alpha_size) for x in suf_list if x not in weekday_dict.values()]

                pre_sax = ''.join(pre_list)
                suf_sax = ''.join(suf_list)

                find_sax = pre_sax + suf_sax

                # Find indices of all instances of pattern
                import re
                indices = [x.start() for x in re.finditer('(?='+find_sax+')',full_sax_rep)]

                # Find weeks starting at instances of patterns
                input_str = ""
                for index in indices:
                    index_range = range(index,min(index+tw,len(calories)-1))
                    for j in index_range:
                        input_str += str(calories[j]) + "|"
                input_str = input_str[:-1]

                #if len(input_str.split("|")) < 100:
                    #print(pre_norm)
                    #print(calories)
                    #input(series_str)

                output.append([input_str,series_str,summary_list[i],supports[i],full_sax_rep,sid,uid])
                sid += 1

        with open(output_filename,'a+',newline='') as csvfile:
            csvwriter = csv.writer(csvfile)
            for i in range(len(output)):
                csvwriter.writerow(output[i])

        uid += 1

        print(len(directory) - uid, "users remaining")


    print("done")