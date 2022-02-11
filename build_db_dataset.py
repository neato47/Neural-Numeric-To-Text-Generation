import csv
import os
from proto_lib import generateDB, summarizer_to_SAX, retrieve_weekdays
from saxpy.sax import ts_to_string
from saxpy.znorm import znorm
from saxpy.alphabet import cuts_for_asize
from saxpy.paa import paa
import numpy as np

weekday_dict = { 1 : "Monday",
                 2 : "Tuesday",
                 3 : "Wednesday",
                 4 : "Thursday",
                 5 : "Friday",
                 6 : "Saturday",
                 7 : "Sunday" }

if __name__ == "__main__":
    data_path = "mfp_data"
    directory = os.listdir(data_path)

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
            dates = dates[:-1*cutoff]

        # Normalize data
        calories = znorm(np.array(calories))

        # Retrieve weekdays
        dates = retrieve_weekdays("MyFitnessPal",dates,weekday_dict)

        num_variations = int(len(calories))
        for _ in range(num_variations):
            calories = list(calories[-1:]) + list(calories[:-1])

            dates = list(dates[-1:]) + list(dates[:-1])

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
            summarizer_7 = ["extremely low","very low","low","moderate","high","very high","extremely high"]
            summary_list, truth_list = generateDB(attr,key_list,sax_list,summarizer_7,alpha,alpha_sizes,letter_map_list,alphabet_list,tw,TW,age,activity_level,dates,quick=True)

            for i in range(len(summary_list)):

                # Get SAX version of pattern
                summary = summary_list[i][:-2].split(" ")
                summarizer = summary[6]
                if summarizer == "very" or summarizer == "extremely":
                    summarizer += " " + summary[7]
                summarizer_sax = summarizer_to_SAX(summarizer,alpha_size)
                weekday = summary[-1]

                date_indices = [i for i in range(len(dates)) if dates[i] == weekday]
                indices = [i for i in date_indices if full_sax_rep[i] == summarizer_sax]

                # Find weeks starting at instances of patterns
                input_str = ""
                for index in indices:
                    index_range = range(index,min(index+tw,len(calories)-1))
                    for j in index_range:
                        input_str += str(calories[j]) + "|"
                input_str = input_str[:-1]

                output.append([input_str,series_str,summary_list[i],truth_list[i],full_sax_rep,sid,uid])
                sid += 1

        uid += 1
        print(str(len(directory)-uid) + " users remaining")

    with open('DB_sequences_concat.csv','w',newline='') as csvfile:
        csvwriter = csv.writer(csvfile)
        header = ["Sequence","Series","Summary","Truth","SAX","sid","uid"]
        csvwriter.writerow(header)
        for i in range(len(output)):
            csvwriter.writerow(output[i])

    print("done")