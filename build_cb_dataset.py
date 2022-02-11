import csv
import os
from proto_lib import generateCB
from saxpy.sax import ts_to_string
from saxpy.znorm import znorm
from saxpy.alphabet import cuts_for_asize
from saxpy.paa import paa
import numpy as np

if __name__ == "__main__":
    data_path = "mfp_data"
    directory = os.listdir(data_path)
    
    cluster_input = False
    uid = 0
    sid = 0
    description_output = []
    cluster_output = []
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
            for i in range(len(calories_norm)):
                series_str += str(calories_norm[i])
                if i != len(calories_norm)-1:
                    series_str += "|"            
            
            # Retrieve input sequence
            input_sequence = list(calories_norm[-7:])
            input_str = ""
            for i in range(len(input_sequence)):
                input_str += str(input_sequence[i])
                if i != len(input_sequence)-1:
                    input_str += "|"          
            
            # Construct mapping from letters in alphabet to integers (starting at 1)
            import string
            alphabet = string.ascii_lowercase
            letter_map = dict()  
            for i in range(alpha_size):
                letter_map[alphabet[i]] = i+1  
            letter_map_list = [letter_map]
            
            # SAX 
            full_sax_rep = ts_to_string(znorm(np.array(calories)), cuts_for_asize(alpha_size))
            sax = list(full_sax_rep)
            sax_list = [full_sax_rep]
            tw_sax_list = [ts_to_string(paa(znorm(np.array(calories)),int(len(full_sax_rep)/tw)), cuts_for_asize(alpha_size))]
            
            # Retrieve summary
            quick = not cluster_input
            quick_prov = not quick
            cluster_summary, truth_list, _, cluster_indices = generateCB(attr,attr_list,key_list,full_sax_rep,tw_sax_list,sax_list,data_list,letter_map_list,alpha_sizes,alpha,tw,TW,age,activity_level,quick=quick,quick_prov=quick_prov,thres=0)
            truth = truth_list[0]
            
            if cluster_input:
                print(tw_sax_list)
                print(cluster_summary)
                input(cluster_indices)
            
            summary_split = cluster_summary.split(".")
            description = summary_split[0] + "."
            cluster = summary_split[1][1:] + "."
        
            description_output.append([input_str,series_str,description,truth,sax,sid,uid])
            cluster_output.append([input_str,series_str,cluster,truth,sax,sid,uid])
        
            sid += 1
        uid += 1    
        
    with open('CB_sequences_description_split_1.csv','w',newline='') as csvfile:
        csvwriter = csv.writer(csvfile)
        header = ["Sequence","Series","Summary","Truth","SAX","sid","uid"]
        csvwriter.writerow(header)
        for i in range(len(description_output)):
            csvwriter.writerow(description_output[i])
            
    with open('CB_sequences_cluster_split_1.csv','w',newline='') as csvfile:
        csvwriter = csv.writer(csvfile)
        header = ["Sequence","Series","Summary","Truth","SAX","sid","uid"]
        csvwriter.writerow(header)
        for i in range(len(cluster_output)):
            csvwriter.writerow(cluster_output[i])    
            
    print("Done")