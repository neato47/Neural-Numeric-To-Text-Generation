import csv
import os
from proto_lib import analyze_patterns, summarizer_to_SAX
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

db_fn_prefix = "series_db_IT_concat"

# Parameters for SPADE
min_conf = 0.8
min_sup = 0.2
path = "/gpfs/u/scratch/TMPR/TMPRhrrs" # Path for pattern data
cygwin_path = r"C:\cygwin64\bin" # Path to Cygwin

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
            
        num_variations = int(len(calories))
        for _ in range(num_variations):
            calories = calories[-1:] + calories[:-1]   
     
            # Normalize data
            calories_norm = znorm(np.array(calories))
            
            # Retrieve input sequence
            series_str = ""
            for i in range(len(calories_norm)):
                series_str += str(calories_norm[i])
                if i != len(calories_norm)-1:
                    series_str += "|"
            
            # Setup
            attr = "MyFitnessPal"
            attr_list = [attr]
            key_list = ["Calorie Intake"]
            alpha_size = 5
            alpha_sizes = [alpha_size]
            tw = 7
            TW = "weeks"
            data_list = [calories]
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
            full_sax_rep = ts_to_string(znorm(np.array(calories)), cuts_for_asize(alpha_size))
            sax = list(full_sax_rep)
            sax_list = [full_sax_rep]
            tw_sax_list = [ts_to_string(paa(znorm(np.array(calories)),tw), cuts_for_asize(alpha_size))]
            
            # Retrieve summary
            summary_list, supports, proto_cnt, numsum_list, summarizers_list = analyze_patterns(key_list,sax_list,alphabet_list,letter_map_list,weekday_dict,tw,alpha_sizes,db_fn_prefix,path,cygwin_path,min_conf,min_sup,proto_cnt)
    
            for i in range(len(summary_list)):
                
                # Get SAX version of pattern
                input_str = ""
                pre_list = summarizers_list[i][0][0]
                suf_list = summarizers_list[i][1][0]
                
                pre_sax = ''.join([summarizer_to_SAX(x,alpha_size) for x in pre_list])
                suf_sax = ''.join([summarizer_to_SAX(x,alpha_size) for x in suf_list])    
                
                find_sax = pre_sax + suf_sax
                
                # Find indices of all instances of pattern
                import re
                indices = [x.start() for x in re.finditer('(?='+find_sax+')',full_sax_rep)]
                
                # Find weeks starting at instances of patterns
                input_str = ""
                for index in indices: 
                    index_range = range(index,min(index+tw,len(calories_norm)-1))
                    for j in index_range:
                        input_str += str(calories_norm[j]) + "|"
                input_str = input_str[:-1]
                            
                output.append([input_str,series_str,summary_list[i],supports[i],full_sax_rep,sid,uid])
                sid += 1
            
        uid += 1
            
        
    with open('IT_sequences_concat_large.csv','w',newline='') as csvfile:
        csvwriter = csv.writer(csvfile)
        header = ["Sequence","Series","Summary","Truth","SAX","sid","uid"]
        csvwriter.writerow(header)
        for i in range(len(output)):
            csvwriter.writerow(output[i])
            
    print("done")