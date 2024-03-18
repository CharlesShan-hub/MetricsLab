import os
import csv

def get_all_files_in_path(path):
    files = []
    for file in sorted(os.listdir(path)):
        if file.startswith('.'):
            continue
        if os.path.isfile(os.path.join(path, file)):
            files.append([file])
        elif os.path.isdir(os.path.join(path, file)):
            sub_files = get_all_files_in_path(os.path.join(path, file))
            for sub_file in sub_files:
                files.append([file] + sub_file)
    return files

def add_full_path(files):
    for i in range(len(files)):
        files[i] = [os.path.join(*files[i])]
    return files

def add_type(files):
    for i in range(len(files)):
        IR_flag = False
        for flag in ['IR','ir.','thermal']:
            if flag in files[i][0]:
                IR_flag = True
        files[i] = [IR_flag]+files[i]
    return files

def add_id(files):
    count1 = count2 = 0
    for i in range(len(files)):
        if(files[i][0]==True):
            files[i] = [count1]+files[i]
            count1 = count1 + 1
        else:
            files[i] = [count2]+files[i]
            count2 = count2 + 1
    return files

def add_name(files):
    for i in range(len(files)):
        if(files[i][1]==True):
            files[i] = ['TNO|IR|'+str(files[i][0])]+files[i]
        else:
            files[i] = ['TNO|VIS|'+str(files[i][0])]+files[i]
    return files

path = './TNO'
files = get_all_files_in_path(path)
files = add_full_path(files)
files = add_type(files)
files = add_id(files)
files = add_name(files)

with open('./logs/TNO_dictionary.csv', 'w', newline='') as file:
    writer = csv.writer(file)
    writer.writerows(files)
