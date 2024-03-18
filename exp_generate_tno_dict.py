'''
    第一步: 根据精选后的数据集生成 ID 与路径的映射关系
'''
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

'''
    第二步: 根据映射关系，生成新的数据集
'''
import shutil

data = {}
with open('./logs/TNO_dictionary.csv', 'r', newline='') as file:
    csv_reader = csv.reader(file)
    for row in csv_reader:
        temp = {}
        temp['ID'] = row[1]
        temp['IS_IR'] = row[2]
        temp['PATH'] = row[3]
        data[row[0]] = temp

for item in data:
    if(data[item]['IS_IR']=='True'):
        t = 'IR'
    else:
        t = 'VIS'
    shutil.copy('./TNO/'+data[item]['PATH'], './MYTNO/'+t+'/'+str(data[item]['ID'])+'.'+data[item]['PATH'].split('.')[-1])

'''
    第三步: 调整图像格式，并且全保存成灰度图
'''
from PIL import Image

folder_path = './imgs/TNO/ir1'
to_path = './imgs/TNO/ir'
for filename in os.listdir(folder_path):
    file_path = os.path.join(folder_path, filename)
    if os.path.isfile(file_path):
        try:
            # 尝试打开图片文件
            image = Image.open(file_path)
            if image.format == 'TIFF':
                # 如果是tif格式的图片，则转换为bmp格式
                image = image.convert('RGB')
                image.save(file_path[:-4] + '.bmp')
            # 转换为黑白
            image_bw = image.convert('L')
            # 保存黑白图片
            image_bw.save(os.path.join(to_path, filename.split('.')[0] + '.bmp'))
            print(f"已处理：{filename}")
        except Exception as e:
            print(f"处理文件 {filename} 时出错：{e}")
