import os
import csv
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
    # shutil.copy('./TNO/'+data[item]['PATH'], './MYTNO1/'+str(data[item]['ID'])+t+'.'+data[item]['PATH'].split('.')[-1])

'''
import matplotlib.pyplot as plt
mi = [1.2629, 1.0765, 0.7445, 1.3802, 1.1020, 0.9410, 2.8204, 1.7022, 2.4377, 1.2624, 1.2100, 1.0020, 1.2700]
te_190 = [  251.9935,277.1870, 32.7949, 33.3386,316.3900, 87.2242, 35.7358, 68.2493,355.0536,240.9850,514.4585, 43.5443, 33.7628]
te_185 = [158.5506, 201.2620, 26.7743, 25.0559, 217.3696, 65.5279, 29.1573, 51.5961, 232.5418, 163.8177, 331.7446, 35.7764, 26.8387]
te_184 = [144.8935,188.9742,25.7544,23.7508,201.9328,61.9618,28.0376,48.8600,214.4324,151.9938,304.3018,34.4360,25.6821]
te_183 = [132.5332,177.4989,24.7868,22.5402,187.6841, 58.6145,26.9743,46.2922,197.9685,141.1337,279.2694,33.1577,24.5904]
te_182 = [121.3401,166.7799,23.8682,21.4163,174.5275,55.4718,25.9639,43.8817,182.9856,131.1532,256.4287,31.9383,23.5594]
te_181 = [111.1976,156.7649,22.9956,20.3718,162.3751,52.5203,25.0036,41.6181,169.3370,121.9758,235.5817,30.7746,22.5853]
te_180 = [102.0016,147.4052,22.1660,19.4001,151.1462,49.7475,24.0902,39.4918,156.8911,113.5319,216.5483,29.6636,21.6644]
te_175 = [67.2589,108.9667,18.5785,15.4420,106.4727,38.1841,20.1387,30.6267,108.9946,80.2930,143.3684,24.8074,17.7486]
te_170 = [45.5543, 81.3518, 15.7421, 12.5984, 76.0518, 29.6441, 17.0181, 24.0793, 77.8820, 57.9829, 96.4812, 20.9138, 14.7455]
te_165 = [31.7428,61.3731,13.4675,10.4995,55.1232,23.2843,14.5223,19.1987,57.1430,42.7616,66.1477,17.7646,12.4106]
te_160 = [22.7732,46.8099,11.6195, 8.9081,40.5635,18.5063,12.5005,15.5235,42.9612,32.1961,46.3057,15.1958,10.5692]
te_150 = [12.7609, 28.1757, 8.8364, 6.6808, 22.9916, 12.1065, 9.4621, 10.5683, 25.8800, 19.3492, 24.3284, 11.3324, 7.8998]
te_140 = [7.8802,17.7502, 6.8699, 5.1986,13.7777, 8.2689, 7.3042, 7.5474,16.6915,12.4306,14.0915, 8.6378, 6.0887]
te_130 = [5.1681, 11.6450, 5.4099, 4.1162, 8.5532, 5.8383, 5.6640, 5.5817, 11.2728, 8.3497, 8.9177, 6.6932, 4.7743]
te_120 = [3.3225,7.8030,4.2336,3.2230,5.1401,4.1503,4.2540,4.1625,7.6882,5.5851,5.9679,5.2167,3.7164]
te_110 = [1.2662, 4.8169, 3.0044, 2.2047, 1.7485, 2.6042, 2.4985, 2.7935, 4.5143, 2.8554, 3.7264, 3.9056, 2.5717]
te_101 = [-19.5154, -9.6679, -5.6413, -6.2604, -28.6412, -7.5607, -14.4913, -6.6281, -16.2042, -19.7543, -8.5319, -1.8701, -6.2008]
te_000 = [  -2.2440e+04,-1.4585e+04,-8.9908e+03,-8.9472e+03,-3.2598e+04,-1.0566e+04,-1.8206e+04,-9.8143e+03,-2.1567e+04,-2.4062e+04,-1.2568e+04,-5.6210e+03,-9.2046e+03]
nmi =[0.2474, 0.2222, 0.1509, 0.2684, 0.2281, 0.1830, 0.5412, 0.3292, 0.4941, 0.2484, 0.2381, 0.2026, 0.2517]

def normalize_scores(scores):
    min_score = min(scores)
    max_score = max(scores)
    return [(score - min_score) / (max_score - min_score) for score in scores]

mi = normalize_scores(mi)
te_190 = normalize_scores(te_190)
te_185 = normalize_scores(te_185)
te_184 = normalize_scores(te_184)
te_183 = normalize_scores(te_183)
te_182 = normalize_scores(te_182)
te_181 = normalize_scores(te_181)
te_180 = normalize_scores(te_180)
te_175 = normalize_scores(te_175)
te_170 = normalize_scores(te_170)
te_165 = normalize_scores(te_165)
te_160 = normalize_scores(te_160)
te_150 = normalize_scores(te_150)
te_140 = normalize_scores(te_140)
te_130 = normalize_scores(te_130)
te_120 = normalize_scores(te_120)
te_110 = normalize_scores(te_110)
te_101 = normalize_scores(te_101)
te_100 = normalize_scores(te_000)

nmi = normalize_scores(nmi)

combined_scores = list(zip(mi, te_190,te_185,te_184,te_183,te_182,te_181,te_180, te_175, te_170, te_165, te_160, te_150, te_140, te_130,te_120,te_110, te_101,te_100, nmi))
combined_scores.sort(key=lambda x: x[0])
mi, te_190,te_185,te_184,te_183,te_182,te_181,te_180, te_175, te_170, te_165, te_160, te_150, te_140, te_130,te_120,te_110, te_101,te_100, nmi = zip(*combined_scores)


plt.plot(mi, label='MI')
plt.plot(nmi,label='NMI')
plt.plot(te_100, label='TE a=1.00')
plt.plot(te_101, label='TE a=1.01')
plt.plot(te_110, label='TE a=1.10')
plt.plot(te_120, label='TE a=1.20')
plt.plot(te_130, label='TE a=1.30')
plt.plot(te_140, label='TE a=1.40')
plt.plot(te_150, label='TE a=1.50')
plt.plot(te_160, label='TE a=1.60')
plt.plot(te_165, label='TE a=1.65')
plt.plot(te_170, label='TE a=1.70')
plt.plot(te_175, label='TE a=1.75')
plt.plot(te_180, label='TE a=1.80')
plt.plot(te_181, label='TE a=1.81')
plt.plot(te_182, label='TE a=1.82')
plt.plot(te_183, label='TE a=1.83')
plt.plot(te_184, label='TE a=1.84')
plt.plot(te_185, label='TE a=1.85')
plt.plot(te_190, label='TE a=1.90')
plt.legend()
plt.show()
'''
