import matplotlib.pyplot as plt
import csv

file = open('./Result/dsb2018_96_NestedUNet_woDS/log.csv')
data = list(csv.reader(file))
row = len(data)
col = len(data[0])
print("共 {} 行数据， {} 列属性".format(row - 1, col))

for i in range(1, col):
    x = list()
    y = list()
    plt.figure()
    for j in range(1, row):
        x.append(data[j][0])
        y.append(data[j][i])
    x = list(map(float, x))
    y = list(map(float, y))
    plt.plot(x, y, color=[1, 0.5, 0], lw=2)
    plt.xlabel(data[0][0])
    plt.ylabel(data[0][i])
    plt.title(data[0][i])
    plt.grid()
    plt.pause(0.5)
plt.show()