import sys
from sklearn.neural_network import MLPRegressor
import numpy as np
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
from sklearn.metrics import explained_variance_score
import statistics
import math

lines = [line.rstrip('\n') for line in open('data_naves.txt')]
num = len(lines)  # synolo  deigmatwn
arr = [0 for i in range(num)]
max_col = [-sys.maxsize-1 for w4 in range(9)]
min_col = [sys.maxsize for w5 in range(9)]
j=0
for data in lines:
    arr[j] = data.split(',')
    j=j+1

for row in range(num):
    for col in range(9):
        if float(max_col[col]) < float(arr[row][col]):
            max_col[col] = arr[row][col]

for row in range(num):
    for col in range(9):
        if float(min_col[col]) > float(arr[row][col]):
            min_col[col] = arr[row][col]



normal_arr = [[0 for i in range(8)] for j in range(num)]
new_arr = [[0 for i in range(5)] for j in range(num)]
xo = [0 for i in range(num)]
sum2 = 0
mesi_timh_x = [0 for i in range(8)]
r = [0 for i in range(8)]
meshtimh_y = 0
for row in range(num):

    for col in range(9):

        if col != 8:
            normal_arr[row][col] = ((float(arr[row][col]) - float(min_col[col])) / (float(max_col[col]) - float(min_col[col])))
            mesi_timh_x[col] = float(mesi_timh_x[col]) + float(normal_arr[row][col])
        else:
            xo[row] = ((float(arr[row][col]) - float(min_col[col])) / (float(max_col[col]) - float(min_col[col])))
            meshtimh_y = meshtimh_y + xo[row]


x = np.array(normal_arr)
y = np.array(xo)

for col in range(8):
    arhthmiths = 0
    par1 = 0
    par2 = 0
    for row in range(num):
        arhthmiths = arhthmiths + (x[row][col] - mesi_timh_x[col]/num)*(y[row] - meshtimh_y/num)
        par1 = par1 + (x[row][col] - mesi_timh_x[col]/num)*(x[row][col] - mesi_timh_x[col]/num)
        par2 = par2 + (y[row] - meshtimh_y/num)*(y[row] - meshtimh_y/num)
    r[col] = arhthmiths/(math.sqrt(par1)*math.sqrt(par2))


kf = KFold(n_splits=10)
#mlp = MLPRegressor(hidden_layer_sizes=(8, 8, 1,), max_iter=500,,)
#mlp = MLPRegressor(hidden_layer_sizes=(8, 8, 1,), learning_rate_init=0.01, max_iter=200, momentum=1,)
mlp = MLPRegressor(hidden_layer_sizes=(8, 9, 1), alpha=0.9, learning_rate_init=0.1, max_iter=2000, momentum=0.1,)
sum_mean = 0
sum_var = 0

for train_index, test_index, in kf.split(x):
    x_train, x_test = x[train_index], x[test_index]
    y_train, y_test = y[train_index], y[test_index]

    mlp.fit(x_train, y_train)
    predicted = mlp.predict(x_test)
    sum_mean = sum_mean + mean_squared_error(predicted, y_test)
    sum_var = sum_var + statistics.stdev(y_test, mean_squared_error(predicted, y_test))
i=0
for col in range(8):
    if r[col] > 0.045 and i<5:
        for row in range(num):
            new_arr[row][i] = normal_arr[row][col]
        i = i+1
x1 = np.array(new_arr)

kf = KFold(n_splits=10)
#mlp = MLPRegressor(hidden_layer_sizes=(8, 8, 1,), max_iter=500,)
#mlp = MLPRegressor(hidden_layer_sizes=(8, 8, 1,), learning_rate_init=0.1, max_iter=100, momentum=0.1,)
mlp = MLPRegressor(hidden_layer_sizes=(8, 9, 1), alpha=0.5, learning_rate_init=0.05, max_iter=2000, momentum=0.6,)
sum_mean1 = 0
sum_var1 = 0

for train_index, test_index, in kf.split(x1):
    x_train, x_test = x[train_index], x[test_index]
    y_train, y_test = y[train_index], y[test_index]

    mlp.fit(x_train, y_train)
    predicted = mlp.predict(x_test)
    sum_mean1 = sum_mean1 + mean_squared_error(predicted, y_test)
    sum_var1 = sum_var1 + statistics.stdev(y_test, mean_squared_error(predicted, y_test))
sum_mean1 = math.sqrt(sum_mean1/10)
sum_var1 = sum_var1/10
sum_var1 = math.sqrt(sum_var1)

print(sum_mean1)
print(sum_mean1/sum_var1)

#gia 6 erwthma
print(sum_mean1)
print(sum_mean1/sum_var1)
