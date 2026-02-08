# additional prep:
# for i in $(seq 1 51000); do ls $PWD/Test/SumStats/$i.ss.txt; done > Test/test_ss.txt
# for i in $(seq 1 51000); do ls $PWD/Test/Targets/1/$i.target.npy; done > Test/test_targets.txt
# for i in $(seq 1 51000); do ls $PWD/Train/SumStats/$i.ss.txt; done > Train/train_ss.txt
# for i in $(seq 1 51000); do ls $PWD/Train/Targets/1/$i.target.npy; done > Train/train_targets.txt

import numpy as np
import sys
from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error, r2_score
from matplotlib import pyplot as plt

max_cutoff = 99999999  # for testing

### load data
print("loading data 1")
fp = sys.argv[1]  # path to linkedNN output directory containing "Train"/", "Test/" etc.

## stats
stats = np.load(fp + "preprocess_params.npy", allow_pickle=True)
Y_mean = stats[2]
Y_sd = stats[3]
train_ss = []
counter = 0
with open(fp + "/Train/train_ss.txt") as infile:
    for line in infile:
        counter += 1
        if counter == max_cutoff:
            break
        print("loading data 1", counter)        
        with open(line.strip()) as infile2:
            line2 = infile2.readline()
            newline = np.array(list(map(float,line2.strip().split(","))))
            train_ss.append(newline)
print("loading data 2")
test_ss = []
counter = 0
with open(fp + "/Test/test_ss.txt") as infile:
    for line in infile:
        counter += 1
        if counter == max_cutoff:
            break
        print("loading data 2", counter)
        with open(line.strip()) as infile2:
            line2 = infile2.readline()
            newline = np.array(list(map(float,line2.strip().split(","))))
            test_ss.append(newline)
print("loading data 3")
train_targets = []
counter = 0
with open(fp + "/Train/train_targets.txt") as infile:
    for line in infile:
        counter += 1
        if counter == max_cutoff:
            break
        print("loading data 3", counter)
        newline = np.load(line.strip())
        train_targets.append(newline)
print("loading data 4")
test_targets = []
counter = 0
with open(fp + "/Test/test_targets.txt") as infile:
    for line in infile:
        counter += 1
        if counter == max_cutoff:
            break
        print("loading data 4", counter)
        newline = np.load(line.strip())
        test_targets.append(newline)
#
train_ss = np.array(train_ss)
test_ss = np.array(test_ss)
train_targets = np.array(train_targets)
test_targets = np.array(test_targets)
unnorm_test_targets = np.exp(test_targets*Y_sd + Y_mean)



### normalizing
print("normalizing")
X_mean = np.mean(train_ss, axis=0)
X_sd = np.std(train_ss, axis=0, ddof=1)
train_ss = (train_ss-X_mean) / X_sd
test_ss = (test_ss-X_mean) / X_sd



### RF
print("running RF")
model = RandomForestRegressor()  
model.fit(train_ss, train_targets)
y_pred = model.predict(test_ss)
unnorm_y_pred = np.exp(y_pred*Y_sd + Y_mean)
if len(y_pred.shape) == 1:  # adjusting dims for single target
    y_pred = np.expand_dims(y_pred, -1)
for target in range(y_pred.shape[-1]):
    err = np.sqrt(mean_squared_error(unnorm_test_targets[:,target], unnorm_y_pred[:,target]))
    r2 = r2_score(test_targets[:,target], y_pred[:,target])
    mrae = np.mean(np.abs((unnorm_test_targets[:,target] - unnorm_y_pred[:,target]) / unnorm_test_targets[:,target]))
    print("\tTarget #", target)
    print("RMSE:", err)
    print("R²:", r2)
    print("MRAE:", mrae)
    print()


### NN
print()
print("running MLP")
model = MLPRegressor()
model.fit(train_ss, train_targets)
y_pred = model.predict(test_ss)
unnorm_y_pred = np.exp(y_pred*Y_sd + Y_mean)
if len(y_pred.shape) == 1:
    y_pred = np.expand_dims(y_pred, -1)
for target in range(y_pred.shape[-1]):    
    err = np.sqrt(mean_squared_error(unnorm_test_targets[:,target], unnorm_y_pred[:,target]))
    r2 = r2_score(test_targets[:,target], y_pred[:,target])
    mrae = np.mean(np.abs((unnorm_test_targets[:,target] - unnorm_y_pred[:,target]) / unnorm_test_targets[:,target]))
    print("\n\tTarget #", target)
    print("RMSE:", err)
    print("R²:", r2)
    print("MRAE:", mrae)
    print()

