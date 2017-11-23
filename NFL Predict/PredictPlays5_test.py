import csv
from sklearn import tree, svm
from sklearn.neural_network import MLPClassifier
import random

gameSituations_train = []
playCalls_train = []

gameSituations_test = []
playCalls_test = []


# initialize different models
treeModel = tree.DecisionTreeClassifier()
linearSVC = svm.LinearSVC()
neuralNetwork = MLPClassifier()


# get data for all downs
with open("2012_nfl_pbp_data_rp.csv") as f:
    reader = csv.reader(f)
    header = next(reader) # Jump the header
    for row in reader:    
        # row = gameid, qtr, min, sec, off, def, down, togo, ydline, description, offscore, defscore, season,,,
    
        down = int(row[6])
        distance = int(row[7])
        spot = int(row[8])
        play = str(row[9])
        time = 60.0 - ( (int(row[2])) - (1.0 - int(row[3])/60))
        pointDiff = int(row[10]) - int(row[11])
        # use roughly 80% of data to train / 20% of data to test 
        if(random.random() < .80):
            gameSituations_train.append([down, distance, spot, time, pointDiff])
            playCalls_train.append(play)
        else:
            gameSituations_test.append([down, distance, spot, time, pointDiff])
            playCalls_test.append(play)
            
f.close()
            
# train models
treeModel = treeModel.fit(gameSituations_train, playCalls_train)
linearSVC = linearSVC.fit(gameSituations_train, playCalls_train)
neuralNetwork = neuralNetwork.fit(gameSituations_train, playCalls_train)

# test decision tree
counter = 0
hits = 0
for situation in gameSituations_test:
    if(playCalls_test[counter] == treeModel.predict([situation])):
        hits += 1
    
    counter += 1
    
print(hits/counter)
print(counter)

# test linearSVC
counter = 0
hits = 0
for situation in gameSituations_test:
    if(playCalls_test[counter] == linearSVC.predict([situation])):
        hits += 1
    
    counter += 1
    
print(hits/counter)
print(counter)

# test neural Network
counter = 0
hits = 0
for situation in gameSituations_test:
    if(playCalls_test[counter] == neuralNetwork.predict([situation])):
        hits += 1
    
    counter += 1
    
print(hits/counter)
print(counter)