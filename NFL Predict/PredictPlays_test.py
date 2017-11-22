import csv
from sklearn import tree, svm
from sklearn.neural_network import MLPClassifier
import random

downDistanceSpot_train = []
playCalls_train = []

downDistanceSpot_test = []
playCalls_test = []


# initialize different models
treeModel = tree.DecisionTreeClassifier()
linearSVC = svm.LinearSVC()
neuralNetwork = MLPClassifier()


# get data for 1st-3rd down calls
with open("2012_nfl_pbp_data_rp.csv") as f:
    reader = csv.reader(f)
    header = next(reader) # Jump the header
    for row in reader:    
        # row = gameid, qtr, min, sec, off, def, down, togo, ydline, description, offscore, defscore, season,,,
        if(int(row[6]) > 0 and int(row[6]) <=3):
            down = int(row[6])
            distance = int(row[7])
            spot = int(row[8])
            play = str(row[9])
            # use roughly 80% of data to train / 20% of data to test 
            if(random.random() < .80):
                downDistanceSpot_train.append([down, distance, spot])
                playCalls_train.append(play)
            else:
                downDistanceSpot_test.append([down, distance, spot])
                playCalls_test.append(play)
            
            
f.close()
            
# train models
treeModel = treeModel.fit(downDistanceSpot_train, playCalls_train)
linearSVC = linearSVC.fit(downDistanceSpot_train, playCalls_train)
neuralNetwork = neuralNetwork.fit(downDistanceSpot_train, playCalls_train)

# test decision tree
counter = 0
hits = 0
for situation in downDistanceSpot_test:
    if(playCalls_test[counter] == treeModel.predict([situation])):
        hits += 1
    
    counter += 1
    
print(hits/counter)
print(counter)

# test linearSVC
counter = 0
hits = 0
for situation in downDistanceSpot_test:
    if(playCalls_test[counter] == linearSVC.predict([situation])):
        hits += 1
    
    counter += 1
    
print(hits/counter)
print(counter)

# test neural Network
counter = 0
hits = 0
for situation in downDistanceSpot_test:
    if(playCalls_test[counter] == neuralNetwork.predict([situation])):
        hits += 1
    
    counter += 1
    
print(hits/counter)
print(counter)