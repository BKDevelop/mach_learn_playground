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
print('Importing 2012 NFL season play by play data...\n')
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
        #pointDiff = int(row[10]) - int(row[11])
        # use roughly 80% of data to train / 20% of data to test 
        if(random.random() < .80):
            gameSituations_train.append([down, distance, spot, time])
            playCalls_train.append(play)
        else:
            gameSituations_test.append([down, distance, spot, time])
            playCalls_test.append(play)
            
f.close()
            
# train models
print("Training Decision Tree...")
treeModel = treeModel.fit(gameSituations_train, playCalls_train)
print("Training Linear SVC...")
linearSVC = linearSVC.fit(gameSituations_train, playCalls_train)
print("Training MLP Neural Network...")
neuralNetwork = neuralNetwork.fit(gameSituations_train, playCalls_train)

# test models
testCounter = 0
treeModelHits = 0
linearSVCHits = 0
neuralNetworkHits = 0
for situation in gameSituations_test:
    if(playCalls_test[testCounter] == treeModel.predict([situation])):
        treeModelHits += 1
    if(playCalls_test[testCounter] == linearSVC.predict([situation])):
        linearSVCHits += 1
    if(playCalls_test[testCounter] == neuralNetwork.predict([situation])):
        neuralNetworkHits += 1
    
    testCounter += 1

# Output Results

print("\nModels and respective percentage of correct predictions out of ", testCounter + 1, " tested game situations:\n")
print("Decision Tree: ", "%.2f" % (treeModelHits * 100 / (testCounter + 1)), " %")
print("Linear SVC: ", "%.2f" % (linearSVCHits * 100 / (testCounter + 1)), " %")
print("MLP Neural Network: ", "%.2f" % (neuralNetworkHits * 100 / (testCounter + 1)), " %")