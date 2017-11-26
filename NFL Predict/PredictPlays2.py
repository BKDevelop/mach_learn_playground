import csv
from sklearn import tree, svm
from sklearn.neural_network import MLPClassifier
import random

gameSituations_train = []
playCalls_train = []


# initialize different models
treeModel = tree.DecisionTreeClassifier()
linearSVC = svm.LinearSVC()
neuralNetwork = MLPClassifier()


# get data for 1st-3rd down calls
print('Importing 2012 NFL season play by play data...\n')
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
            time = 60.0 - ( (int(row[2])) - (1.0 - int(row[3])/60))
             
            gameSituations_train.append([down, distance, spot, time])
            playCalls_train.append(play)
            
f.close()
            
# train models
print("Training Decision Tree...")
treeModel = treeModel.fit(gameSituations_train, playCalls_train)
print("Training Linear SVC...")
linearSVC = linearSVC.fit(gameSituations_train, playCalls_train)
print("Training MLP Neural Network...")
neuralNetwork = neuralNetwork.fit(gameSituations_train, playCalls_train)

# Ask User for Input and give prediction
# Note: No checks for user input (e.g. spot of ball inbounds) implemented yet!
while (True):
    print("\nGive details over game situation: down(1-3), distance(1-99), spot of ball(1-99) (yards to go), game time (min)")
    print("End program with x")
    inString = input().split(",")
    if(inString[0] == "x"):
        break
    situation = [int(x.strip()) for x in inString]

    
    print("\nThe predictions for given game situation ", situation, " are:\n")
    print("Decision Tree: ", treeModel.predict([situation]))
    print("Linear SVC: ", treeModel.predict([situation]))
    print("MLP Neural Network: ", treeModel.predict([situation]))
    print("-----------------------------------------------")

print("\nGoodbye!")