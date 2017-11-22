import csv
from sklearn import tree

downDistanceSpot = []
playCalls = []

playCaller = tree.DecisionTreeClassifier()


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
            
            downDistanceSpot.append([down, distance, spot])
            playCalls.append(play)
            
f.close()
            
# train decision tree

playCaller = playCaller.fit(downDistanceSpot, playCalls)

# Ask User for Input and give prediction
# Note: No checks for user input (e.g. spot of ball inbounds) implemented yet!
while (True):
    print("Give details over game situation: down(1-3), distance(1-99), spot of ball(1-99) (yards to go)")
    print("End program with x")
    inString = input().split(",")
    if(inString[0] == "x"):
        break
    situation = [int(x.strip()) for x in inString]

    prediction = playCaller.predict([situation])
    print("The next play should be a: ")
    print(prediction)

print("Goodbye!")