# NFL Predict
## Overview


## Test cases
In general, files ending with '_test.py' can be executed to train and test several methods (80/20 data split). Files without said ending promt the user to feed in details of a game situation and predicts a play call. 
### 1
This program takes roughly 80% of the data, taking into account

* Number of down
* Distance to first down marker
* Field position (Yards until opponent goal line)

Roughly 20% of the data is used to test and compare predictions of three different models:

* Decision Tree
* Linear SVC
* MLP Neural Network

### 2
Same as 1, but takes playing time into account
### 3
Same as 2, but also predict 4th down calls (Field Goal, Punt)
### 4
Same as 1, but takes point differential in account
### 5
Same as 4, but also predicts 4th down calls (Field Goal, Punt)
