# NFL Predict
## Overview
coming soon...

## Test cases
In general, files ending with '_test.py' can be executed to train and test several methods (80/20 data split). Files without said ending promt the user to feed in details of a game situation and predicts a play call. 
### 1
This program takes roughly 80% of the data, taking into account

* Number of down
* Distance to first down marker
* Field position (Yards until opponent goal line)
* Analysing 1st-3rd down calls (RUN, PASS)

Roughly 20% of the data is used to test and compare predictions of three different models:

* Decision Tree
* Linear SVC
* MLP Neural Network

### 2
Same as 1, but adds:

* playing time

### 3
Same as 1, but adds:

* playing time
* 4th down calls (Field Goal, Punt)

### 4
Same as 1, but adds:

* playing time
* point differential 

### 5

Same as 4, but adds:

* playing time
* point differential
* 4th down calls (Field Goal, Punt)

## Results
For each run 80% of the data was used to train different models, 20 % of the data was used to test the predictions. To obtain the following data, the percentages of correct prediction from 5 runs were averaged for each test case.

### 1

Decision Tree:		60,89 +/- 0,36 %
Linear SVC:		    55,78 +/- 5,81 %
Neural Network:	    61,76 +/- 0,33 %	

		

### 2
Decision Tree:		59,12 +/- 0,30 %
Linear SVC:		    57,69 +/- 4,03 %
Neural Network:	    61,50 +/- 1,50 %

### 3
Decision Tree:		61,06 +/- 0,51 %
Linear SVC:		    56,03 +/- 4,81 %
Neural Network:	    64,37 +/- 0,51 %
### 4

Decision Tree:		61,89 +/- 0,44 %
Linear SVC:	        60,60 +/- 2,88 %
Neural Network:	    64,91 +/- 1,21 %
		
### 5

Decision Tree:		64,04 +/- 0,48 %
Linear SVC:		    58,48 +/- 2,09 %
Neural Network:	    67,31 +/- 0,42 %
		