
# This implements a simple version of a k-nearest neighbour model for k=1
# iris_train.tsv is a tab-separated file with training data to predict a species of iris. 
# The first column is petal length in mm; the second column is petal width in mm; the third column is the species.
# Each row is data from a separate flower.
# iris_test.tsv is an independent test set.

import sys

# (1) a function distance_euc(v1, v2) which computes the (squared) Euclidean distance between two sequences (Euclidean distance is the sum of the squared element-wise differences).
def distance_euc(v1, v2):
    assert len(v1) == len(v2),  "vectors wrong length"
    d = 0.0
    for i in range(len(v1)):
        d = d + (v1[i]-v2[i])**2
    return d


# (2) A function knn_train(m, x_train, y_train) where m is a list data structure for the knn model, and where x is the training data explanatory variables and y is the training data labels. m is passed by reference and is modified to store x_train and y_train.
def knn_train(m, x_train, y_train):
    m.append(x_train)
    m.append(y_train)

# (3) A function knn_predict(m, x_test) where x_test is a sequence of explanatory variables, which returns a sequence of predictions, one for each row of data.
#def knn_predict(m, x_test):
    # create an empty list to store the predictions
    #predictions = []
    # for each datapoint in x_test
    #for i in range(len(x_test)):
        #import sys
        # maximum representable positive finite float
        # essentially this is setting best_match to a very large number to start, so 
        # later on anything less than this number will replace it (d < best_match)
        #best_match = sys.float_info.max..
        #best_match_index = -1
        # for each point in the first list of m (which is the training data points/values)
        #for j in range(len(m[0])):
            # calculate the euclidean distance betweeen that test explanatory variable, and each 
            # test variable 
            #d = distance_euc(m[0][j], x_test[i])
            # Need to change this - need to assign species label of 3 closest neighbors.. 
            #right now, it is going through each training set and updating the best match ..
            # if d is less than best_match (above)
            #if d < best_match:
                # update best match 
                #best_match = d
                # update the index with the index of the training variable in m that was closest to the test variable
                #best_match_index = j
        # in list m, fetch the "best match indexes" from the second list within m (m[1]), which contains all 
        # the species names for the corresponding training data (m[0])
        # append these to the predictions list 
        #predictions.append(m[1][best_match_index])
    #return predictions

# (4) Define the data structure for the model and use these functions to train the model on the data in "iris_train.tsv", then predict the labels for the test data in "iris_test.tsv" and print them.

m = []  # for our model we will use a two-element list

# train model with training data, then predict labels of independent test set, and print predictions
with open("iris_train.tsv", "r") as f:
    x_train = [];
    y_train = [];
    data = f.readlines()
    for i in range(len(data)):
        s = data[i].strip().split('\t')
        y_train.append(s[-1])
        x_train.append(s[:-1])
        x_train[i] = [float(i) for i in x_train[i] ]

with open("iris_test.tsv", "r") as f:
    x_test = [];
    data = f.readlines()
    for i in range(len(data)):
        x_test.append(data[i].strip().split('\t'))
        x_test[i] = [float(i) for i in x_test[i] ]

knn_train(m, x_train, y_train)
p = knn_predict(m, x_test)
print(p)
print("\n")

# (5) A custom class called "knn_class" which can be used in the following way:
# a = knn_class(x_train, y_train)
# pred = a.predict(x_test)

class knn_class:
    def __init__(self, x_train, y_train):
        self.m = []
        self.m.append(x_train)
        self.m.append(y_train)

    def predict(self, x_test):
        return knn_predict(self.m, x_test)

a = knn_class(x_train, y_train)
pred = a.predict(x_test)
print("class-based:")
print(pred)


