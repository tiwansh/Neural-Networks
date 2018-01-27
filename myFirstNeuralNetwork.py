#my first neural network 
#Networks Training gone wild
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import random
#picking an architecture
warnings.filterwarnings("ignore")

#function to calculate sigmoid of the i/p layer
def sigmoid(ipFrame):
	retFrame = (1 / (1 + np.exp(-ipFrame)))
	return retFrame

def derivative_sigmoid(ipFrame):
	return (ipFrame) * (1 - (ipFrame))
#titanic dataset analyse

input_url = 'train.csv'
dataFrame = pd.read_csv(input_url, header = 0)

#print dataFrame.shape
############ZZZZZZZZZZIIIIIIIIIIIDDDDDDDDDDDDDDD
#plt.plot(dataFrame.Age)
#plt.show()

#------------Input units defined---------------------#

#Pclass
#Sex
#Age
#Fare

inputLayer = dataFrame.drop(['PassengerId', 'Survived', 'Name', 'SibSp', 'Parch', 'Embarked', 'Cabin', 'Ticket'], axis = 1)
inputLayer.Sex[inputLayer.Sex == 'male'] = (float)(0)
inputLayer.Sex[inputLayer.Sex == 'female'] = (float)(1)

inputLayer.Fare = inputLayer.Fare.fillna(0)
inputLayer.Age = inputLayer.Age.fillna(0)


inputLayer.Age = inputLayer.Age.astype(float)
inputLayer.Sex = inputLayer.Sex.astype(float)
inputLayer.insert(0,'Bias',1);
#print inputLayer

#------------inputLayer dataframe has 4 i/p units which will be the i/p of neural n/w-----------#

#print 'pehla : '
#print type(inputLayer.Age[10])
#print type(inputLayer.Pclass[10])
#print type(inputLayer.Sex[10])
#print type(inputLayer.Fare[10])



operated = sigmoid(inputLayer)
inputLayerArray = operated.as_matrix()
#print operated

#---------------input Layer created and typecasted and sigmoided---------------#

#------------adding bias unit to input layer initially----------#





#-------------- Creating output layer yy ----------#

outputLayer = dataFrame.drop(['PassengerId', 'Name', 'SibSp', 'Parch', 'Embarked', 'Cabin', 'Ticket', 'Age', 'Pclass', 'Sex', 'Fare'], axis = 1)
outputLayerArray = outputLayer.as_matrix()
#print outputLayer

#-----------Output layer created---------------#

#forward feed ka code peo and then backpropogation kar ke correct kar do0

#-------------------------------------Creating all the params needed for forward feed and backprop--------------------#

theta1 = np.random.random((5, 5))
theta2 = np.random.random((5, 1)) 

print theta1
print theta2
#Converting theta1 and theta2 into 1D series
#theta1 = map(lambda x: x[0], theta1)
#theta2 = map(lambda x: x[0], theta2)
#print (theta1) 
#print theta2

#theta1_gradient = np.zeros((4, 1))
#theta2_gradient = np.zeros((4, 1))
#print type(theta1_gradient)
#print type(theta2_gradient)
#theta1_gradient = map(lambda x: x[0], theta1_gradient)
#theta2_gradient = map(lambda x: x[0], theta2_gradient)

#theta1 = theta1.as_matrix()

#-------------------------------------All the required params for forward feed and backprop created-----------------------------#
a1 = inputLayerArray
lr = 0.1

for i in range(0, 60000):
	z2 = inputLayerArray.dot(theta1)
	#print "load"
	#print len(z2)
	#print len(z2[0])

	#-------add bias unit here----#
	a2 = sigmoid(z2)
	#print a2

	z3 = a2.dot(theta2)
	#print len(z3)
	#print len(z3[0])

	yy = sigmoid(z3)
	#print a3
	#if i == 500:
	#	print a3
	#	print outputLayerArray
	#	dede = raw_input("do ")

	#error at output layer
	error = outputLayerArray - yy
	#print del3
	sol = derivative_sigmoid(outputLayerArray)
	shl = derivative_sigmoid(a2)

	#delta at output layer
	delta = error * sol 

	#error at hidden layer
	error_hidden = delta.dot(theta2.T)
	#delta at hidden layer
	delta_hidden = error_hidden * shl

	theta2 += a2.T.dot(delta) * lr 
	theta1 += inputLayerArray.T.dot(delta_hidden) * lr

	#del2 = (np.dot(del3,theta2.T) * (a2 * (1 - a2)))


	#theta2 += a2.T.dot(del3) * lr
	#theta1 += a1.T.dot(del2) * lr
	#print len(del2)
	#print len(del2[0])

#print "Printing Theta1 "
#print theta1
#print "Printing Theta2"
#print theta2

z2 = a1.dot(theta1)
a2 = sigmoid(z2)
z3 = a2.dot(theta2)
a3 = sigmoid(z3)
#a3 = a3.round()
print a3

#-----------No on the basis of this trained theta1 and theta2-----------try to learn for the new dataset-----------#
'''
testDataFrame = pd.read_csv('test.csv', header = 0)
#print testDataFrame
passId = testDataFrame.PassengerId
testDataFrame = testDataFrame.drop(['PassengerId', 'Name', 'SibSp', 'Parch', 'Embarked', 'Cabin', 'Ticket'], axis = 1)
testDataFrame.Sex[testDataFrame.Sex == 'male'] = (float)(0)
testDataFrame.Sex[testDataFrame.Sex == 'female'] = (float)(1)

testDataFrame.Fare = testDataFrame.Fare.fillna(0)
testDataFrame.Age = testDataFrame.Age.fillna(0)

testDataFrame.Age = inputLayer.Age.astype(float)
testDataFrame.Sex = inputLayer.Sex.astype(float)
testDataFrame.insert(0,'Bias',1)
testLayerInput = testDataFrame.as_matrix()
#print testLayerInput
z2 = np.dot(testLayerInput, theta1)
a2 = (z2)
z3 = np.dot(a2, theta2)
a3 = (z3)




predFrame = pd.DataFrame(data = a3, columns= ['Survived'])
PassengerIdFrame= pd.DataFrame(data = passId, columns = ['PassengerId'])
#predFrame.Survived = predFrame.Survived.round()

PassengerIdFrame = PassengerIdFrame.assign(Survived = predFrame.Survived)

print PassengerIdFrame
PassengerIdFrame.to_csv('solution.csv', encoding= 'utf-8', index = False)'''

#------------------------------------------------#

#My Intuition














'''To add - 
1. Bias units - Check'''	