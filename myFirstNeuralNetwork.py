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
	retFrame = 1 / (1 + np.exp(ipFrame))
	return retFrame

#titanic dataset analyse

input_url = 'train.csv'
dataFrame = pd.read_csv(input_url, header = 0)

#print dataFrame.shape

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

#print inputLayer

#------------inputLayer dataframe has 4 i/p units which will be the i/p of neural n/w-----------#

print 'pehla : '
print type(inputLayer.Age[10])
print type(inputLayer.Pclass[10])
print type(inputLayer.Sex[10])
print type(inputLayer.Fare[10])



operated = sigmoid(inputLayer)
#print operated

#---------------input Layer created and typecasted and sigmoided---------------#

#-------------- Creating output layer yy ----------#

outputLayer = dataFrame.drop(['PassengerId', 'Name', 'SibSp', 'Parch', 'Embarked', 'Cabin', 'Ticket', 'Age', 'Pclass', 'Sex', 'Fare'], axis = 1)

print outputLayer

#-----------Output layer created---------------#

#forward feed ka code peo and then backpropogation kar ke correct kar do0

#-------------------------------------Creating all the params needed for forward feed and backprop--------------------#

theta1 = np.random.random((4, 1))
theta2 = np.random.random((4, 1)) 

#Converting theta1 and theta2 into 1D series
theta1 = map(lambda x: x[0], theta1)
theta2 = map(lambda x: x[0], theta2)
#print (theta1) 
#print theta2

theta1_gradient = np.zeros((4, 1))
theta2_gradient = np.zeros((4, 1))
#print type(theta1_gradient)
#print type(theta2_gradient)
theta1_gradient = map(lambda x: x[0], theta1_gradient)
theta2_gradient = map(lambda x: x[0], theta2_gradient)

#-------------------------------------All the required params for forward feed and backprop created-----------------------------#

for i in range(0, 891):
	a1 = operated.loc[i]
	z2 = theta1 * a1
	#print "load"
	#print z2
	a2 = sigmoid(z2)
	#print a2

	z3 = theta2 * a2
	#print z3

	a3 = sigmoid(z3)
	#print a3
	print outputLayer.loc[i]
	del3 = a3 - outputLayer.loc[i] 
	#print del3
	#k = raw_input("dede")
	#del2 = (np.transpose(theta2) * del3)
	#print del2



























'''To add - 
1. Bias units'''	