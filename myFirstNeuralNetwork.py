#neuralgonecyazy

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
def sigmoid (x):
	return 1/(1 + np.exp(-x))


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
#print inputLayer

#------------inputLayer dataframe has 4 i/p units which will be the i/p of neural n/w-----------#

#print 'pehla : '
#print type(inputLayer.Age[10])
#print type(inputLayer.Pclass[10])
#print type(inputLayer.Sex[10])
#print type(inputLayer.Fare[10])


operated = (inputLayer - inputLayer.mean()) / (inputLayer.max() - inputLayer.min())
operated = sigmoid(operated)
#operated.insert(0,'Bias',1);
inputLayerArray = operated.as_matrix()
#print "lolmax"
#print inputLayerArray
#print inputLayerArray.shape[1]
#---------------input Layer created and typecasted and sigmoided---------------#

#------------adding bias unit to input layer initially----------#





#-------------- Creating output layer yy ----------#

outputLayer = dataFrame.drop(['PassengerId', 'Name', 'SibSp', 'Parch', 'Embarked', 'Cabin', 'Ticket', 'Age', 'Pclass', 'Sex', 'Fare'], axis = 1)
outputLayerArray = outputLayer.as_matrix()
#print outputLayer

#-----------Output layer created---------------#

#forward feed ka code peo and then backpropogation kar ke correct kar do0

#-------------------------------------Creating all the params needed for forward feed and backprop--------------------#

theta1 = np.random.uniform(size=(5, 5))
theta2 = np.random.uniform(size=(5, 1)) 

#print theta1
#print theta2
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

inputLayer_neurons = inputLayerArray.shape[1]
hiddenLayer_neurons = 4 #inputLayerArray.shape[1]
outputLayer_neurons= 1

wh = np.random.uniform(size=(inputLayer_neurons, hiddenLayer_neurons))
bh = np.random.uniform(size=(1,hiddenLayer_neurons))

wout = np.random.uniform(size= (inputLayer_neurons,outputLayer_neurons))
bout = np.random.uniform(size = (1,outputLayer_neurons))

#-------------------------------------All the required params for forward feed and backprop created-----------------------------#
X = inputLayerArray
lr = 0.03

#Input array
#X=np.array([[1,0,1,0],[1,0,1,1],[0,1,0,1]])

#Output
#y=np.array([[1],[1],[0]])

#inputlayer_neurons = X.shape[1] #number of features in data set
#hiddenlayer_neurons = 3 #number of hidden layers neurons
#output_neurons = 1 #number of neurons at output layer

#weight and bias initialization
'''wh=np.random.uniform(size=(inputlayer_neurons,hiddenlayer_neurons))
bh=np.random.uniform(size=(1,hiddenlayer_neurons))
wout=np.random.uniform(size=(hiddenlayer_neurons,output_neurons))
bout=np.random.uniform(size=(1,output_neurons))
'''

for i in range(0, 900000):
	hli = X.dot(wh) + bh
	hla = sigmoid(hli)

	oli = hla.dot(wout) + bout
	out = sigmoid(oli)

	E = outputLayerArray - out

	sol = derivative_sigmoid(out)
	shl = derivative_sigmoid(hla)

	d_out = E * sol
	
	E_hl = d_out.dot(wout.T)
	d_hl = E_hl * sol

	wout += hla.T.dot(d_out) * lr
	bout += np.sum(d_out, axis = 0, keepdims = True) * lr
	wh += X.T.dot(d_hl) * lr
	bh += np.sum(d_hl) * lr


#print out
#print "Printing Theta1 "
#print theta1
#print "Printing Theta2"
#print theta2

z2 = X.dot(wh) + bh
a2 = sigmoid(z2)
z3 = a2.dot(wout) + bout
a3 = sigmoid(z3)
a3 = a3.round()
#print a3

#compare a3 with survived
survived_dataFrame = pd.DataFrame(data = dataFrame.Survived,columns=['Survived'])
calc_dataFrame = pd.DataFrame(data = a3, columns = ['MySurvived'])
survived_dataFrame = survived_dataFrame.assign(MySurvived = calc_dataFrame.MySurvived)
survived_dataFrame['match'] = np.where(survived_dataFrame['Survived'] == survived_dataFrame['MySurvived'],1,0)
correct = survived_dataFrame['match'] == 1

print survived_dataFrame
print correct
lol = correct.value_counts()
print (float)(lol[1].item() / (float)(lol[1].item() + lol[0].item())) * 100

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