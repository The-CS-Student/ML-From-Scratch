import numpy
import random
X = [1,2,3,4,5,6,7,8,9,10]
Y = [0,0,0,0,1,1,1,1,1,1]
w = random.random()
b = random.random()
iterations = 1000
learningRate = 0.1
def prediction(x,w,b):
	linear = w*x+b
	return 1/(1+numpy.exp(-linear))
def crossEntropy(y,predicted):
	return -(y*numpy.log(predicted)+(1-y)*numpy.log(1-predicted))
def backPropagate(x,y,predicted,w,b):
	w-=x*(predicted-y)
	b-=predicted-y
	return w,b
def train(X,Y,w,b):
	for i in range(iterations):
		print("Iterations : "+str(i))
		for j in range(len(X)):

			predicted = prediction(X[j],w,b)
			error = crossEntropy(Y[j],predicted)
			w,b = backPropagate(X[j],Y[j],predicted,w,b)
			print(str(j)+"Error : "+str(error))
	print(w,b)
train(X,Y,w,b)