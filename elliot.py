# This is a simple NN that takes stock price data and tries to perdict next day stock price change. 
# Output layer is 12 nodes, each a different magnitude of price change. 

import datetime as dt
import matplotlib.pyplot as plt
from matplotlib import style
import numpy as np 
import pandas as pd 
import pandas_datareader.data as web 
from scipy.optimize import fmin_l_bfgs_b

# Data collection class. Creates spread sheets and percent changes of stock price 
class GetInput:
    def __init__(self, company, perdict):
        self.end = dt.date(dt.date.today().year, dt.date.today().month-1, dt.date.today().day)
        self.start = dt.date(dt.date.today().year-1, dt.date.today().month-1, dt.date.today().day-6)
        self.company = company
        self.company_size = len(company) 
        self.perdict = perdict
    
    def get_data(self, company):
        if self.perdict == True:
            start = dt.date(dt.date.today().year-1, dt.date.today().month, dt.date.today().day-6)
            end = dt.date(dt.date.today().year, dt.date.today().month, dt.date.today().day)
            return (web.DataReader(company, 'yahoo', start, end))
        else: 
            return (web.DataReader(company, 'yahoo', self.start, self.end))
    
    def per_change(self):
        i = 0
        while i < self.company_size:
            df = self.get_data(self.company[i])
            j = 0 
            percent = [1]
            while j < df.shape[0]-1:
                percent.append((df.Close[j+1]-df.Close[j])/df.Close[j])
                j += 1
            df['Perc Chg'] = percent
            if i == 0: 
                percent_grid = np.asmatrix(df.values)[:, 6]
            else: 
                percent_grid = np.concatenate((percent_grid, np.asmatrix(df.values)[:, 6]), axis=1)
            i += 1
        return(percent_grid)

# Simple class that handles data calling and saving
class Rob:
    def __init__(self, company_list, pred):
        self.company_list = company_list
        self.pred = pred 
    
    def build_set(self):
        test = GetInput(self.company_list, self.pred)
        if self.pred == True: 
            np.save("pred", test.per_change())
        else: 
            np.save("test", test.per_change())
        return  

# Main section of code, trains the NN. Handles weight training and saving trained weights for use later in the code. 
# Function "perdict" is called when weights are trained and a perdiction is to be made 
class Train: 
    def __init__(self, dataframe):
        self.dataset = np.asmatrix(dataframe)
        self.inputsize = np.asmatrix(dataframe).shape[0]-5
        self.nodes = 25
        self.outputsize = 12 
        self.connections = (self.inputsize*(self.nodes)) + ((self.nodes+1)*self.outputsize)
        self.m = np.asmatrix(dataframe).shape[1]
        self.y = 0

    def gen_theta(self):
        espilone = np.sqrt(6)/np.sqrt(self.inputsize + self.nodes)
        espiltwo = np.sqrt(6)/np.sqrt((self.nodes + 1) + self.outputsize)
        theta_one = (np.random.uniform(size=(self.inputsize, self.nodes)) * 2 * espilone - espilone)
        np.save("thetaone", theta_one)
        theta_two = (np.random.uniform(size=((self.nodes + 1), self.outputsize)) * 2 * espiltwo - espiltwo)
        np.save("thetatwo", theta_two)
        return 

    def sigmoid(self, input):
        return (1/(1+np.exp(-1*input)))

    def sigmoid_grad(self, input):
        return np.multiply((1/(1+np.exp(-input))), (1-1/(1+np.exp(-1*input))))

    def y_new(self): 
        y = sum(self.dataset[self.dataset.shape[0]-5:self.dataset.shape[0],:])*100
        index = y.shape[1]
        i = 0 
        y_new = np.matrix(np.zeros(12))
        while i < index:  
            if y[0, i] < -10:
                y_new = np.concatenate((y_new, np.matrix([1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])), axis=0)
            elif y[0, i] > -10 and y[0, i] < -8: 
                y_new = np.concatenate((y_new, np.matrix([0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])), axis=0)
            elif y[0, i] > -8 and y[0, i] < -6:
                y_new = np.concatenate((y_new, np.matrix([0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0])), axis=0)
            elif y[0, i] > -6 and y[0, i] < -4: 
                y_new = np.concatenate((y_new, np.matrix([0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0])), axis=0)
            elif y[0, i] > -4 and y[0, i] < -2: 
                y_new = np.concatenate((y_new, np.matrix([0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0])), axis=0)
            elif y[0, i] > -2 and y[0, i] < 0: 
                y_new = np.concatenate((y_new, np.matrix([0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0])), axis=0)
            elif y[0, i] > 0 and y[0, i] < 2:
                y_new = np.concatenate((y_new, np.matrix([0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0])), axis=0)
            elif y[0, i] > 2 and y[0, i] < 4: 
                y_new = np.concatenate((y_new, np.matrix([0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0])), axis=0)
            elif y[0, i] > 4 and y[0, i] < 6: 
                y_new = np.concatenate((y_new, np.matrix([0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0])), axis=0)
            elif y[0, i] > 6 and y[0, i] < 8: 
                y_new = np.concatenate((y_new, np.matrix([0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0])), axis=0)
            elif y[0, i] > 8 and y[0, i] < 10: 
                y_new = np.concatenate((y_new, np.matrix([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0])), axis=0)
            elif y[0, i] > 10: 
                y_new = np.concatenate((y_new, np.matrix([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1])), axis=0)
            else:
                return ("You messed up son")
            i = i + 1
        self.y = y_new[1:, :]

    def cost(self, theta): 
        y = self.y
        data_input = np.transpose(np.asmatrix(self.dataset[:self.inputsize, :]))

        m = self.m
        lamb = 1
        
        theta_one = np.matrix(np.reshape(theta[(self.nodes*self.outputsize) + self.outputsize:], (self.inputsize, self.nodes)))
        theta_two = np.matrix(np.reshape(theta[:(self.nodes*self.outputsize) + self.outputsize], (self.nodes+1, self.outputsize)))

        hidden = self.sigmoid(np.matmul(data_input, theta_one))
        hidden = np.concatenate((np.transpose(np.matrix(np.ones(m))), hidden), axis=1)
        out = self.sigmoid(np.matmul(hidden, theta_two))

        J = ((1/m) * np.matrix.sum(np.matrix.sum(np.multiply(-1*y, np.log(out)) - np.multiply((1-y), np.log(1-out)), axis = 1))) + (lamb/(2*m))*(np.matrix.sum(np.matrix.sum((np.square(theta_one[:, 1:])), axis = 1)) + np.matrix.sum(np.matrix.sum((np.square(theta_one[:, 1:])), axis = 1)))
        return J

    def grad(self, theta):
        y = self.y
        data_input = np.transpose(np.asmatrix(self.dataset[:self.inputsize, :]))
        
        m = self.m
        lamb = 1
        
        theta_one = np.matrix(np.reshape(theta[(self.nodes*self.outputsize) + self.outputsize:], (self.inputsize, self.nodes)))
        theta_two = np.matrix(np.reshape(theta[:(self.nodes*self.outputsize) + self.outputsize], (self.nodes+1, self.outputsize)))

        hidden = self.sigmoid(np.matmul(data_input, theta_one))
        hidden = np.concatenate((np.transpose(np.matrix(np.ones(m))), hidden), axis=1)
        out = self.sigmoid(np.matmul(hidden, theta_two))

        sigma3 = out - y
        sigma2 = np.multiply((np.matmul(sigma3, np.transpose(theta_two))), self.sigmoid_grad(np.concatenate((np.transpose(np.matrix(np.ones(m))), (np.matmul(data_input, theta_one))), axis=1)))
        
        delta2 = np.matmul(np.transpose(sigma3), hidden)
        delta1 = np.matmul(np.transpose(sigma2), data_input)

        preTheta2_grad = lamb/m*theta_two
        preTheta1_grad = lamb/m*theta_one

        Theta2_grad = np.transpose(np.add((delta2/m), np.transpose(np.reshape(np.insert(np.matrix.flatten(preTheta2_grad[1:, :]), 0, np.zeros(self.outputsize)), (self.nodes+1, self.outputsize)))))
        Theta1_grad = np.transpose(np.add((delta1[1:,:]/m), np.transpose(np.reshape(np.insert(np.matrix.flatten(preTheta1_grad[1:, :]), 0, np.zeros(self.nodes)), (self.inputsize, self.nodes)))))
        
        theta = np.matrix(np.zeros((1,self.connections)))
        theta[0, (self.nodes*self.outputsize + self.outputsize):] = np.matrix.flatten(Theta1_grad)
        theta[0, :(self.nodes*self.outputsize + self.outputsize)] = np.matrix.flatten(Theta2_grad)
        return theta

    def train(self, num):
        theta_one = np.asmatrix(np.load("thetaone.npy"))
        theta_two = np.asmatrix(np.load("thetatwo.npy"))
        theta = np.matrix(np.zeros((1,self.connections)))
        theta[0, :self.nodes*self.inputsize] = np.matrix.flatten(theta_one)
        theta[0, self.nodes*self.inputsize:] = np.matrix.flatten(theta_two)
        self.y_new()
        results = fmin_l_bfgs_b(self.cost, theta, fprime=self.grad, maxfun=num)
        np.save("thetapred", np.matrix(results[0]))
    
    def perdict(self):
        theta = np.load("thetapred.npy")
        dataset = np.load("pred.npy")
        m = 1

        data_input = np.transpose(np.asmatrix(dataset[:self.inputsize, :]))
        theta_one = np.matrix(np.reshape(theta[0, (self.nodes*self.outputsize) + self.outputsize:], (self.inputsize, self.nodes)))
        theta_two = np.matrix(np.reshape(theta[0, :(self.nodes*self.outputsize) + self.outputsize], (self.nodes+1, self.outputsize)))

        hidden = self.sigmoid(np.matmul(data_input, theta_one))
        hidden = np.concatenate((np.transpose(np.matrix(np.ones(m))), hidden), axis=1)
        out = self.sigmoid(np.matmul(hidden, theta_two))

        print()
        print("Actual change")
        print(sum(dataset[dataset.shape[0]-5:dataset.shape[0],:])*100)
        print("Hours of work say")
        print(out)


companylist = ["AAPL", "BLK", "CF", "DOV", "ETR", "FLT", "GPS", "HOG", "IRM", "JPM", "KIM", "LMT", "M", "NKE", "OXY", "PCAR", "QRVO", 
"RE", "SEE", "TROW", "UNM", "V", "WELL", "XLNX", "YUM", "ZTS"]
companylist = ["TSLA"]
test = Rob(companylist, False)
test.build_set()

train = Train(np.load("test.npy"))
train.gen_theta()
train.train(15)

pred = Rob(["TSLA"], True)
pred.build_set()
train.perdict()