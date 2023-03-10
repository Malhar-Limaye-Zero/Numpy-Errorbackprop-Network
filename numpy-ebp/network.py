import numpy as np
class NN:
    def __init__(self,x,y,LR,epochs):
        #np.random.seed(129)
        self.input = x
        self.w = np.random.rand(self.input.shape[1],6)
        self.v = np.random.rand(6,1)
        self.y = y
        self.output = np.zeros(self.y.shape)
        self.learning_rate = LR
        self.epochs = epochs
        self.costlist = []

    @staticmethod
    def Sigmoid(x):
        return 1.0/(1.0+np.exp(-x))

    @staticmethod
    def Sigmoid_derivative(x):
        return x*(1-x)
    
    @staticmethod
    def cost(t,y):
        return 0.5*np.sum(np.square(np.subtract(t,y)))
    def forward(self):
        self.Z = self.Sigmoid(np.dot(self.input,self.w))
        self.output = self.Sigmoid(np.dot(self.Z,self.v))
    
    def backprop(self):
        dv = np.dot(self.Z.T,((self.y-self.output)*self.Sigmoid_derivative(self.output)))
        dw = np.dot(self.input.T,  (np.dot((self.y - self.output) * self.Sigmoid_derivative(self.output), self.v.T) * self.Sigmoid_derivative(self.Z)))
        self.w+=dw*self.learning_rate
        self.v+=dv*self.learning_rate
    
    def train(self):
        for i in range(self.epochs):
            print("forward step of {} epoch".format(i))
            self.forward()
            print("back step of {} epoch".format(i))
            self.backprop()
            self.costlist.append(self.cost(self.y,self.output))
    
    def predict(self,data):
        self.input = data
        self.forward()
        return self.output
    