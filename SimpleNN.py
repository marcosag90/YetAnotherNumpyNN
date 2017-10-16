import numpy as np

"""
For a neural network with an input layer with size 10, a hidden layer of size 50, and an output layer with size 3, we will need:
10*50+50*3 = 650 weights
53 biasses.
"""

class Neural_Net(object):
    def __init__(self, input_nodes, hidden_nodes, output_nodes, learning_rate):
        self.input_nodes = input_nodes
        self.hidden_nodes = hidden_nodes
        self.output_nodes = output_nodes
        self.lr = learning_rate

        self.hidden_out = np.zeros(hidden_nodes)
        self.out = np.zeros(output_nodes)

        self.Wi2h = np.random.normal(0.0, self.input_nodes**-0.5, (self.input_nodes, self.hidden_nodes))
        self.Wh2o = np.random.normal(0.0, self.hidden_nodes**-0.5, (self.hidden_nodes, self.output_nodes))
        
        self.activation_function = lambda x: np.maximum(x, 0, x)
        self.softmax = lambda x: np.exp(x)*(np.sum(np.exp(x)))**-1

        self.delta_weights_i_h = np.zeros(self.Wi2h.shape)
        self.delta_weights_h_o = np.zeros(self.Wh2o.shape)

        
    def forward_pass(self, x):
        def input_to_hidden( x, W):
            hidden_output = self.activation_function(np.dot(x, W))
            return hidden_output
        def hidden_to_output( W, x_h):
            return self.softmax(np.dot(x_h, W))
            
        self.hidden_out = input_to_hidden(x, self.Wi2h)
        self.out = hidden_to_output(self.Wh2o, self.hidden_out)
        return self.out
        
    def cross_entropy_loss(self, Targets, Output):
        return -np.sum(np.dot(Targets, np.log(Output)))
        
    def backward_pass(self, t, X):    
        output_error_term = t-self.out    #Softmax derivative = p - y
        hidden_error =np.dot(self.Wh2o, t.transpose())
        hidden_error_term = hidden_error*np.maximum(0, self.hidden_out) #Relu's derivative is 1 if x>0. In this case, with the chain function, the result is 
        # hidden_error>0? 1*hidden_error : 0     Whish is the relu function itself
        # Weight step (input to hidden)    
        self.delta_weights_i_h += self.lr*hidden_error_term*X[:, None]
        self.delta_weights_h_o += self.lr*output_error_term*self.hidden_out[:, None] 

        self.Wh2o += self.delta_weights_h_o
        self.Wi2h += self.delta_weights_i_h 


NN = Neural_Net(10, 50, 3, 0.01)
x=np.array([0.5, 0.6, 0.1, 0.25, 0.33, 0.9, 0.88, 0.76, 0.69, 0.95])
print("Training...")
for i in range(0, 5):
    print("//-----------------Iteration ", i, "-----------------//")
    O = NN.forward_pass(x)
    print("Output :", O)
    L = NN.cross_entropy_loss(np.array([1, 0, 0]), O)
    print("Loss: ", L)
    NN.backward_pass(np.array([1, 0, 0]), x)


