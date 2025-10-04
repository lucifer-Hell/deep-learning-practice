# This is a very outof scope and hard level stuff we can omit as derivation of it is not really necesssary 

import numpy as np
class NN :

    @staticmethod
    def generate_arr(dims:list[int],is_bias=False):
        return np.zeros(dims) if is_bias else  np.random.rand(*dims)

    def __init__(self,hidden_layers:list[int],input_parameters_len:int ):
        # intialize hidden layers based on paramters : 
        layers=list()
        # Weights and bias intialization 
        # the 0th layer is input layer so no intialization
        layers.append(None)
        # first layer intialiaztion
        layers.append({
            # Weights add 
            "W": self.generate_arr([hidden_layers[0],input_parameters_len]),
            "B": self.generate_arr([hidden_layers[0],1],is_bias=True)
            })
        # remaining layer intialization till last layer
        for i in range (2,len(hidden_layers)+1):
            print(f'creating layer for {i}')
            layers.append({
                # Weights add 
                "W": self.generate_arr([hidden_layers[i-1],hidden_layers[i-2]]),
                "B": self.generate_arr([hidden_layers[i-1],1],is_bias=True)
            })
        # add the last layer sigmoid 
        layers.append({
            "W":self.generate_arr([1,hidden_layers[len(hidden_layers)-1]]),
            "B":self.generate_arr([1,1],is_bias=True)
        })
        self.layers = layers
        print("Total layers generated : ",len(layers))



    @staticmethod
    def pretty_print_layers(layers: list):
        for i in range(len(layers)):
            print(f"Printing {i+1} layer \n")
            print(layers[i])
            print("\n\n")
    
    @staticmethod
    def sig(X):
        return 1/(1+np.exp(-X))
    
    @staticmethod
    def relu(X):
        return np.maximum(X,0)
    
    @staticmethod
    def relu_derivate(X):
        return (X > 0).astype(int)   # 1 where X > 0, else 0
    
    def forward_prop(self,X):
        print("X is {X}")
        layers_len = len(self.layers)
        for i in range(1, len(self.layers)):
            W= self.layers[i]["W"]
            B= self.layers[i]["B"]
            Z= A = None 
            if i ==1:
                Z= np.dot(W , X) +B 
            else :
                A_PREV = self.layers[i-1]["A"]
                Z = np.dot(W , A_PREV) +B
            if i == layers_len-1 : 
                A = NN.sig(Z)
            else :
                A = NN.relu(Z)
            self.layers[i]["Z"] = Z
            self.layers[i]["A"] = A
        # return the activation func of last layer 
        return self.layers[layers_len-1]["A"]

    def back_prop(self,loss,learning_rate,X):
        layers= self.layers
        # SET TEMPORARY 
        layers[0]= {"A":X}
        layers_len = len(layers)

        for i in reversed(range(1,layers_len)):
            if i == layers_len-1:
                #  Update last layer
                layers[i]["layer_cost"]= loss
                dW = np.dot(loss , layers[i-1]["A"])
                db = loss 
            else:
                layers[i]["layer_cost"]=np.dot(layers[i+1]["layer_cost"],layers[i+1]["W_OLD"])
                layers[i]["layer_cost"]=np.dot(layers[i]["layer_cost"],self.relu_derivate(layers[i-1]["A"]))
                dW = np.dot(layers[i]["layer_cost"],layers[i-1]["A"])
                db = layers[i]["layer_cost"]

            # Update W
            layers[i]["W_OLD"] = layers[i]["W"]
            layers[i]["W"]= layers[i]["W_OLD"] - learning_rate * dW
            # Update B
            layers[i]["B_OLD"] = layers[i]["B"]
            layers[i]["B"]= layers[i]["B_OLD"] - learning_rate * db
        # RESET LAYER 0
        layers[0]=None
        self.layers=layers

    def train(self,X,Y,epocs:int,learning_rate:float):
        print(f"starting training for {len(X)} examples")
        for i in range (epocs):
            # forward prop
            Y_HAT = self.forward_prop(X)
            # loss 
            loss = Y_HAT - Y
            print(f'Y_HAT: {Y_HAT}')
            print(f'Y: {Y}')
            # cost
            cost = np.mean(loss)
            print(f"cost for epoch {i+1} is {cost}")
            # backward prop
            self.back_prop(loss,learning_rate,X)
        print("training completed sucessfully ")
    
    def test(self,x,y):
        # forward prop and y_hat
        y_hat= 0.7
        y_hat = 1 if y_hat>=0.5 else 0
        print(f"The predicted answer by model is {'correct' if y_hat==y else 'incorrect :-( '}")


print(NN.generate_arr([2,3]))

network = NN([3,2,3],3)
network.pretty_print_layers(network.layers)
X = NN.generate_arr([3,4])
Y = np.random.randint(0,2,[1,4])
network.train(X,Y,3,10e-1)