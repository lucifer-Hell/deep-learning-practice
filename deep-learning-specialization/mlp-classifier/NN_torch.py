import torch
import torch.nn as nn
import torch.optim as optim

class NN_torch(nn.Module):
    def __init__(self,hidden_layers:list[int],input_parameters_len:int ,output_size=1, learning_rate=0.1):
        super(NN_torch, self).__init__()   # ✅ add this
        layers=[]
        input_dim=input_parameters_len
        for hidden_dim in hidden_layers:
            layers.append(nn.Linear(input_dim,hidden_dim))
            layers.append(nn.ReLU())
            input_dim=hidden_dim
        layers.append(nn.Linear(input_dim,output_size))
        layers.append(nn.Sigmoid())
        self.model= nn.Sequential(*layers)
        self.criterion= nn.BCELoss()
        self.optimizer= optim.Adam(self.model.parameters(),lr=learning_rate)
    
    def forward(self,x):
        return self.model(x)
    
    def train_model(self, X , Y , epochs:int = 1000):
        for i in range(epochs):
            Y_HAT = self.forward(X)
            loss = self.criterion(Y_HAT,Y)
            # to remove old gradients 
            self.optimizer.zero_grad()
            # propogate loss backward 
            loss.backward()
            # udpate the wts 
            self.optimizer.step()

    def test(self,x,y):
        with torch.no_grad():
            output = self.forward(x)
            output = 1 if output >= 0.5 else 0
            if output == y :
                print("correct prediction")
            else: 
                print("wrong prediction ")


