
import torch
import torch.nn as nn 
import torch.nn.functional as F 
class MNISTDNN(nn.Module):
    def __init__(self):
        super(MNISTDNN, self).__init__()
        self.fc1 = nn.Linear(784,1000)
        self.fc2 = nn.Linear(1000,500)
        self.fc3 = nn.Linear(500,250)
        self.fc4 = nn.Linear(250,250)
        self.fc5 = nn.BatchNorm1d(250)
        self.fc5_drop = nn.Dropout(p=0.5)
        self.fc6 = nn.Linear(250,10)
        """
        self.ff = nn.Sequential(
                self.fc1, nn.ReLU(),
                self.fc2, nn.ReLU(),
                self.fc3, nn.ReLU(),
                self.fc4, nn.ReLU(),
                self.fc5, self.fc5_drop,
                self.fc6)
        """

    def forward(self,x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = F.relu(self.fc4(x))
        x = self.fc5(x)
        x = self.fc5_drop(x)
        return self.fc6(x)
        #return(self.ff(x.view(-1,28*28)))



