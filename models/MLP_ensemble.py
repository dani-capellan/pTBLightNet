import torch
from torch import nn

class MLP(nn.Module):
    '''
    Multilayer Perceptron.
    '''
    def __init__(self,in_dim,outputs):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(in_dim,256),
            nn.ReLU(),
            nn.Linear(256,outputs)
        )
        
    # def __init__(self,in_dim,outputs):
    #     super().__init__()
    #     self.layers = nn.Sequential(
    #         nn.Linear(in_dim,outputs)
    #     )

    def forward(self, x):
        '''Forward pass'''
        return self.layers(x)

class MLPv2(nn.Module):
    '''
    Multilayer Perceptron - with dimensionality reduction of the each of the inputs (x1,x2). x1:AP, x2:LAT.
    '''
    def __init__(self,in_dim_x1,in_dim_x2,outputs):
        super().__init__()
        self.enc_x1 = nn.Sequential(
            nn.Linear(in_dim_x1,16),
            nn.ReLU(),
        )
        self.enc_x2 = nn.Sequential(
            nn.Linear(in_dim_x2,16),
            nn.ReLU(),
        )
        self.clf = nn.Sequential(
            nn.Linear(32,10),
            nn.ReLU(),
            nn.Linear(10,outputs)
        )

    def forward(self, x1, x2, return_features=False):
        '''Forward pass'''
        x1 = self.enc_x1(x1)
        x2 = self.enc_x2(x2)
        x = torch.cat((x1,x2), dim=-1)
        if(return_features):
            return self.clf(x), x1, x2  
        else:
            return self.clf(x)


class MLPv2_onlyclv(nn.Module):
    '''
    Multilayer Perceptron - CLF for only clinical variables
    Includes clinical variables (clv)
    '''
    def __init__(self,in_dim_clv,outputs):
        super().__init__()
        self.clf = nn.Sequential(
            nn.Linear(in_dim_clv,10),
            nn.ReLU(),
            nn.Linear(10,outputs)
        )

    def forward(self, clv):
        '''Forward pass'''
        return self.clf(clv) 


class MLPv3(nn.Module):
    '''
    Multilayer Perceptron - with dimensionality reduction of the each of the inputs (x1,x2). x1:AP, x2:LAT.
    Includes a batchnorm
    '''
    def __init__(self,in_dim_x1,in_dim_x2,outputs):
        super().__init__()
        self.enc_x1 = nn.Sequential(
            nn.Linear(in_dim_x1,16),
            nn.ReLU(),
        )
        self.enc_x2 = nn.Sequential(
            nn.Linear(in_dim_x2,16),
            nn.ReLU(),
        )
        self.clf = nn.Sequential(
            nn.Linear(32,10),
            nn.ReLU(),
            nn.Linear(10,outputs)
        )
        self.bn = nn.BatchNorm1d(16+16)

    def forward(self, x1, x2, return_features=False):
        '''Forward pass'''
        x1 = self.enc_x1(x1)
        x2 = self.enc_x2(x2)
        x = self.bn(torch.cat((x1,x2), dim=-1))
        if(return_features):
            return self.clf(x), x1, x2  
        else:
            return self.clf(x)


class MLPv2_clv(nn.Module):
    '''
    Multilayer Perceptron - with dimensionality reduction of the each of the inputs (x1,x2). x1:AP, x2:LAT.
    Includes clinical variables (clv)
    '''
    def __init__(self,in_dim_x1,in_dim_x2,in_dim_clv,outputs):
        super().__init__()
        self.enc_x1 = nn.Sequential(
            nn.Linear(in_dim_x1,16),
            nn.ReLU(),
        )
        self.enc_x2 = nn.Sequential(
            nn.Linear(in_dim_x2,16),
            nn.ReLU(),
        )
        self.clf = nn.Sequential(
            nn.Linear(32+in_dim_clv,10),
            nn.ReLU(),
            nn.Linear(10,outputs)
        )

    def forward(self, x1, x2, clv, return_features=False):
        '''Forward pass'''
        x1 = self.enc_x1(x1)
        x2 = self.enc_x2(x2)
        x = torch.cat((x1,x2,clv), dim=-1)
        if(return_features):
            return self.clf(x), x1, x2  
        else:
            return self.clf(x)


class MLPv3_clv(nn.Module):
    '''
    Multilayer Perceptron - with dimensionality reduction of the each of the inputs (x1,x2). x1:AP, x2:LAT.
    Includes a batchnorm
    Includes clinical variables (clv)
    '''
    def __init__(self,in_dim_x1,in_dim_x2,in_dim_clv,outputs):
        super().__init__()
        self.enc_x1 = nn.Sequential(
            nn.Linear(in_dim_x1,16),
            nn.ReLU(),
        )
        self.enc_x2 = nn.Sequential(
            nn.Linear(in_dim_x2,16),
            nn.ReLU(),
        )
        self.clf = nn.Sequential(
            nn.Linear(32+in_dim_clv,10),
            nn.ReLU(),
            nn.Linear(10,outputs)
        )
        self.bn = nn.BatchNorm1d(16+16+in_dim_clv)

    def forward(self, x1, x2, clv, return_features=False):
        '''Forward pass'''
        x1 = self.enc_x1(x1)
        x2 = self.enc_x2(x2)
        x = self.bn(torch.cat((x1,x2,clv), dim=-1))
        if(return_features):
            return self.clf(x), x1, x2  
        else:
            return self.clf(x)


class MLPv4(nn.Module):
    '''
    MLPv4 = MLPv3
    Multilayer Perceptron - with dimensionality reduction of the each of the inputs (x1,x2). x1:AP, x2:LAT.
    Includes a batchnorm
    '''
    def __init__(self,in_dim_x1,in_dim_x2,outputs):
        super().__init__()
        self.enc_x1 = nn.Sequential(
            nn.Linear(in_dim_x1,16),
            nn.ReLU(),
        )
        self.enc_x2 = nn.Sequential(
            nn.Linear(in_dim_x2,16),
            nn.ReLU(),
        )
        self.clf = nn.Sequential(
            nn.Linear(32,10),
            nn.ReLU(),
            nn.Linear(10,outputs)
        )
        self.bn = nn.BatchNorm1d(16+16)

    def forward(self, x1, x2, return_features=False):
        '''Forward pass'''
        x1 = self.enc_x1(x1)
        x2 = self.enc_x2(x2)
        x = self.bn(torch.cat((x1,x2), dim=-1))
        if(return_features):
            return self.clf(x), x1, x2  
        else:
            return self.clf(x)


class MLPv4_clv(nn.Module):
    '''
    Multilayer Perceptron - with dimensionality reduction of the each of the inputs (x1,x2). x1:AP, x2:LAT.
    Includes batchnorms and feature scaling by using linear layers
    Includes clinical variables (clv)
    '''
    def __init__(self,in_dim_x1,in_dim_x2,in_dim_clv,outputs):
        super().__init__()
        self.enc_x1 = nn.Sequential(
            nn.Linear(in_dim_x1,16),
            nn.ReLU(),
        )
        self.enc_x2 = nn.Sequential(
            nn.Linear(in_dim_x2,16),
            nn.ReLU(),
        )
        self.enc_clv = nn.Sequential(
            nn.Linear(in_dim_clv,in_dim_clv),
            nn.ReLU(),
        )
        self.clf = nn.Sequential(
            nn.Linear(32+in_dim_clv,10),
            nn.ReLU(),
            nn.Linear(10,outputs)
        )
        self.bn = nn.BatchNorm1d(16+16+in_dim_clv)

    def forward(self, x1, x2, clv, return_features=False):
        '''Forward pass'''
        x1 = self.enc_x1(x1)
        x2 = self.enc_x2(x2)
        clv = self.enc_clv(clv)
        x = self.bn(torch.cat((x1,x2,clv), dim=-1))
        if(return_features):
            return self.clf(x), x1, x2  
        else:
            return self.clf(x)
    
    
class MLPv4_clv_wo_bn(nn.Module):
    '''
    Multilayer Perceptron - with dimensionality reduction of the each of the inputs (x1,x2). x1:AP, x2:LAT.
    Includes batchnorms and feature scaling by using linear layers
    Includes clinical variables (clv)
    '''
    def __init__(self,in_dim_x1,in_dim_x2,in_dim_clv,outputs):
        super().__init__()
        self.enc_x1 = nn.Sequential(
            nn.Linear(in_dim_x1,16),
            nn.ReLU(),
        )
        self.enc_x2 = nn.Sequential(
            nn.Linear(in_dim_x2,16),
            nn.ReLU(),
        )
        self.enc_clv = nn.Sequential(
            nn.Linear(in_dim_clv,in_dim_clv),
            nn.ReLU(),
        )
        self.clf = nn.Sequential(
            nn.Linear(32+in_dim_clv,10),
            nn.ReLU(),
            nn.Linear(10,outputs)
        )

    def forward(self, x1, x2, clv, return_features=False):
        '''Forward pass'''
        x1 = self.enc_x1(x1)
        x2 = self.enc_x2(x2)
        clv = self.enc_clv(clv)
        x = torch.cat((x1,x2,clv), dim=-1)
        if(return_features):
            return self.clf(x), x1, x2  
        else:
            return self.clf(x)