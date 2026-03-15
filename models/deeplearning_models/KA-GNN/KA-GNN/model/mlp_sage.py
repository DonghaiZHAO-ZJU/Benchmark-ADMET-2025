import dgl
import dgl.function as fn
import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F


from dgl.nn import SAGEConv
from dgl.nn import SortPooling, WeightAndSum, GlobalAttentionPooling, Set2Set, SumPooling, AvgPooling, MaxPooling



device = "cpu"
if torch.cuda.is_available():
    device = torch.device("cuda")



class MLPGNN_two(nn.Module):
    def __init__(self, in_feat, hidden_feat, out_feat, out, grid_feat, num_layers, pooling, use_bias=False):
        super(MLPGNN_two, self).__init__()
        self.num_layers = num_layers
        self.pooling = pooling
        #self.lin_in = nn.Linear(in_feat, hidden_feat, bias=use_bias)
        self.layers = nn.ModuleList()

        self.leaky_relu = nn.LeakyReLU()
        self.sigmoid = nn.Sigmoid()
        self.kan_line = nn.Linear(in_feat, hidden_feat, bias=use_bias)

        for _ in range(num_layers - 1):
            self.layers.append(SAGEConv(hidden_feat, hidden_feat, 'mean'))

        #self.layers.append()
        #self.layers.append(NaiveFourierKANLayer(hidden_feat, hidden_feat, grid_feat, addbias=use_bias))
        self.sumpool = SumPooling()
        self.avgpool = AvgPooling()
        self.maxpool = MaxPooling()

        #self.layers.append(KAN_linear(hidden_feat, out_feat, grid_feat, addbias=use_bias))
        #self.layers.append(NaiveFourierKANLayer(hidden_feat, out_feat, grid_feat, addbias=use_bias))


        #self.layers.append(NaiveFourierKANLayer(out_feat, out_feat, grid_feat, addbias=use_bias))
        self.linear_1 = nn.Linear(hidden_feat, out,bias=True)
        #self.linear_2 = KAN_linear(out_feat, out, grid_feat, addbias=True)
        
        layers_kan = [
                        #nn.Linear(self.hidden_size*2, self.hidden_size),
                        self.linear_1,
                        nn.Sigmoid()
                        ]
        
        self.Readout = nn.Sequential(*layers_kan)  

        
    def forward(self, g, h):
        h = self.kan_line(h)

        for i, layer in enumerate(self.layers):
            m = layer(g, h)  
            #h = self.leaky_relu(torch.add(m,h)
            h = nn.functional.leaky_relu(torch.add(m, h))
        
        if self.pooling == 'avg':
            y = self.avgpool(g, h)


        elif self.pooling == 'max':
            y = self.maxpool(g, h)
            
        
        elif self.pooling == 'sum':
            y = self.sumpool(g, h)


        else:
            print('No pooling found!!!!')

        out = self.Readout(y)    
        return out
    
    def get_grad_norm_weights(self) -> nn.Module:
        return self.parameters()
    


class MLPGNN(nn.Module):
    def __init__(self, in_feat, hidden_feat, out_feat, out, grid_feat, num_layers, pooling, use_bias=False):
        super(MLPGNN, self).__init__()
        self.num_layers = num_layers
        self.pooling = pooling
        #self.lin_in = nn.Linear(in_feat, hidden_feat, bias=use_bias)
        self.lin_in = nn.Linear(in_feat, hidden_feat,bias=use_bias)
        self.layers = nn.ModuleList()
        self.leaky_relu = nn.LeakyReLU()
        self.dropout = nn.Dropout(0.1)

        for _ in range(num_layers - 1):
            self.layers.append(SAGEConv(hidden_feat, hidden_feat, 'mean'))

        #self.layers.append(nn.Linear(hidden_feat, out_feat, bias=use_bias))
        self.linear_1 = nn.Linear(hidden_feat, out_feat,bias=use_bias)
        self.linear_2 = nn.Linear(out_feat, out, bias=use_bias)
        
        #self.linear = nn.Linear(hidden_feat, out, bias=use_bias)
        
        self.sumpool = SumPooling()
        self.avgpool = AvgPooling()
        self.maxpool = MaxPooling()
        
        layers_kan = [
                        #nn.Linear(self.hidden_size*2, self.hidden_size),
                        self.linear_1,
                        #nn.Sigmoid(),
                        self.leaky_relu,
                        self.linear_2,
                        nn.Sigmoid()
                        ]
        
        self.Readout = nn.Sequential(*layers_kan)  

    def forward(self, g, features):
        h = self.lin_in(features)
        for i, layer in enumerate(self.layers):
            if i < self.num_layers - 1:
                h = layer(g, h) 
                
            else:
                h = layer(h)  
        if self.pooling == 'avg':
            y = self.avgpool(g, h)
            

        elif self.pooling == 'max':
            y = self.maxpool(g, h)
            
        
        elif self.pooling == 'sum':
            y = self.sumpool(g, h)


        else:
            print('No pooling found!!!!')

        out = self.Readout(y)     
        return out
    
    def get_grad_norm_weights(self) -> nn.Module:
        
        return self.parameters()
    
