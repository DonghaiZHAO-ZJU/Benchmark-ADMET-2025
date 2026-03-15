import dgl
import dgl.function as fn
import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from dgl.nn import SortPooling, WeightAndSum, GlobalAttentionPooling, Set2Set, SumPooling, AvgPooling, MaxPooling
from scipy.interpolate import BSpline



device = "cpu"
if torch.cuda.is_available():
    device = torch.device("cuda")




class KAN_polynomial(nn.Module):
    def __init__(self, inputdim, outdim, degree, addbias=True):
        super(KAN_polynomial, self).__init__()
        self.degree = degree  # Polynomial degree
        self.addbias = addbias
        self.inputdim = inputdim
        self.outdim = outdim

        # Initialize polynomial coefficients for each input and output dimension
        self.coeffs = nn.Parameter(torch.randn(outdim, inputdim, self.degree + 1) / 
                                   (np.sqrt(inputdim) * np.sqrt(self.degree)))

        if self.addbias:
            self.bias = nn.Parameter(torch.zeros(1, outdim))

    def forward(self, x):
        xshp = x.shape
        outshape = xshp[0:-1] + (self.outdim,)
        x = x.view(-1, self.inputdim)

        # Generate powers of x from 0 to degree for each element in the batch
        x_powers = torch.stack([x**i for i in range(self.degree + 1)], dim=-1)

        # Compute the polynomial using broadcasting
        y = torch.einsum("bij, oij->bo", x_powers, self.coeffs)
        
        if self.addbias:
            y += self.bias

        y = y.view(outshape)
        return y



class NaivePolynomialKANLayer(nn.Module):
    def __init__(self, in_feats, out_feats, degree, addbias=True):
        super(NaivePolynomialKANLayer, self).__init__()
        self.degree = degree 
        self.addbias = addbias
        self.in_feats = in_feats
        self.out_feats = out_feats

        self.coeffs = nn.Parameter(torch.randn(out_feats, in_feats, self.degree + 1) / 
                                   (np.sqrt(in_feats) * np.sqrt(self.degree + 1)))
        if self.addbias:
            self.bias = nn.Parameter(torch.zeros(out_feats))

    def forward(self, g, x):
        
        with g.local_scope():
            g.ndata['h'] = x
            
            g.update_all(message_func=self.polynomial_transform, reduce_func=fn.sum(msg='m', out='h'))
            if self.addbias:
                g.ndata['h'] += self.bias

            return g.ndata['h']

    def polynomial_transform(self, edges):

        src_feat = edges.src['h'] 

        powers = torch.stack([src_feat**i for i in range(self.degree + 1)], dim=-1)

        m = torch.einsum("bij,oij->bo", powers, self.coeffs)

        return {'m': m}
    

class KA_GNN_two(nn.Module):
    def __init__(self, in_feat, hidden_feat, out_feat, out, grid_feat, num_layers, pooling, use_bias=False):
        super(KA_GNN_two, self).__init__()
        self.num_layers = num_layers
        self.pooling = pooling
        #self.lin_in = nn.Linear(in_feat, hidden_feat, bias=use_bias)
        self.layers = nn.ModuleList()

        self.leaky_relu = nn.LeakyReLU()
        self.sigmoid = nn.Sigmoid()
        #self.kan_line = KAN_linear(in_feat, hidden_feat, grid_feat, addbias=use_bias)
        self.kan_line =KAN_polynomial(in_feat, hidden_feat, grid_feat, addbias=use_bias)

        for _ in range(num_layers - 1):
            self.layers.append(NaiveFourierKANLayer(hidden_feat, hidden_feat, grid_feat, addbias=use_bias))

        #self.layers.append()
        #self.layers.append(NaiveFourierKANLayer(hidden_feat, hidden_feat, grid_feat, addbias=use_bias))
        

        #self.layers.append(KAN_linear(hidden_feat, out_feat, grid_feat, addbias=use_bias))
        #self.layers.append(NaiveFourierKANLayer(hidden_feat, out_feat, grid_feat, addbias=use_bias))


        #self.layers.append(NaiveFourierKANLayer(out_feat, out_feat, grid_feat, addbias=use_bias))
        #self.linear_1 = KAN_linear(hidden_feat, out, 1, addbias=True)
        self.linear_1 = KAN_polynomial(hidden_feat, out, 1, addbias=True)
        
        self.sumpool = SumPooling()
        self.avgpool = AvgPooling()
        self.maxpool = MaxPooling()

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
    


class KA_GNN(nn.Module):
    def __init__(self, in_feat, hidden_feat, out_feat, out, grid_feat, num_layers, pooling, use_bias=False):
        super(KA_GNN, self).__init__()
        self.num_layers = num_layers
        self.pooling = pooling
        #self.lin_in = nn.Linear(in_feat, hidden_feat, bias=use_bias)
        #self.kan_line = KAN_linear(in_feat, hidden_feat, grid_feat, addbias=use_bias)
        #self.kan_line = KAN_linear_mask([0], in_feat, hidden_feat, grid_feat, addbias=use_bias)
        self.kan_line = KAN_polynomial(in_feat, hidden_feat, grid_feat, addbias=use_bias)

        self.layers = nn.ModuleList()
        self.leaky_relu = nn.LeakyReLU()
        self.dropout = nn.Dropout(0.1)

        for _ in range(num_layers - 1):
            self.layers.append(NaiveFourierKANLayer(hidden_feat, hidden_feat, grid_feat, addbias=use_bias))
       
        #self.linear_1 = KAN_linear(hidden_feat, out_feat, grid_feat, addbias=use_bias)
        #self.linear_2 = KAN_linear(out_feat, out, grid_feat, addbias=use_bias)
        
        #self.linear_1 = KAN_linear_mask([0],hidden_feat, out_feat, grid_feat, addbias=use_bias)
        #self.linear_2 = KAN_linear_mask([0],out_feat, out, grid_feat, addbias=use_bias)

        self.linear_1 = KAN_polynomial(hidden_feat, out_feat, grid_feat, addbias=use_bias)
        self.linear_2 = KAN_polynomial(out_feat, out, grid_feat, addbias=use_bias)



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
        h = self.kan_line(features)
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

