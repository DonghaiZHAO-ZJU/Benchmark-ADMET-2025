import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl
import dgl.function as fn
import math
import numpy as np
import copy

from dgl.nn import SortPooling, WeightAndSum, GlobalAttentionPooling, Set2Set, SumPooling, AvgPooling, MaxPooling
from dgl.nn.functional import edge_softmax


device = "cpu"
if torch.cuda.is_available():
    device = torch.device("cuda")



class KAN_linear(nn.Module):
    def __init__(self, inputdim, outdim, gridsize, addbias=False):
        super(KAN_linear,self).__init__()
        self.gridsize= gridsize
        self.addbias = addbias
        self.inputdim = inputdim
        self.outdim = outdim

        self.fouriercoeffs = nn.Parameter(torch.randn(2, outdim, inputdim, gridsize) / 
                                             (np.sqrt(inputdim) * np.sqrt(self.gridsize)))
        if self.addbias:
            self.bias = nn.Parameter(torch.zeros(1, outdim))

    def forward(self,x):

        xshp = x.shape
        outshape = xshp[0:-1] + (self.outdim,)
        x = x.view(-1, self.inputdim)
        #Starting at 1 because constant terms are in the bias
        k = torch.reshape(torch.arange(1, self.gridsize+1, device=x.device), (1, 1, 1, self.gridsize))
        xrshp = x.view(x.shape[0], 1, x.shape[1], 1)
        #This should be fused to avoid materializing memory
        c = torch.cos(k * xrshp)
        s = torch.sin(k * xrshp)

        c = torch.reshape(c, (1, x.shape[0], x.shape[1], self.gridsize))
        s = torch.reshape(s, (1, x.shape[0], x.shape[1], self.gridsize))
        y = torch.einsum("dbik,djik->bj", torch.concat([c, s], axis=0), self.fouriercoeffs)
        if self.addbias:
            y += self.bias
        
        y = y.view(outshape)
        return y
    


class Gat_mlp_layer(nn.Module):
    def __init__(self, in_node_feats, in_edge_feats, out_node_feats, out_edge_feats, num_heads, grid_size, bias=True):
        super(Gat_mlp_layer, self).__init__()
        self._num_heads = num_heads
        self._out_node_feats = out_node_feats
        self._out_edge_feats = out_edge_feats
        self.fc_node = nn.Linear(in_node_feats+in_edge_feats, out_node_feats * num_heads, bias=True)
        self.fc_ni = nn.Linear(in_node_feats, out_edge_feats * num_heads, bias=False)
        self.fc_fij = nn.Linear(in_edge_feats, out_edge_feats * num_heads, bias=False)
        self.fc_nj = nn.Linear(in_node_feats, out_edge_feats * num_heads, bias=False)
        self.attn = nn.Parameter(torch.FloatTensor(size=(1, num_heads, out_edge_feats)))
        self.output_node = nn.Linear(out_node_feats, out_node_feats, bias=False)
        self.output_edge = nn.Linear(out_edge_feats, out_edge_feats, bias=False)
        if bias:
            self.bias = nn.Parameter(torch.FloatTensor(size=(num_heads * out_edge_feats,)))
        else:
            self.register_buffer('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_normal_(self.fc_node.weight)
        nn.init.xavier_normal_(self.fc_ni.weight)
        nn.init.xavier_normal_(self.fc_fij.weight)
        nn.init.xavier_normal_(self.fc_nj.weight)
        nn.init.xavier_normal_(self.attn)
        nn.init.xavier_normal_(self.output_node.weight)
        nn.init.xavier_normal_(self.output_edge.weight)
        if self.bias is not None:
            nn.init.constant_(self.bias, 0)
    '''
        self.fc_node = KAN_linear(in_node_feats+in_edge_feats, out_node_feats * num_heads, grid_size, addbias=True)
        self.fc_ni = KAN_linear(in_node_feats, out_edge_feats * num_heads, grid_size)
        self.fc_fij = KAN_linear(in_edge_feats, out_edge_feats * num_heads, grid_size)
        self.fc_nj = KAN_linear(in_node_feats, out_edge_feats * num_heads, grid_size)
        self.attn = nn.Parameter(torch.FloatTensor(size=(1, num_heads, out_edge_feats)))
        if bias:
            self.bias = nn.Parameter(torch.FloatTensor(size=(num_heads * out_edge_feats,)))
        else:
            self.register_buffer('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_normal_(self.attn)
        if self.bias is not None:
            nn.init.constant_(self.bias, 0)
    '''
    def message_func(self, edges):
        return {'feat': edges.data['feat']}

    def reduce_func(self, nodes):

        num_edges = nodes.mailbox['feat'].size(1) 
        agg_feats = torch.sum(nodes.mailbox['feat'], dim=1) / num_edges  
        return {'agg_feats': agg_feats}
    

    def forward(self, graph, nfeats, efeats, get_attention=False):
        with graph.local_scope():
            graph.ndata['feat'] = nfeats
            graph.edata['feat'] = efeats
            in_degrees = graph.in_degrees().float().unsqueeze(-1)
            in_degrees[in_degrees == 0] = 1  
            f_ni = self.fc_ni(nfeats)# in_node_feats --> out_edge_feats
            f_nj = self.fc_nj(nfeats)# in_node_feats --> out_edge_feats
            f_fij = self.fc_fij(efeats)# in_edge_feats --> out_edge_feats

            graph.srcdata.update({'f_ni': f_ni})
            graph.dstdata.update({'f_nj': f_nj})
            graph.apply_edges(fn.u_add_v('f_ni', 'f_nj', 'f_tmp'))
            
            f_out = graph.edata.pop('f_tmp') + f_fij
            if self.bias is not None:
                f_out = f_out + self.bias
            f_out = nn.functional.leaky_relu(f_out)
            f_out = f_out.view(-1, self._num_heads, self._out_edge_feats)
            
            e = (f_out * self.attn).sum(dim=-1).unsqueeze(-1)

            graph.send_and_recv(graph.edges(), self.message_func, reduce_func=self.reduce_func)
            m_feats = torch.cat((graph.ndata['feat'],graph.ndata['agg_feats']),dim=1)
            
            graph.edata['a'] = edge_softmax(graph, e)
            
            graph.ndata['h_out'] = self.fc_node(m_feats).view(-1, self._num_heads, self._out_node_feats)
            
            graph.update_all(fn.u_mul_e('h_out', 'a', 'm'),
                             fn.sum('m', 'h_out'))

            h_out = nn.functional.leaky_relu(graph.ndata['h_out'])
            h_out = h_out.view(-1, self._num_heads, self._out_node_feats)

            h_out = torch.sum(h_out, dim=1)
            f_out = torch.sum(f_out, dim=1)

            out_n = self.output_node(h_out)
            out_e = self.output_edge(f_out)
            if get_attention:
                return out_n, out_e, graph.edata.pop('a')
            else:
                return out_n, out_e



class MLP_GAT(nn.Module):
    def __init__(self, in_node_dim, in_edge_dim, hidden_dim, out_1, out_2, gride_size, head, layer_num, pooling):
        super(MLP_GAT, self).__init__()
        self.in_node_dim = in_node_dim
        self.in_edge_dim = in_edge_dim
        self.hidden_dim = hidden_dim
        self.out_1 = out_1
        self.out_2 = out_2
        self.head = head
        self.layer = layer_num

        self.grid_size = gride_size
        self.pooling = pooling

        
        self.attentions = nn.ModuleList()
        

        
        self.attentions.append(Gat_mlp_layer(in_node_feats=in_node_dim,in_edge_feats=in_edge_dim,
                                             out_node_feats=hidden_dim,out_edge_feats=hidden_dim,
                                             num_heads=self.head,grid_size=self.grid_size))
        
        for _ in range(self.layer-1):
            self.attentions.append(Gat_mlp_layer(in_node_feats=hidden_dim,in_edge_feats=hidden_dim,
                                                 out_node_feats=hidden_dim,out_edge_feats=hidden_dim,
                                                 num_heads=self.head,grid_size=self.grid_size))

        self.leaky_relu = nn.LeakyReLU()
        self.sumpool = SumPooling()
        self.avgpool = AvgPooling()
        self.maxpool = MaxPooling()


        out_layers = [
            nn.Linear(hidden_dim, out_1, bias=False),
            self.leaky_relu,
            nn.Linear(out_1, out_2, bias=True),
            nn.Sigmoid()
        ]
        self.Readout = nn.Sequential(*out_layers)



    def forward(self, g, node_feature, edge_feature):
        
        
        for i in range(len(self.attentions)):
            atten = self.attentions[i]
            
            node_feature, edge_feature = atten(g, node_feature, edge_feature)

            
        
        out1 = F.leaky_relu(node_feature)

        if self.pooling == 'avg':
            y = self.avgpool(g, out1)
            
        elif self.pooling == 'max':
            y = self.maxpool(g, out1)
            
        elif self.pooling == 'sum':
            y = self.sumpool(g, out1)
            
        else:
            print('No pooling found!!!!')
        
        out = self.Readout(y)
        
        

        return out


