import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.utils import to_dense_batch

from molmcl.models.gnn import GNN, GPS
from molmcl.models.aggr import PromptAggr


class SwiGLU(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.linear_1 = nn.Linear(dim, dim)
        self.linear_2 = nn.Linear(dim, dim)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.linear_1.weight)
        nn.init.xavier_uniform_(self.linear_2.weight)

    def forward(self, x):
        output = self.linear_1(x)
        swish = output * torch.sigmoid(output)
        swiglu = swish * self.linear_2(x)

        return swiglu


class GNNPredictor(nn.Module):
    def __init__(self, num_layer, emb_dim, num_tasks, JK="last", drop_ratio=0, hidden_dim=64,
                 attn_drop_ratio=0, use_prompt=True, model_head=4, aggr_head=4, normalize=False,
                 atom_feat_dim=None, bond_feat_dim=None, random_init=True, init_weight=None,
                 use_descriptor=False, use_fingerprint=False, temperature=1, baseline=None,
                 add_mean_pool=False, layer_norm=True, layer_norm_out=True, backbone='gnn', act='softmax'):
        super(GNNPredictor, self).__init__()
        self.num_layer = num_layer
        self.drop_ratio = drop_ratio
        self.attn_drop_ratio = attn_drop_ratio
        self.JK = JK
        self.emb_dim = emb_dim
        self.num_tasks = num_tasks
        self.temperature = temperature
        self.use_prompt = use_prompt
        self.baseline = baseline
        self.add_mean_pool = add_mean_pool
        self.normalize = normalize
        self.layer_norm = layer_norm
        self.layer_norm_out = layer_norm_out
        self.backbone = backbone
        self.act = act
        self.use_descriptor = use_descriptor
        self.use_fingerprint = use_fingerprint

        if self.num_layer < 2:
            raise ValueError("Number of GNN layers must be greater than 1.")

        if self.backbone == 'gnn':
            self.gnn = GNN(self.num_layer, self.emb_dim, JK=self.JK, drop_ratio=self.drop_ratio,
                           atom_feat_dim=atom_feat_dim, bond_feat_dim=bond_feat_dim)
        elif self.backbone == 'gps':
            self.gnn = GPS(channels=self.emb_dim, pe_dim=20,
                           node_dim=atom_feat_dim, edge_dim=bond_feat_dim,
                           num_layers=self.num_layer, heads=model_head,
                           attn_type='multihead', dropout=self.drop_ratio,
                           attn_dropout=self.attn_drop_ratio)

        if self.use_prompt:
            self.prompt_token = ['<molsim>', '<scaffsim>', '<context>']
            self.aggrs = nn.ModuleList([PromptAggr(emb_dim=emb_dim,
                                                   num_heads=aggr_head,
                                                   dropout=self.drop_ratio,
                                                   layer_norm_out=layer_norm_out)
                                        for _ in range(len(self.prompt_token))])

            if init_weight is not None:
                self.prompt_weight = nn.Parameter(torch.FloatTensor(init_weight))
            elif random_init:
                self.prompt_weight = nn.Parameter(torch.FloatTensor(len(self.prompt_token)))
                nn.init.normal_(self.prompt_weight)
            else:
                self.prompt_weight = nn.Parameter(torch.FloatTensor(len(self.prompt_token)))
                nn.init.constant_(self.prompt_weight, 1)

            # self.gate_layer = nn.Linear(self.emb_dim, len(self.prompt_token))  # for finetune MoE ablation

        in_dim = self.emb_dim
        if self.use_descriptor:
            self.desc_fn = nn.Linear(200, self.emb_dim)
            in_dim += self.emb_dim

        if self.use_fingerprint:
            self.fp_fn = nn.Linear(512, self.emb_dim)
            in_dim += self.emb_dim

        self.graph_pred_linear = nn.Sequential(
            nn.Linear(in_dim, self.emb_dim),
            nn.Softplus(),
            nn.Linear(self.emb_dim, self.num_tasks)
        )
        # self.graph_pred_linear = nn.Linear(in_dim, self.num_tasks)

        # self.norm_dec = nn.LayerNorm(self.emb_dim)
        # self.norm_fp = nn.LayerNorm(self.emb_dim)
        # self.norm = nn.LayerNorm(self.emb_dim)
        # self.norm = nn.BatchNorm1d(in_dim)
        # self.dropout = nn.Dropout(self.drop_ratio)
        # self.dropout_channel = nn.Dropout(0.5)
        # self.dropout_ft = nn.Dropout(0.3)

        self.reset_parameters()

    def reset_parameters(self):
        if self.use_descriptor:
            nn.init.xavier_uniform_(self.desc_fn.weight)
        if self.use_fingerprint:
            nn.init.xavier_uniform_(self.fp_fn.weight)
        nn.init.xavier_uniform_(self.graph_pred_linear[0].weight)
        nn.init.xavier_uniform_(self.graph_pred_linear[2].weight)

    def set_prompt_weight(self, init_weight):
        # Deactivate channel with really low weight
        # prompt_weight = prompt_weight * (torch.abs(prompt_weight) > 0.1).long()
        #     prompt_weight = prompt_weight / prompt_weight.sum(-1)

        self.prompt_weight.data = init_weight.data.float()
    
    def get_prompt_weight(self, act='softmax'):
        if act == 'softmax':
            prob = F.softmax(self.prompt_weight / self.temperature, dim=-1)
            return prob  # torch.clamp(prob, min=0.1, max=1)
        elif act == 'neg-softmax':
            prob = F.softmax(self.prompt_weight / self.temperature, dim=-1)
            prob = 1 - prob * 2
            return prob
        elif act == 'none':
            return self.prompt_weight
        else:
            raise NotImplementedError

    def freeze_aggr_module(self):
        if self.use_prompt:
            for param in self.aggrs.parameters():
                param.requires_grad = False

    def freeze_gnn_module(self):
        for param in self.gnn.parameters():
            param.requires_grad = False

    def unfreeze_gnn_module(self):
        for param in self.gnn.parameters():
            param.requires_grad = True
    
    def get_representations(self, data, channel_idx=-1, return_score=False):
        if self.backbone == 'gps':
            h_g, node_repres = self.gnn(data.x, data.pe, data.edge_index, data.edge_attr, data.batch)
        else:
            h_g, node_repres = self.gnn(data.x, data.edge_index, data.edge_attr, data.batch)

        if not self.use_prompt:
            return h_g
        else:
            scores = []
            graph_reps = []
            batch_x, batch_mask = to_dense_batch(node_repres, data.batch)

            for i in range(len(self.prompt_token)):
                h_g, h_x, score = self.aggrs[i](batch_x, batch_mask, score_head_i=-1)
                if self.normalize:
                    h_g = F.normalize(h_g, dim=-1)
                scores.append(score)
                graph_reps.append(h_g)

            if channel_idx > -1:
                if return_score:
                    return graph_reps[channel_idx], return_score[channel_idx]
                else:
                    return graph_reps[channel_idx]
            else:
                if return_score:
                    return torch.stack(graph_reps), scores
                else:
                    return torch.stack(graph_reps)
    
    def forward(self, data, channel_idx=-1):
        output = {}

        graph_reps = self.get_representations(data, channel_idx=channel_idx)

        if not self.use_prompt or channel_idx > -1:
            output['predict'] = self.graph_pred_linear(graph_reps)
            output['graph_rep'] = graph_reps
        else:
            prompt_weight = self.get_prompt_weight(act=self.act)
            # prompt_weight = self.dropout_channel(prompt_weight)
            graph_rep = torch.matmul(graph_reps.transpose(0, 2), prompt_weight).transpose(0, 1)

            if self.use_descriptor:
                graph_desc = self.desc_fn(data.desc.view(-1, 200))
                graph_rep = torch.cat([
                    graph_rep,
                    graph_desc
                ], dim=-1)
            if self.use_fingerprint:
                graph_fp = self.fp_fn(data.fp.view(-1, 512))
                graph_rep = torch.cat([
                    graph_rep,
                    graph_fp
                ], dim=-1)

            # graph_rep = self.dropout_ft(graph_rep)
            # graph_rep = self.norm(graph_rep)
            output['predict'] = self.graph_pred_linear(graph_rep)
            output['graph_rep'] = graph_rep
            
        return output
