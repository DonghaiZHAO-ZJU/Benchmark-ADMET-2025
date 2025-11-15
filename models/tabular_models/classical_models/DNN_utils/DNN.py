import torch.nn as nn

class mlp(nn.Module):
    def __init__(self, input_dim, layer_num, hidden_dim, dropout, out_dim=1):
        super(mlp, self).__init__()
        self.layer_num=layer_num
        self.layers=nn.ModuleList()
        if self.layer_num == 1:
            self.layers.append(self.fc_layer(dropout, input_dim, hidden_dim))
        else:
            self.layers.append(self.fc_layer(dropout, input_dim, hidden_dim))
            for _ in range(self.layer_num-1):
                self.layers.append(self.fc_layer(dropout, hidden_dim, hidden_dim))
        self.predict = self.output_layer(hidden_dim, out_dim)
    
    def forward(self, embedding):
        for layer in self.layers:
            embedding = layer(embedding)
        out = self.predict(embedding)
        return out

    def fc_layer(self, dropout, in_feats, hidden_feats):
        return nn.Sequential(
                nn.Dropout(dropout),
                nn.Linear(in_feats, hidden_feats),
                nn.ReLU(),
                nn.BatchNorm1d(hidden_feats)
                )

    def output_layer(self, hidden_feats, out_feats):
        return nn.Sequential(
                nn.Linear(hidden_feats, out_feats)
                )