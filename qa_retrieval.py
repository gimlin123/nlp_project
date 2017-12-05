import torch
import torch.nn as nn
import torch.nn.functional as F

# model
class  CNN_Text(nn.Module):

    def __init__(self, V, D, C, Ci, Co, Ks):
        super(CNN_Text,self).__init__()

        self.embed = nn.Embedding(V, D)
        self.convs1 = nn.ModuleList([nn.Conv2d(Ci, Co, (K, D)) for K in Ks])

        self.dropout = nn.Dropout()
        self.fc1 = nn.Linear(len(Ks)*Co, C)

    def forward(self, x):
        x = self.embed(x) # (N,W,D)

        x = x.unsqueeze(1) # (N,Ci,W,D)

        x = [F.relu(conv(x)).squeeze(3) for conv in self.convs1] #[(N,Co,W), ...]*len(Ks)


        x = [F.max_pool1d(i, i.size(2)).squeeze(2) for i in x] #[(N,Co), ...]*len(Ks)

        x = torch.cat(x, 1)

        x = self.dropout(x) # (N,len(Ks)*Co)
        feature_vec = self.fc1(x) # (N,C)
        return feature_vec
