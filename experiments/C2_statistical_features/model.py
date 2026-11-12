import torch
from torch import nn
class SF(nn.Module):
    def __init__(self,d,hidden=(256,128),dropout=.1):
        super().__init__();layers=[]
        for h in hidden:layers.extend([nn.Linear(d,h),nn.ReLU(),nn.Dropout(dropout)]);d=h
        layers.append(nn.Linear(d,1));self.net=nn.Sequential(*layers)
    def forward(self,x):return self.net(x).squeeze(-1)
