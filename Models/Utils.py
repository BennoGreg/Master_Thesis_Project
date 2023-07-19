import torch.nn as nn

class Printlayer(nn.Module):

    def __init__(self):
        super(Printlayer, self).__init__()


    def forward(self, x):
        print(x.shape)
        return x

