
import torch.nn as nn


class Autoencoder1D(nn.Module):

    def __init__(self):
        super(Autoencoder1D, self).__init__()
        self.encoder = nn.Sequential(
            #Printlayer(),
            nn.Conv1d(1, 225, 3,),
            nn.LeakyReLU(True),
            nn.BatchNorm1d(225),
            
            nn.Conv1d(225, 256, 3),
            nn.LeakyReLU(True),
            nn.BatchNorm1d(256),
            
            nn.Conv1d(256, 256, 3),
            nn.LeakyReLU(True),
            nn.BatchNorm1d(256),
            #Printlayer(),
            
            nn.Conv1d(256, 128, 3,),
            nn.LeakyReLU(True),
            nn.BatchNorm1d(128),
            
            nn.Conv1d(128, 128, 3,),
            nn.LeakyReLU(True),
            nn.BatchNorm1d(128),
            
            nn.Conv1d(128, 128, 3,),
            nn.LeakyReLU(True),
            nn.BatchNorm1d(128),
            
            #Printlayer(),
            nn.Conv1d(128, 64, 3, ),
            nn.LeakyReLU(True),
            nn.BatchNorm1d(64),
            
            nn.Conv1d(64, 64, 3, ),
            nn.LeakyReLU(True),
            nn.BatchNorm1d(64),
            
            nn.Conv1d(64, 1, 3, ),
            nn.LeakyReLU(True),
            nn.BatchNorm1d(1),
            
            
            
        )

        self.decoder = nn.Sequential(
            
            
            nn.ConvTranspose1d(1, 64, 3,),
            nn.LeakyReLU(),
            nn.BatchNorm1d(64),
            
            nn.ConvTranspose1d(64, 64, 3,),
            nn.LeakyReLU(),
            nn.BatchNorm1d(64),
            

            nn.ConvTranspose1d(64, 128,3,),
            nn.LeakyReLU(),
            nn.BatchNorm1d(128),
            
            nn.ConvTranspose1d(128, 128,3,),
            nn.LeakyReLU(),
            nn.BatchNorm1d(128),
            
            nn.ConvTranspose1d(128, 128,3,),
            nn.LeakyReLU(),
            nn.BatchNorm1d(128),
            
            # Printlayer(),
            nn.ConvTranspose1d(128, 256, 3),
            nn.LeakyReLU(),
            nn.BatchNorm1d(256),
            
            nn.ConvTranspose1d(256, 256, 5),
            nn.LeakyReLU(),
            nn.BatchNorm1d(256),
            
            nn.ConvTranspose1d(256,225, 3),
            nn.LeakyReLU(),
            nn.BatchNorm1d(225),
            
            # Printlayer(),
            nn.ConvTranspose1d(225, 1, 1, ),
            
            # Printlayer(),
        )

    def forward(self, x):
        x = self.encode(x)
        #print(x.shape)
        return self.decode(x)

    def encode(self,x):
        x = self.encoder(x)
        return x

    def decode(self,x):
        x = self.decoder(x)
        return x



''' # other Network configurations tried out with 1D-Convolutions
    def __init__(self):
        super(Autoencoder1D, self).__init__()
        self.encoder = nn.Sequential(
            #Printlayer(),
            nn.Conv1d(1, 225, 5, dilation=1),
            nn.ReLU(True),
            nn.BatchNorm1d(225),
            #Printlayer(),
            nn.Conv1d(225, 256, 3, stride=2),
            nn.ReLU(True),
            nn.BatchNorm1d(256),
            #Printlayer(),
            nn.Conv1d(256, 256, 3, stride=2),
            nn.ReLU(True),
            nn.BatchNorm1d(256),
            #Printlayer(),
            nn.Conv1d(256, 1, 3, stride=1, ),
            nn.ReLU(True),
            nn.BatchNorm1d(1),
            #Printlayer(),

            #nn.Conv1d(100, 1, 3, stride=1, ),
            #nn.ReLU(True),
            #nn.BatchNorm1d(1),
            #Printlayer()
        )

        self.decoder = nn.Sequential(
            #nn.ConvTranspose1d(1, 100, 3, stride=1),
            #nn.ReLU(),
            #nn.BatchNorm1d(100),

            nn.ConvTranspose1d(1, 256, 5, stride=2),
            nn.ReLU(),
            nn.BatchNorm1d(256),
            # Printlayer(),
            nn.ConvTranspose1d(256, 256, 5, stride=1),
            nn.ReLU(),
            nn.BatchNorm1d(256),
            # Printlayer(),
            nn.ConvTranspose1d(256, 256, 5, stride=2),
            nn.ReLU(),
            nn.BatchNorm1d(256),
            # Printlayer(),
            nn.ConvTranspose1d(256, 1, 1, stride=1, ),
            # Printlayer(),
        )
'''

'''
            nn.ConvTranspose1d(1, 2, 3),
            nn.ReLU(),
            nn.BatchNorm1d(2),
            
            nn.ConvTranspose1d(2, 4, 3),
            nn.ReLU(),
            nn.BatchNorm1d(4),
            
            nn.ConvTranspose1d(4, 8, 3),
            nn.ReLU(),
            nn.BatchNorm1d(8),
            
            nn.ConvTranspose1d(8, 16, 3),
            nn.ReLU(),
            nn.BatchNorm1d(16),
            
            nn.ConvTranspose1d(16, 32, 3),
            nn.ReLU(),
            nn.BatchNorm1d(32),
            
            nn.ConvTranspose1d(32, 64, 3),
            nn.ReLU(),
            nn.BatchNorm1d(64),
            
            nn.ConvTranspose1d(64, 128, 3),
            nn.ReLU(),
            nn.BatchNorm1d(128),
            
            '''

'''
            nn.Conv1d(64, 32, 3, ),
            nn.ReLU(True),
            nn.BatchNorm1d(32),
            
            nn.Conv1d(32, 16, 3, ),
            nn.ReLU(True),
            nn.BatchNorm1d(16),
            
            nn.Conv1d(16, 8, 3, ),
            nn.ReLU(True),
            nn.BatchNorm1d(8),
            
            nn.Conv1d(8, 4, 3, ),
            nn.ReLU(True),
            nn.BatchNorm1d(4),
            
            nn.Conv1d(4, 2, 3, ),
            nn.ReLU(True),
            nn.BatchNorm1d(2),
            
            nn.Conv1d(2, 1, 3, ),
            nn.ReLU(True),
            nn.BatchNorm1d(1),
            
            
            
            
            def __init__(self):
        super(Autoencoder1D, self).__init__()
        self.encoder = nn.Sequential(
            #Printlayer(),
            nn.Conv1d(1, 225, 3,),
            nn.ReLU(True),
            nn.BatchNorm1d(225),
            #Printlayer(),
            
            nn.Conv1d(225, 225, 3,),
            nn.ReLU(True),
            nn.BatchNorm1d(225),
            #Printlayer(),
            
            nn.Conv1d(225, 225, 3,),
            nn.ReLU(True),
            nn.BatchNorm1d(225),
            
            nn.Conv1d(225, 256, 3,),
            nn.ReLU(True),
            nn.BatchNorm1d(256),
            
            #Printlayer(),
            nn.Conv1d(256, 256, 3,),
            nn.ReLU(True),
            nn.BatchNorm1d(256),
            
            nn.Conv1d(256, 256, 3,),
            nn.ReLU(True),
            nn.BatchNorm1d(256),
            
            nn.Conv1d(256, 256, 3,),
            nn.ReLU(True),
            nn.BatchNorm1d(256),
            
            #Printlayer(),
            nn.Conv1d(256, 128, 3, ),
            nn.ReLU(True),
            nn.BatchNorm1d(128),
            
            nn.Conv1d(128, 64, 3, ),
            nn.ReLU(True),
            nn.BatchNorm1d(64),
            
            nn.Conv1d(64, 1, 3, ),
            nn.ReLU(True),
            nn.BatchNorm1d(1),
            
            
            
        )

        self.decoder = nn.Sequential(
            
            nn.ConvTranspose1d(1, 64, 3),
            nn.ReLU(True),
            nn.BatchNorm1d(64),
            
            nn.ConvTranspose1d(64, 64, 3),
            nn.ReLU(),
            nn.BatchNorm1d(64),
            
            nn.ConvTranspose1d(64, 64, 3),
            nn.ReLU(),
            nn.BatchNorm1d(64),
            
            nn.ConvTranspose1d(64, 128, 3),
            nn.ReLU(),
            nn.BatchNorm1d(128),
            
            nn.ConvTranspose1d(128, 128, 3),
            nn.ReLU(),
            nn.BatchNorm1d(128),
            
            nn.ConvTranspose1d(128, 128, 3),
            nn.ReLU(),
            nn.BatchNorm1d(128),

            nn.ConvTranspose1d(128, 256, 3,),
            nn.ReLU(),
            nn.BatchNorm1d(256),
            
            # Printlayer(),
            nn.ConvTranspose1d(256, 256, 5),
            nn.ReLU(),
            nn.BatchNorm1d(256),
            
            nn.ConvTranspose1d(256, 256, 3, stride=1),
            nn.ReLU(),
            nn.BatchNorm1d(256),
            
            # Printlayer(),
            nn.ConvTranspose1d(256, 1, 1, ),
            # Printlayer(),
        )
            
            '''
# -


