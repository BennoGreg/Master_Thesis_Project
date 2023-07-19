
from torch import nn


class Autoencoder2DDoubleStride(nn.Module):

    def __init__(self):
        super(Autoencoder2DDoubleStride, self).__init__()
        self.encoder = nn.Sequential(

            nn.Conv2d(1, 225, (3, 3), padding=(1,0)),
            nn.ReLU(True),
            nn.BatchNorm2d(225),

            nn.Conv2d(225, 256, (3, 3), stride=(1,2), padding=(1,0)),
            nn.ReLU(True),
            nn.BatchNorm2d(256),

            nn.Conv2d(256, 128, (3, 3), stride=(1,1),padding=(1,0)),
            nn.ReLU(True),
            nn.BatchNorm2d(128),

            nn.Conv2d(128, 64, (3, 3), stride=(1,2),padding=(1,0)),
            nn.ReLU(True),
            nn.BatchNorm2d(64),

            nn.Conv2d(64, 1, (3, 3), stride=(1,1)),
            nn.ReLU(True),
            nn.BatchNorm2d(1),


        )

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(1,64, (3,3), stride=(1,1)),
            nn.ReLU(),
            nn.BatchNorm2d(64),

            nn.ConvTranspose2d(64,128, (3,3), stride=(1,2), padding=(1,0)),
            nn.ReLU(),
            nn.BatchNorm2d(128),

            nn.ConvTranspose2d(128,256, (3,3), stride=(1,1), padding=(1,0)),
            nn.ReLU(),
            nn.BatchNorm2d(256),

            nn.ConvTranspose2d(256,225, (3,5),stride=(1,2), padding=(1,0)),
            nn.ReLU(),
            nn.BatchNorm2d(225),

            nn.ConvTranspose2d(225,1, (1,1), stride=(1,1)),

        )


    def forward(self, x):
        x = self.encode(x)
        return self.decode(x)

    def encode(self,x):
        x = self.encoder(x)
        return x

    def decode(self,x):
        x = self.decoder(x)
        return x


''' other configuration tried out

    def __init__(self):
        super(AutoStyleEncoderNetwork, self).__init__()
        self.encoder = nn.Sequential(
            # Printlayer(),
            nn.Conv2d(1, 225, (3, 3), padding=(1, 0)),
            nn.ReLU(True),
            nn.BatchNorm2d(225),
            # Printlayer(),
            nn.Conv2d(225, 256, (3, 3), stride=(1, 2), padding=(1, 0)),
            nn.ReLU(True),
            nn.BatchNorm2d(256),
            # Printlayer(),
            # nn.Conv2d(256, 256, (3, 3), stride=(1,2),padding=(1,0)),
            # nn.ReLU(True),
            # nn.BatchNorm2d(256),
            # Printlayer(),
            nn.Conv2d(256, 128, (3, 3), stride=(1, 2), padding=(1, 0)),
            nn.ReLU(True),
            nn.BatchNorm2d(128),

            # nn.Conv2d(128, 128, (3, 3), stride=(1,1),padding=(1,0)),
            # nn.ReLU(True),
            # nn.BatchNorm2d(128),
            # Printlayer(),

            nn.Conv2d(128, 64, (3, 3), stride=(1, 1), padding=(1, 0)),
            nn.ReLU(True),
            nn.BatchNorm2d(64),

            # nn.Conv2d(64, 64, (3, 3), stride=(1,1),padding=(1,0)),
            # nn.ReLU(True),
            # nn.BatchNorm2d(64),

            nn.Conv2d(64, 1, (3, 3), stride=(1, 1)),
            nn.ReLU(True),
            nn.BatchNorm2d(1),
            #Printlayer(),
        )

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(1, 64, (3, 3), stride=(1, 1)),
            nn.ReLU(),
            nn.BatchNorm2d(64),

            # nn.ConvTranspose2d(64,64, (3,3), stride=(1,1), padding=(1,0)),
            # nn.ReLU(),
            # nn.BatchNorm2d(64),

            nn.ConvTranspose2d(64, 128, (3, 3), stride=(1, 1), padding=(1, 0)),
            nn.ReLU(),
            nn.BatchNorm2d(128),

            # nn.ConvTranspose2d(128,128, (3,3), stride=(1,1), padding=(1,0)),
            # nn.ReLU(),
            # nn.BatchNorm2d(128),

            # Printlayer(),
            nn.ConvTranspose2d(128, 256, (3, 3), stride=(1, 2), padding=(1, 0)),
            nn.ReLU(),
            nn.BatchNorm2d(256),
            # Printlayer(),
            # nn.ConvTranspose2d(256,256, (3,3), stride=(1,2), padding=(1,0)),
            # nn.ReLU(),
            # nn.BatchNorm2d(256),
            # Printlayer(),
            nn.ConvTranspose2d(256, 225, (3, 5), stride=(1, 2), padding=(1, 0)),
            nn.ReLU(),
            nn.BatchNorm2d(225),
            # Printlayer(),
            nn.ConvTranspose2d(225, 1, (1, 1), stride=(1, 1)),
            # Printlayer(),
        )

'''