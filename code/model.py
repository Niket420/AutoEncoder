import torch
import torch.nn as nn
import torch.nn.functional as F

import torchviz
from collections import OrderedDict

class Encoder(nn.Module):
    def __init__(self, in_channels, latent_space, features, act_fxn):
        super(Encoder, self).__init__()
        self.model = self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, features[0], 3, stride=2, padding=1),
            act_fxn,
            nn.Conv2d(features[0], features[1], 3, stride=2, padding=1),
            act_fxn,
            nn.Conv2d(features[1], features[2], 3, stride=2, padding=1),
            act_fxn,
            nn.Conv2d(features[2], features[3], 3, stride=2, padding=1),
            act_fxn,
            nn.Conv2d(features[3], features[3], 3, stride=2, padding=1),
            act_fxn,
            nn.Flatten(),
            nn.Linear(features[3]*7*7, latent_space),
        )

    def forward(self, x):
        return self.model(x)

class Decoder(nn.Module):
    def __init__(self, in_channels, features = [32, 64, 128, 256], act_fxn = nn.ReLU(True)):
        super(Decoder, self).__init__()
        
        self.model = nn.Sequential(
            nn.ConvTranspose2d(features[3], features[3], 5, stride=2, padding=1),  # 15x15
            act_fxn,
            nn.ConvTranspose2d(features[3], features[2], 3, stride=2, padding=1),   # 29x29
            act_fxn,
            nn.ConvTranspose2d(features[2], features[1], 3, stride=2, padding=1),   # 57x57
            act_fxn,
            nn.ConvTranspose2d(features[1], features[0], 3, stride=2, padding=1),   # 113x113
            act_fxn,
            nn.ConvTranspose2d(features[0], in_channels, 2, stride=2, padding=1),    # 224x224
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.model(x)


class ConvAutoencoder(nn.Module):
    def __init__(self, in_channels, latent_space, features = [32, 64, 128, 256], act_fxn = nn.ReLU(True)):
        super(ConvAutoencoder, self).__init__()
        self.encoder = Encoder(in_channels, latent_space, features = features, act_fxn = act_fxn)

        self.bottelneck = nn.Sequential(
            nn.Linear(latent_space, features[3]*7*7), 
            act_fxn,
            nn.Unflatten(1, (features[3], 7, 7))
        )
        self.decoder = Decoder(in_channels, features = features, act_fxn = act_fxn)

    def encode(self,x):
        return self.encoder(x)
    
    def decode(self, x):
        decoded = self.decoder(x)
        return decoded

    def skip_encoded(self ,x):
        return self.bottelneck(x)
        
    def forward(self, x):
        encoded = self.encode(x)
        skip_enc = self.skip_encoded(encoded)
        decoded = self.decode(skip_enc)
        return decoded, skip_enc
    

class UNetDown(nn.Module):
    def __init__(self, in_size, out_size, bn=True):
        super(UNetDown, self).__init__()
        layers = [nn.Conv2d(in_size, out_size, 4, 2, 1, bias=False)]
        if bn: layers.append(nn.BatchNorm2d(out_size, momentum=0.8))
        layers.append(nn.LeakyReLU(0.2))
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)


class UNetUp(nn.Module):
    def __init__(self, in_size, out_size):
        super(UNetUp, self).__init__()
        layers = [
            nn.ConvTranspose2d(in_size, out_size, 4, 2, 1, bias=False),
            nn.BatchNorm2d(out_size, momentum=0.8),
            nn.ReLU(inplace=True),
        ]
        self.model = nn.Sequential(*layers)

    def forward(self, x, skip_input):
        x = self.model(x)
        x = torch.cat((x, skip_input), 1)
        return x


class GeneratorGAN(nn.Module):
    def __init__(self, in_channels=3, out_channels=3):
        super(GeneratorGAN, self).__init__()
        self.down1 = UNetDown(in_channels, 32, bn=False)
        self.down2 = UNetDown(32, 128)
        self.down3 = UNetDown(128, 256)
        self.down4 = UNetDown(256, 256)
        self.down5 = UNetDown(256, 256, bn=False)

        self.up1 = UNetUp(256, 256)
        self.up2 = UNetUp(512, 256)
        self.up3 = UNetUp(512, 128)
        self.up4 = UNetUp(256, 32)
        self.final = nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.ZeroPad2d((1, 0, 1, 0)),
            nn.Conv2d(64, out_channels, 4, padding=1),
            nn.Tanh(),
        )


    def forward(self, x, encoder_skip):
        d1 = self.down1(x)
        d2 = self.down2(d1)
        d3 = self.down3(d2)
        d4 = self.down4(d3)
        d5 = self.down5(d4)

        u1 = self.up1(F.leaky_relu(d5 + encoder_skip, 0.2), d4)
        u2 = self.up2(u1, d3)
        u3 = self.up3(u2, d2)
        u45 = self.up4(u3, d1)
        return self.final(u45)


class DiscriminatorGAN(nn.Module):
    def __init__(self, in_channels=3):
        super(DiscriminatorGAN, self).__init__()

        def discriminator_block(in_filters, out_filters, bn=True):
            layers = [nn.Conv2d(in_filters, out_filters, 4, stride=2, padding=1)]
            if bn: layers.append(nn.BatchNorm2d(out_filters, momentum=0.8))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers

        self.model = nn.Sequential(
            *discriminator_block(in_channels*2, 32, bn=False),
            *discriminator_block(32, 64),
            *discriminator_block(64, 128),
            *discriminator_block(128, 256),
            nn.ZeroPad2d((1, 0, 1, 0)),
            nn.Conv2d(256, 1, 4, padding=1, bias=False)
        )

    def forward(self, img_A, img_B):
        img_input = torch.cat((img_A, img_B), 1)
        return self.model(img_input)
    

if __name__ == "__main__":

    Model =  ConvAutoencoder(3, 500, features = [32, 64, 128, 256], act_fxn = nn.ReLU(True))
    # print(Model)
    x = torch.randn((1, 3, 224, 224) ,requires_grad=False)
    y = Model(x)
    dot = torchviz.make_dot(y, params=dict(Model.named_parameters()))
    # Remove AccumulatedGrad and Extra nodes
    # for node in dot.body:
    #     print(node)
    #     # if not 'fillcolor=lightblue' in node:
    #     #     dot.body.remove(node)
    dot.render('Encoder_model', format='png')

    # Model = GeneratorGAN()
    # print(Model)
    # x = torch.randn((1, 3, 224, 224))
    # x = Model(x,skip)
    # print(x.shape)
    # print(f'# {Model._get_name()} parameters:', sum(param.numel() for param in Model.parameters()))