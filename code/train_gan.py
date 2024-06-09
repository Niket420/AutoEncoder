from tqdm import tqdm
import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.optim import Adam, lr_scheduler
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset
import matplotlib.pyplot as plt
from PIL import Image
from torch.utils.tensorboard import SummaryWriter


import glob
import os
import random
import numpy as np

from utils import *
from model import *

random.seed(input('Seed :'))

def save(model, postfix = ""):
    torch.save(model.state_dict(), f'saves/{model._get_name()}_model{postfix}.pth')

def load(model, postfix = ""):
    print('## Loading:',model._get_name())
    model.load_state_dict(torch.load(f'saves/{model._get_name()}_model{postfix}.pth'))

load_models = False
cuda = True
device = torch.device("cuda" if cuda else "cpu")

LEARNING_RATE = 3e-4
VALID_BATCH_SIZE = 8
BATCH_SIZE = 8
IMAGE_SIZE = 224
CHANNELS_IMG = 3

common_trans = transforms.Compose(
    [
        transforms.Resize((IMAGE_SIZE,IMAGE_SIZE)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.ToTensor(),
        # transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        # transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ]
)

enc = ConvAutoencoder(3, 500, features = [32, 64, 128, 256], act_fxn = nn.LeakyReLU(0.2))
load(enc, 'best_leaky')
enc.to(device)

class ImageData(Dataset):
    def __init__(self, data_list):
        self.data_list = data_list
    def __len__(self):
          return len(self.data_list)

    def __getitem__(self, index):
        real_img = common_trans(Image.open(self.data_list[index]))
        return real_img

def write_valid(writer, writer_step):
    with torch.no_grad():
        encoded = enc.encode(real_test_batch)
        encoder_skip = enc.skip_encoded(encoded)
        decoded = enc.decode(encoder_skip)
        reconstructed = gen(decoded, encoder_skip)
        valid_loss = mse_loss(real_test_batch, reconstructed)

        writer.add_images('AA REAL AA', real_test_batch, writer_step)
        writer.add_images('AutoEncoded', decoded, writer_step)
        writer.add_images('Reconstructed', reconstructed, writer_step)

        writer.add_scalar('valid_loss', valid_loss, writer_step)
    
    return valid_loss
    
folder_path = '../EUVP_Dataset'
img_list = [f for f in glob.glob(folder_path + '/**/trainA/*.jpg', recursive=True) if os.path.isfile(f)]
random.shuffle(img_list)


train_list = img_list[:-VALID_BATCH_SIZE]
valid_list = img_list[-VALID_BATCH_SIZE:]

print(f'# Train Set Size:',len(train_list))
print(f'# Test Set Size:',len(valid_list))

train_dataset = ImageData(train_list)
test_dataset = ImageData(valid_list)

train_loader = DataLoader(dataset=train_dataset, batch_size=BATCH_SIZE, shuffle=True)
test_loader  = DataLoader(dataset=test_dataset,  batch_size=VALID_BATCH_SIZE, shuffle=False)
real_test_batch = next(iter(test_loader))
real_test_batch = real_test_batch.to(device)
decoded_test_batch, encoder_skip_test_batch = enc(real_test_batch)

# Initialize the model, loss function, and optimizer

gen = GeneratorGAN(3, 3).to(device)
disc = DiscriminatorGAN().to(device)

if load_models: 
    load(gen, 'best')
    load(disc, 'best')
else:
    gen.apply(Weights_Normal)
    disc.apply(Weights_Normal)

gen.train()
disc.train()
enc.eval()

opt_gen = Adam(gen.parameters(), lr = 0.0003, betas=(0.5,0.999))
opt_disc = Adam(disc.parameters(), lr = 0.0003, betas=(0.5,0.999))

# lr_scheduler_opt_gen = lr_scheduler.MultiStepLR(opt_gen, milestones=[5], gamma=0.75)
# lr_scheduler_opt_disc = lr_scheduler.MultiStepLR(opt_disc, milestones=[5], gamma=0.75)



patch = (1, IMAGE_SIZE//16, IMAGE_SIZE//16) # 14x14 for 224x224

num_epochs = 50
train_GAN = True

Adv_cGAN = torch.nn.MSELoss()
L1_G  = torch.nn.L1Loss() # similarity loss (l1)
L_vgg = VGG19_PercepLoss() 
mse_loss = nn.MSELoss()

# see if cuda is available
if torch.cuda.is_available():
    generator = gen.cuda()
    discriminator = disc.cuda()
    Adv_cGAN.cuda()
    mse_loss = mse_loss.cuda()
    L1_G = L1_G.cuda()
    L_vgg = L_vgg.cuda()
    Tensor = torch.cuda.FloatTensor
else:
    Tensor = torch.FloatTensor

min_valid_loss = 1e5
valid_loss = 1e5

valid_all = Variable(Tensor(np.ones((BATCH_SIZE, *patch))), requires_grad=False).to(device)
fake_all = Variable(Tensor(np.zeros((BATCH_SIZE, *patch))), requires_grad=False).to(device)
# norm = transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
# inv_norm = transforms.Normalize(mean=[-0.5 / 0.5, -0.5 / 0.5, -0.5 / 0.5],std=[1 / 0.5, 1 / 0.5, 1 / 0.5])

writer = SummaryWriter('logs_GAN')
writer_step = 0

for epoch in range(num_epochs):
    train_bar = tqdm(train_loader)

    batch_sizes = 0
    valid_loss = write_valid(writer, writer_step)
    if valid_loss < min_valid_loss:
        save(gen, 'best')
        save(disc, 'best')
        min_valid_loss = valid_loss
    
    for i, (real) in enumerate(train_bar):
        batch_size = real.size(0)
        batch_sizes += batch_size
        imgs_good_gt = real.to(device)

        valid = valid_all[:batch_size]
        fake = fake_all[:batch_size]

        outputs, encoder_skip = enc(imgs_good_gt)
        imgs_distorted, encoder_skip = outputs.detach(), encoder_skip.detach()
        ## Train Discriminator
        opt_disc.zero_grad()
        imgs_fake = gen(imgs_distorted, encoder_skip)
        pred_real = disc(imgs_good_gt, imgs_distorted)
        loss_real = Adv_cGAN(pred_real, valid)
        pred_fake = disc(imgs_fake, imgs_distorted)
        loss_fake = Adv_cGAN(pred_fake, fake)
        loss_D = 0.5 * (loss_real + loss_fake) * 10.0
        loss_D.backward()
        opt_disc.step()

        ## Train Generator
        opt_gen.zero_grad()
        imgs_fake = gen(imgs_distorted, encoder_skip)
        pred_fake = disc(imgs_fake, imgs_distorted)
        loss_GAN =  Adv_cGAN(pred_fake, valid) # GAN loss
        loss_1 = L1_G(imgs_fake, imgs_good_gt) # similarity loss
        loss_con = L_vgg(imgs_fake, imgs_good_gt)


        loss_G = loss_GAN + (7 * loss_1) + (3 * loss_con)
        loss_G.backward()
        opt_gen.step()

        if (i+1) % 50 == 0:
            valid_loss = write_valid(writer, writer_step)

        writer.add_scalar('loss_D', loss_D, writer_step)
        writer.add_scalar('loss_GAN', loss_GAN, writer_step)
        writer.add_scalar('loss_1', loss_1, writer_step)
        writer.add_scalar('loss_con', loss_con, writer_step)
        writer.add_scalar('loss_G', loss_G, writer_step)
        writer_step += 1
        description = '[%d/%d] ' % (epoch, num_epochs)

        train_bar.set_description(description)
        train_bar.set_postfix_str('MSE_Valid: %.4f' % (valid_loss))

    # lr_scheduler_opt_disc.step()
    # lr_scheduler_opt_gen.step()
    # print('# New LR DISC:', lr_scheduler_opt_disc.get_last_lr())
    # print('# New LR gen:', lr_scheduler_opt_gen.get_last_lr())
    
save(gen, 'last')
save(disc, 'last')
