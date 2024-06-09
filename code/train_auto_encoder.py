from tqdm import tqdm
import torch
import torch.nn as nn
from torch.optim import Adam, lr_scheduler
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset
import matplotlib.pyplot as plt
from torch.utils.tensorboard import SummaryWriter
from PIL import Image

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
cuda = True
device = torch.device("cuda" if cuda else "cpu")

load_models = False
LEARNING_RATE = 0.001
GAMMA = 0.8
VALID_BATCH_SIZE = 32
BATCH_SIZE = 32
IMAGE_SIZE = 224
CHANNELS_IMG = 3
num_epochs = 40

LAMBDA_MSE, LAMBDA_VGG = 1.0, 0.0

common_trans = transforms.Compose(
    [
        transforms.Resize((IMAGE_SIZE,IMAGE_SIZE)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.ToTensor(),
        # transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ]
)

inv_transform = transforms.Compose([
    # transforms.Normalize(mean=[-0.5 / 0.5, -0.5 / 0.5, -0.5 / 0.5],std=[1 / 0.5, 1 / 0.5, 1 / 0.5]),
    transforms.ToPILImage()]
)

class ImageData(Dataset):
    def __init__(self, data_list):
        self.data_list = data_list
    def __len__(self):
          return len(self.data_list)

    def __getitem__(self, index):
        real_img = common_trans(Image.open(self.data_list[index]))
        return real_img

    
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

enc = ConvAutoencoder(3, 500, features = [32, 64, 128, 256], act_fxn = nn.LeakyReLU(0.2)).to(device)
opt_enc = Adam(enc.parameters(), lr=LEARNING_RATE)

print(enc)

if load_models:
    load(enc, 'best')

lr_scheduler_opt_enc = lr_scheduler.MultiStepLR(opt_enc, milestones=[num_epochs//2], gamma=GAMMA)

mse_loss = nn.MSELoss().to(device)
vgg_loss = VGG19_PercepLoss().to(device)

fig, axes = plt.subplots(4, 2)

def plot_valid(writer = None, writer_step = 0):
    with torch.no_grad():
        encoded = enc.encode(real_test_batch)
        skip = enc.skip_encoded(encoded)
        reconstructed = enc.decode(skip)

        valid_loss = mse_loss(real_test_batch, reconstructed)
    for j in range(4):
        axes[j, 0].imshow(inv_transform(real_test_batch[j].cpu()))  # Move image back to CPU for visualization
        axes[j, 0].axis('off')
        axes[j, 1].imshow(inv_transform(reconstructed[j].cpu()))  # Move image back to CPU for visualization
        axes[j, 1].axis('off')
    plt.pause(0.1)

    if writer:
        writer
        writer.add_images('Reconstructed', reconstructed, writer_step)
        writer.add_scalar('valid_loss', valid_loss, writer_step)

    return valid_loss

min_valid_loss = 1e5
valid_loss = 1e5

writer = SummaryWriter('logs')
writer_step = 0
writer.add_images('REAL IMG', real_test_batch, 0)

for epoch in range(num_epochs):
    train_bar = tqdm(train_loader)

    batch_sizes = 0
    valid_loss = plot_valid(writer, writer_step)
    running_results = {'loss': torch.zeros_like(valid_loss), 'mse_loss': torch.zeros_like(valid_loss), 'vgg_loss':torch.zeros_like(valid_loss)}
    if valid_loss < min_valid_loss:
        save(enc, 'best_leaky')
        print('## Saving best model with valid loss:', valid_loss)
        
        min_valid_loss = valid_loss
    for i, (real) in enumerate(train_bar):
        batch_size = real.size(0)
        batch_sizes += batch_size
        imgs_good_gt = real.to(device)
        # Training Encoder
        output, encoder_skip = enc(imgs_good_gt)

        loss_mse = mse_loss(output, imgs_good_gt)
        if LAMBDA_VGG > 0:
            loss_con = vgg_loss(output, imgs_good_gt)
        else:
            loss_con = torch.zeros_like(loss_mse)

        loss = LAMBDA_MSE * loss_mse + LAMBDA_VGG * loss_con
        opt_enc.zero_grad()
        loss.backward()
        opt_enc.step()

        running_results['loss'] += loss * batch_size
        running_results['mse_loss'] += loss_mse * batch_size
        running_results['vgg_loss'] += loss_con * batch_size

        writer.add_scalar('loss', loss, writer_step)
        writer.add_scalar('mse_loss', loss_mse, writer_step)
        writer.add_scalar('vgg_loss', loss_con, writer_step)

        description = '[%d/%d] ' % (epoch, num_epochs)
        for key in running_results:
            description += f'{key}: ' + '%.4f ' % (running_results[key].item()/batch_sizes)
        writer_step += 1
        
        train_bar.set_description(description)
        train_bar.set_postfix_str('MSE_Valid: %.4f' % (valid_loss.item()))



    lr_scheduler_opt_enc.step()
    print('# New LR ENC:', lr_scheduler_opt_enc.get_last_lr())

save(enc,'last_leaky')
plt.show()