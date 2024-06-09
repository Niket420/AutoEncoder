from tqdm import tqdm
import torch
import torchvision
import torch.nn as nn
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset
import matplotlib.pyplot as plt
from PIL import Image
from matplotlib.animation import FuncAnimation

import glob
import os

from utils import *
from model import *

def save(model, postfix = ""):
    torch.save(model.state_dict(), f'saves/{model._get_name()}_model{postfix}.pth')

def load(model, postfix = ""):
    print('## Loading:',model._get_name())
    model.load_state_dict(torch.load(f'saves/{model._get_name()}_model{postfix}.pth'))

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

IMAGE_SIZE = 224
cuda = True
device = torch.device("cuda" if cuda else "cpu")

common_trans = transforms.Compose(
    [
        transforms.Resize((IMAGE_SIZE,IMAGE_SIZE)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.ToTensor(),
    ]
)

dataset = ImageData(img_list[::20])

enc = ConvAutoencoder(3, 500, features = [32, 64, 128, 256], act_fxn = nn.LeakyReLU(0.2))
gen = GeneratorGAN(3, 3)

load(enc, 'best_leaky')
load(gen, 'best')

enc.to(device)
gen.to(device)

criterion = nn.MSELoss()

enc.eval()
gen.eval()

fig, axes = plt.subplots(1, 2)
image1 = np.random.uniform(0, 1, (IMAGE_SIZE, IMAGE_SIZE,3))
image2 = np.random.uniform(0, 1, (IMAGE_SIZE, IMAGE_SIZE,3))
img_plot_real = axes[0].imshow(image1, cmap='viridis')
axes[0].set_title('Real Image')

img_plot_compressed = axes[1].imshow(image2, cmap='viridis')
axes[1].set_title('Compressed')
for ax in axes:
    ax.axis('off')

pbar = tqdm(total=dataset.__len__())
def update(index):
    images = dataset.__getitem__(index).unsqueeze(0)
    images = images.to(device)
    encoded = enc.encode(images)
    encoded_skip = enc.skip_encoded(encoded)
    decoded = enc.decode(encoded_skip)
    recon = gen(decoded, encoded_skip)

    inp = torchvision.utils.make_grid(images, normalize=True).cpu().numpy().transpose(1, 2, 0)
    out = torchvision.utils.make_grid(recon, normalize=True).cpu().numpy().transpose(1, 2, 0)
    img_plot_real.set_array(inp)
    img_plot_compressed.set_array(out)
    pbar.update()
    return [img_plot_real, img_plot_compressed]

print("## Number of Image:", dataset.__len__())
fps = 10
interval = 1000 / fps
ani = FuncAnimation(fig, update, frames=range(dataset.__len__()), interval = interval)
ani.save('demo_video.mp4', writer='ffmpeg')