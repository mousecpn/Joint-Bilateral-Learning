import os
from PIL import Image
import torch.nn.functional as F

import torch
from torchvision import transforms
from torch.utils.data import Dataset
import random

def resize(image, size):
    image = F.interpolate(image.unsqueeze(0), size=size, mode="nearest").squeeze(0)
    return image

class JBLDataset(Dataset):
    def __init__(self,cont_img_path,style_img_path,img_size):
        self.cont_img_path = cont_img_path
        self.style_img_path = style_img_path
        self.img_size = img_size
        self.cont_img_files = self.list_files(self.cont_img_path)
        self.style_img_files = self.list_files(self.style_img_path)
        self.transform = transforms.Compose([
            transforms.Resize((self.img_size,self.img_size), Image.BICUBIC),
            transforms.ToTensor()
        ])

    def __len__(self):
        return len(self.cont_img_files)

    def __getitem__(self,idx):
        cont_img = Image.open(self.cont_img_files[idx]).convert('RGB')
        style_idx = random.randint(0,len(self.style_img_files) - 1)
        style_img = Image.open(self.style_img_files[style_idx]).convert('RGB')
        cont_img = self.transform(cont_img)
        style_img = self.transform(style_img)
        low_cont = resize(cont_img,cont_img.shape[-1]//2)
        low_style = resize(style_img, style_img.shape[-1]//2)

        return low_cont, cont_img,style_img,low_style


    def list_files(self, in_path):
        files = []
        for (dirpath, dirnames, filenames) in os.walk(in_path):
            files.extend(filenames)
            break
        files = sorted([os.path.join(in_path, x) for x in files])
        return files

