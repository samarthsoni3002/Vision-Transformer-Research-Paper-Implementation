import torch
from torch.utils.data import Dataset
from PIL import Image
from utils import get_class_names
from torchvision import transforms



class Dog_breed_custom_data(Dataset):
    
    def __init__(self,directory:str, transform=None):

        self.paths = list(directory.glob("*/*.jpg"))
        self.transform = transform
        self.classes, self.class_to_idx, self.idx_to_class = get_class_names(directory)

    def load_images(self,index:int):
        image_path = self.paths[index]
        img = Image.open(image_path)
        return img.convert("RGB")

    def __len__(self):
        return len(self.paths)

    def __getitem__(self,index:int):

        img = self.load_images(index)
        class_name = self.paths[index].parent.name
        filtered_names = class_name.split("-")[1]
        class_to_idx = self.class_to_idx[filtered_names]

        if self.transform:
            return self.transform(img), class_to_idx
        else:
            return img,class_to_idx