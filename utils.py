import os
import matplotlib.pyplot as plt
from pathlib import Path
import random
from PIL import Image


import torch
from torchvision import transforms,models
from torch.utils.data import Dataset,DataLoader,random_split
from torchinfo import summary


def walk_through_directory(directory):
    
    for dirpaths,dirnames,filenames in os.walk(directory):
        print(f"There are {len(dirnames)} directories, {len(filenames)} filenames in {dirpaths}")




def show_images(images_directory: str, n: int,transform=None):
    

    image_paths = list(images_directory.glob("*/*.jpg"))
    random_images = random.sample(image_paths, k=n)

    for path_img in random_images:
        img = Image.open(path_img)
        fig,ax = plt.subplots(1,2,figsize=(5,5))
        ax[0].imshow(img)
      
        ax[0].set_title("Original")
        ax[0].axis("off")

        if transform:
            transformed_image = transform(img).permute(1,2,0)
            ax[1].imshow(transformed_image)
            ax[1].set_title("Transformed")
            ax[1].axis("off")
    


def get_class_names(images_directory:str):
    
    class_names = sorted([classes.name for classes in os.scandir(images_directory)])
    filtered_names = [i.split("-")[1] for i in class_names]

    class_to_idx = {classes:index for index,classes in enumerate(filtered_names)}
    idx_to_class = {index:classes for index,classes in enumerate(filtered_names)}

    return filtered_names,class_to_idx,idx_to_class


def model_summary(model):
    
    summary(model=model,
        input_size=(32,3,224,224),
        col_names=["input_size","output_size","num_params","trainable"],
        col_width=20,
        row_settings=["var_names"]
    )
    
    
def accuracy_fn(y_true,y_pred):
    
    correct = torch.eq(y_true,y_pred).sum().item()
    print(y_pred)
    acc = (correct/len(y_pred))
    return acc*100


