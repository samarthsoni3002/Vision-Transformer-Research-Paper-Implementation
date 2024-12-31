import torch
from torch.utils.data import Dataset,DataLoader,random_split
from dataset_creator import Dog_breed_custom_data



def create_dataloader(images_directory: str,
                      batch_size: int,
                      data_transforms: str,
                      num_workers: int):
    
    
    image_dataset = Dog_breed_custom_data(images_directory,data_transforms)

    test_size = int(0.2*len(image_dataset))
    train_size = len(image_dataset) - test_size
    train_dataset,test_dataset = random_split(image_dataset,[train_size,test_size])
    
    train_dataloader = DataLoader(dataset=train_dataset,batch_size=batch_size,num_workers=num_workers,shuffle=True,drop_last=True)
    test_dataloader = DataLoader(dataset=test_dataset,batch_size=batch_size,num_workers=num_workers,shuffle=False,drop_last=True)
    
    
    return train_dataloader,test_dataloader