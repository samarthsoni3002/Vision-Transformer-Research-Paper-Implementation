from data_setup import create_dataloader
from model import model_builder
from train_test_step import train
from pathlib import Path
import torch
import torch.nn as nn
from torchvision import transforms


data_directory = Path("data")
images_directory = data_directory / "Images"

device = "cuda" if torch.cuda.is_available() else "cpu"

data_transforms = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor()
])

train_dataloader,test_dataloader = create_dataloader(images_directory,
                                                     batch_size=32,
                                                     data_transforms=data_transforms,
                                                     num_workers=16)


print(len(train_dataloader))

'''

model = model_builder()
model = model.to(device)

loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

train_loss,train_acc,test_loss,test_acc = train(epochs=20,
                                                model=model,
                                                train_dataloader=train_dataloader,
                                                test_dataloader=test_dataloader,
                                                loss_fn=loss_fn,
                                                optimizer=optimizer,
                                                device=device)

'''