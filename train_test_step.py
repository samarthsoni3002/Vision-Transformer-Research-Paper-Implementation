import torch
import torch.nn as nn
from utils import accuracy_fn


def train_step(model: torch.nn.Module,
               train_dataloader: torch.utils.data.DataLoader,
               loss_fn: torch.nn.Module,
               optimizer: torch.optim.Optimizer,
               device : torch.device,
               ):
    
    
    epoch_train_loss = 0.0
    epoch_train_acc = 0.0
    
    for X,y in train_dataloader:

        model.train()
        X = X.to(device)
        y = y.to(device)

        y_pred = model(X)
        tr_loss = loss_fn(y_pred.squeeze(),y)
        tr_acc = accuracy_fn(y,y_pred.argmax(dim=1))
        
        epoch_train_loss+=tr_loss
        epoch_train_acc+=tr_acc

        optimizer.zero_grad()
        tr_loss.backward()
        optimizer.step()


    epoch_train_loss/=len(train_dataloader)
    epoch_train_acc/=len(train_dataloader) 
    
    return epoch_train_loss,epoch_train_acc   
    
    
def test_step(model: torch.nn.Module,
               test_dataloader: torch.utils.data.DataLoader,
               loss_fn: torch.nn.Module,
               device : torch.device,
):
    
    
    epoch_test_loss= 0.0
    epoch_test_acc = 0.0
    
    with torch.inference_mode():
        model.eval()
        for X,y in test_dataloader:
            
            X = X.to(device)
            y = y.to(device)
            y_pred = model(X)
            te_loss = loss_fn(y_pred,y)
            te_acc = accuracy_fn(y,y_pred.argmax(dim=1))

            epoch_test_loss += te_loss
            epoch_test_acc += te_acc



    epoch_test_loss = epoch_test_loss/len(test_dataloader)
    epoch_test_acc = epoch_test_acc/len(test_dataloader)

    return epoch_test_loss,epoch_test_acc


def train(epochs: int,
          model: torch.nn.Module,
          train_dataloader: torch.utils.data.DataLoader,
          test_dataloader: torch.utils.data.DataLoader,
          loss_fn: torch.nn.Module,
          optimizer: torch.optim.Optimizer,
          device : torch.device):
    
    train_loss = []
    train_acc = []
    test_loss = []
    test_acc = []
    
    for epoch in range(epochs):
        
        epoch_train_loss,epoch_train_acc = train_step(model,
                                                      train_dataloader,
                                                      loss_fn,
                                                      optimizer,
                                                      device,
                                                      )
        epoch_test_loss,epoch_test_acc = test_step(model,
                                                   test_dataloader,
                                                   loss_fn,
                                                   device
                                                   )
        train_loss.append(epoch_train_loss)
        train_acc.append(epoch_train_acc)
        test_loss.append(epoch_test_loss)
        test_acc.append(epoch_test_acc)
        
        print(
          f"Epoch: {epoch+1} | "
          f"train_loss: {epoch_train_loss:.4f} | "
          f"train_acc: {epoch_train_acc:.4f} | "
          f"test_loss: {epoch_test_loss:.4f} | "
          f"test_acc: {epoch_test_acc:.4f}"
        )
        
    return train_loss, train_acc, test_loss, test_acc