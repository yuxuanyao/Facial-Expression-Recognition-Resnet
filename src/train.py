"""
Training function

inputs:
    model = PyTorch nn
    batch_size = batch size of the training and validation loaders
    train_loader, val_loader = training and validation dataloader containing features of the imgs
    num_epochs = # of epochs
    lr = learning rate
    
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.models as models
import os
import numpy as np
import pandas as pd
from PIL import Image
import matplotlib.pyplot as plt
import time
import random

import util

use_cuda = True # Use cuda

def get_loss(model, data_loader, criterion):
    total_loss = 0
    total_epoch = 0
    model.eval()
    for i, data in enumerate(data_loader, 0):
        images, labels = data
        # Use cuda if available
        if use_cuda and torch.cuda.is_available():
            images = images.cuda()
            labels = labels.cuda()
        images = images.float()       # Convert to float from double
        output = model(images)
        
        # select index with maximum prediction score
        loss = criterion(output, labels)
        total_loss += loss.item()
        total_epoch += len(labels)
    loss = float(total_loss) / (i + 1)
    return loss

"""
Get accuracy function for the entire dataset

"""
def get_accuracy(model, data_loader):
    correct = 0
    total = 0
    model.eval()
    for images, labels in data_loader:
        # Use cuda if available
        if use_cuda and torch.cuda.is_available():
            images = images.cuda()
            labels = labels.cuda()
        images = images.float()       # Convert to float from double
        output = model(images)
        
        # select index with maximum prediction score
        pred = output.max(1, keepdim=True)[1]
        correct += pred.eq(labels.view_as(pred)).sum().item()
        total += images.shape[0]    # Increment total by the number of samples
    return correct / total


"""
Get accuracy function for the entire dataset

"""
def get_class_accuracy(model, data_loader):
    correct = [0, 0, 0, 0, 0, 0]
    total = [0, 0, 0, 0, 0, 0]
    model.eval()
    for images, labels in data_loader:
        # Use cuda if available
        if use_cuda and torch.cuda.is_available():
            images = images.cuda()
            labels = labels.cuda()
        images = images.float()                             # Convert to float from double
        output = model(images)
        pred = output.max(1, keepdim=True)[1].squeeze()     # select index with maximum prediction score
        
        # Iterate through all predicted values
        for i in range(labels.shape[0]):
            if(labels[i] == pred[i]): 
                correct[labels[i]] += 1
            total[labels[i]] += 1
        
    # Compute accuracy 
    class_accuracy = []
    for i in range(len(total)):
        if total[i] == 0:
            class_accuracy.append(0)
        else:
            class_accuracy.append(correct[i]/total[i])
    
    # Return numpy array rounded to 3 decimals
    return np.round(np.asarray(class_accuracy), 3)

"""
Just to make sure this way of calculation works, result should equal the get accuracy function

"""
def get_handcoded_accuracy(model, data_loader):
    correct = 0
    total = 0
    model.eval()
    for images, labels in data_loader:
        # Use cuda if available
        if use_cuda and torch.cuda.is_available():
            images = images.cuda()
            labels = labels.cuda()
        images = images.float()                             # Convert to float from double
        output = model(images)
        pred = output.max(1, keepdim=True)[1].squeeze()     # select index with maximum prediction score
        
        # Iterate through all predicted values
        for i in range(labels.shape[0]):
            if(labels[i] == pred[i]): 
                correct += 1
            total += 1
        

    return correct/total


def train(model, batch_size, train_loader, val_loader, num_epochs, lr):

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr, momentum=0.9)

    iters, epochs, train_losses, val_losses, train_acc, val_acc = [], [], [], [], [], []

    # training
    n = 0 # the number of iterations
    start_time = time.time()
    
    # Iterate through epochs
    for epoch in range(num_epochs):
        mini_b=0
        mini_batch_correct = 0
        mini_batch_total = 0
        
        # For each batch
        model.train()
        for images, labels in iter(train_loader):
            
            # Use cuda if available
            if use_cuda and torch.cuda.is_available():
                images = images.cuda()
                labels = labels.cuda()
            images = images.float()       # Convert to float from double
            out = model(images)           # forward pass
            loss = criterion(out, labels) # compute the total loss
            loss.backward()               # backward pass (compute parameter updates)
            optimizer.step()              # make the updates for each parameter
            optimizer.zero_grad()         # a clean up step for PyTorch

            # Mini_batch Accuracy: We don't compute accuracy on the whole trainig set in every iteration!
            pred = out.max(1, keepdim=True)[1]
            mini_batch_correct = pred.eq(labels.view_as(pred)).sum().item()
            mini_batch_total = images.shape[0]
            

            # Save the current training information
            iters.append(n)
            train_losses.append(float(loss)/batch_size)   # compute *average* loss
            train_acc.append((mini_batch_correct / mini_batch_total))
            
            
            n += 1
            mini_b += 1
            print("Iteration: ",n,'Train acc: % 6.2f ' % train_acc[-1], 
                  "Train loss: % 6.8f " % train_losses[-1], 
                  "Time Elapsed: % 6.2f s " % (time.time()-start_time))
        
        # Compute validation accuracy at the end of each epoch to speed up training
        epochs.append(epoch)
        val_acc.append(get_accuracy(model, val_loader))
        val_losses.append(get_loss(model, val_loader, criterion))
        print ("Epoch %d Finished. " % epoch ,"Time per Epoch: % 6.2f s "% ((time.time()-start_time) / (epoch +1)))
        print("Validation acc: % 6.3f " % val_acc[-1])

    
    end_time= time.time()
    
    # Save the current model (checkpoint) to a file
    model_name = util.get_model_name("Resnet50-Pretrained-12000-Dropout", batch_size, lr, num_epochs)
    torch.save(model.state_dict(), '../torch_checkpoints/' + model_name + '.pt')
    
    # Plot Training Curves
    util.plot_training_curve(model_name, iters, epochs, train_losses, val_losses, train_acc, val_acc)
    
    # Final values
    train_acc.append(get_accuracy(model, train_loader))
    print("Final Training Accuracy: {}".format(train_acc[-1]))
    print("Final Validation Accuracy: {}".format(val_acc[-1]))
    print ("Total time:  % 6.2f s  Time per Epoch: % 6.2f s " % ( (end_time-start_time), ((end_time-start_time) / num_epochs) ))