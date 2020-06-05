import torch.nn as nn
import torchvision
from functools import partial
import torch
import time
import matplotlib
import numpy as np
import matplotlib.pyplot as plt
from plot import *
train_folder = "../garbage_dataset/train_set"
valid_folder = "../garbage_dataset/valid_set"


def trainmodel(model_name, loss, batchsize, opti, epoch_num, device_num,filename):
    model = model_name
    batch_size = batchsize
    epoch = epoch_num
    device = device_num
    model1 = model.cuda(device)
    optimizer = opti
    lossfunc = loss.cuda(device)
    trainset,size1=image_data_loader(train_folder,batch_size)
    validset,size2=image_data_loader(valid_folder,batch_size)
    train_data_size = size1
    valid_data_size = size2
    print(train_data_size,valid_data_size)

    epoch_info = []
    for x in range(epoch):
        epoch_start = time.time()
        print("Epoch: {}".format(x + 1))
        train_loss = 0.0
        train_acc = 0.0
        valid_loss = 0.0
        valid_acc = 0.0
        for i, data in enumerate(trainset):
            optimizer.zero_grad()
            (inputs, labels) = data
            inputs = torch.autograd.Variable(inputs).cuda(device)
            labels = torch.autograd.Variable(labels).cuda(device)
            outputs = model1(inputs)
            outputs.cuda(device)
            loss = lossfunc(outputs, labels)
            train_loss += loss.item() * inputs.size(0)
            ret, predictions = torch.max(outputs.data, 1)
            correct_counts = predictions.eq(labels.data.view_as(predictions))
            acc = torch.mean(correct_counts.type(torch.FloatTensor))
            train_acc += acc.item() * inputs.size(0)
            loss.backward()
            optimizer.step()

        with torch.no_grad():
            model1.eval()
            for i, data in enumerate(validset):
                (inputs, labels) = data
                inputs = inputs.cuda(device)
                labels = labels.cuda(device)
                outputs = model1(inputs)
                loss = lossfunc(outputs, labels)
                valid_loss += loss.item() * inputs.size(0)
                ret, predictions = torch.max(outputs.data, 1)
                correct_counts = predictions.eq(labels.data.view_as(predictions))
                acc = torch.mean(correct_counts.type(torch.FloatTensor))
                valid_acc += acc.item() * inputs.size(0)

        avg_train_loss = train_loss / train_data_size
        avg_train_acc = train_acc / train_data_size
        avg_valid_loss = valid_loss / valid_data_size
        avg_valid_acc = valid_acc / valid_data_size
        epoch_end = time.time()
        epoch_info.append([epoch, avg_train_loss, avg_train_acc, avg_valid_loss, avg_valid_acc, epoch_start, epoch_end])
        print("Epoch: {:03d}, Training: Loss: {:.4f}, Accuracy: {:.4f}%,Valid: Loss: {:.4f}, Accuracy: {:.4f}%,"
              "Time: {:.4f}s".format(
                x + 1, avg_train_loss, avg_train_acc * 100, avg_valid_loss, avg_valid_acc * 100,
                epoch_end - epoch_start))            
    torch.save(model1.state_dict(), "../param/"+filename+str(x))        
    torch.save(epoch_info,"../history/"+filename+"info.pt")
    plot_save(epoch_info,filename,epoch)
    torch.cuda.empty_cache() #清除显存




