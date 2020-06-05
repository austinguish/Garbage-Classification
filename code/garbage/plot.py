import matplotlib
import numpy as np
import matplotlib.pyplot as plt
from imageloader import image_data_loader
import torch.nn as nn
import torchvision
from functools import partial
import torch
import time

def plot_save(history,filename,epoch):
    num_epochs = epoch
    epoch_info = np.array(history)
    plt.plot(epoch_info[:, 1:2])
    plt.plot(epoch_info[:, 3:4])
    plt.legend(['Train Loss', 'Val Loss'])
    plt.xlabel('Epoch Number')
    plt.ylabel('Loss')
    plt.ylim(0, 2)
    plt.title(filename+'_loss_curve.png')
    plt.savefig("./image/"+filename+'_loss_curve.png')
    plt.show()
    
    plt.plot(epoch_info[:, 2:3])
    plt.plot(epoch_info[:, 4:5])
    plt.legend(['Train Accuracy', 'Val Accuracy'])
    plt.xlabel('Epoch Number')
    plt.ylabel('Accuracy')
    plt.ylim(0, 1)
    plt.title(filename+'_accuracy_curve.png')
    plt.savefig("./image/"+filename+'_accuracy_curve.png')
    plt.show()

