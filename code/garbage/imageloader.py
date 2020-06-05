import torchvision.datasets as datasets
import torchvision.transforms as transforms
import torch.utils.data
import cv2
import numpy as np
from PIL import Image


def image_data_loader(root_path, batchsize):
    img_path = root_path
# follow the preprocess on the website to create a transform object
    '''preprocess = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])'''
    preprocess = transforms.Compose([transforms.Resize(224), transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),])
    garbage_dataset = datasets.ImageFolder(root=img_path,transform=preprocess)
    print(garbage_dataset.class_to_idx)
    loader = torch.utils.data.DataLoader(dataset=garbage_dataset, shuffle=True, batch_size=batchsize, num_workers=8)
    return loader,len(garbage_dataset)
