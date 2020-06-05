import torchvision.models as models
import torch.optim as optim
from Train import trainmodel
from alexnet import *
from vgg import *
from densenet import *
from mobilenet import *
from ResNet import *
batch_size = [4,8,12,24,36]
device = 6
model_name_list = ["alexnet","vgg11_bn","vgg19_bn","mobilenet_v2",\
                   "densenet161","densenet201","resnet18","resnet101"]
model_list = [alexnet(pretrained=False),vgg19_bn(pretrained=False),\
              vgg11_bn(pretrained=False),mobilenet_v2(pretrained=False),\
              densenet161(pretrained=False),densenet201(pretrained=False),\
              resnet18(3,6),resnet101(3,6)]
lossfunc = torch.nn.CrossEntropyLoss().cuda(device)
for size in batch_size:
    i = 0
    for models in model_list:
        file_name = model_name_list[i]+str(batch_size)+"rmsprop"
        optimizer = optim.rmsprop(models.parameters())
        trainmodel(model_name=models,loss = lossfunc,batchsize=batch_size,opti=optimizer,\
                   epoch_num=50,device_num=device,filename=file_name)
        i+=1



