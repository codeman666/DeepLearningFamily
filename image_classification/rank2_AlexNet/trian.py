import os
import sys
import json

import torch
import torch.nn as nn
from torchvision import transforms,datasets,utils
import matplotlib.pyplot as plt
import numpy as np
import torch.optim as optim
from tqdm import tqdm
from model import AlexNet

def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")  
      
    data_transform = {
        "train": transforms.Compose([
            transforms.RandomResizedCrop(224),  #训练的时候需要数据增强的数据 [论文创新点]
            transforms.RandomHorizontalFlip(),  #水平翻转 [论文创新点]
            transforms.ToTensor(),
            transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))
        ]),
        "test":transforms.Compose([
            transforms.Resize((224,224)),
            transforms.ToTensor(),
            transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))
        ])
    }

    
    image_path = 'data_set/flower_data'
    
    #返回二元组(image,label)
    train_dataset = datasets.ImageFolder(root=os.path.join(image_path,"train"),
                                         transform=data_transform["train"])
    flower_list = train_dataset.class_to_idx 
    cla_dict = dict((value,key) for key,value in flower_list.items())
    json_str = json.dumps(cla_dict,indent=4)  #由字典转为json格式的字符串,并首行缩进
    with open("class_indices.json","w") as json_file:
        json_file.write(json_str)
    
    
    batch_size = 32
    train_loader = torch.utils.data.DataLoader(train_dataset,
                                               batch_size=batch_size,
                                               shuffle=True,
                                               num_workers=0)
    train_num = len(train_dataset)
    test_dataset = datasets.ImageFolder(root=os.path.join(image_path,"test"),
                                        transform=data_transform["test"])
    test_num = len(test_dataset)
    test_loader = torch.utils.data.DataLoader(test_dataset,
                                              batch_size=4,
                                              shuffle=False,
                                              num_workers=0)
    print(f"using {train_num} images for training,{test_num} images for test")
    
    net = AlexNet(num_classes=5)
    net.to(device)
    loss_function = nn.CrossEntropyLoss()
    optimizer = optim.Adam(net.parameters(),lr=0.001)
    
    epochs = 10
    save_path = "./AlexNet.pth"
    best_accurate = 0.0
    train_steps = len(train_loader)
    for epoch in range(epochs):
        net.train() # 训练模式 主要影响模型中的dropout
        running_loss = 0.0
        #train_loader = tqdm(train_loader, file = sys.stdout)  # 显示训练的进度条
        for _,data in enumerate(train_loader):
            images,labels = data
            optimizer.zero_grad()
            outputs = net(images.to(device))
            loss = loss_function(outputs,labels.to(device))
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            
            #print(f"train epoch[{epoch+1}/{epochs}] loss:{loss:.3f}")
            
        net.eval() # 评估模式 主要影响模型中的dropout
        acc = 0.0
        with torch.no_grad():
            test_loader = tqdm(test_loader,file=sys.stdout)
            for test_data in test_loader:
                test_image,test_labels = test_data
                outputs = net(test_image.to(device))
                predict_y = torch.argmax(outputs,dim=1)
                acc += torch.eq(predict_y,test_labels.to(device)).sum().item()
        
            test_accurate = acc/test_num
            print("[epoch %d] train_loss: %3.f test_accuracy:%.3f" %
                (epoch+1,running_loss/train_steps,test_accurate))
        
        if test_accurate > best_accurate:  # 每一轮数据进行对比
            print(best_accurate)
            best_accurate = test_accurate
            torch.save(net.state_dict(),save_path)
    print("Finished Training")
        
  
if __name__ == "__main__":
    main()    