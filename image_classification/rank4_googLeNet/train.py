import os
import sys
import json
import torch
import torch.nn as nn
from torchvision import transforms,datasets
import torch.optim as optim
from tqdm import tqdm
from model import GoogLeNet

def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("using {} device.".format(device))
    
    data_transform = {
        "train": transforms.Compose([transforms.RandomResizedCrop(224),
                                     transforms.RandomHorizontalFlip(),
                                     transforms.ToTensor(),
                                     transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))]),
        "val": transforms.Compose([transforms.Resize((224,224)),
                                   transforms.ToTensor(),
                                   transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))])}
    
    data_root = os.path.abspath(os.path.join(os.getcwd(),"../.."))
    image_path = os.path.join(data_root,"data_set","flower_data")
    print(data_root)
    print(image_path)
    
    assert os.path.exists(image_path) ,"{} path does not exits".format(image_path)
    
    train_dataset = datasets.ImageFolder(root = os.path.join(image_path,"train"),
                                         transform = data_transform["train"])
    train_num = len(train_dataset)
    val_dataset = datasets.ImageFolder(root=os.path.join(image_path,"val"),
                                       transform=data_transform["val"])
    val_num = len(val_dataset)
    print("using {} images for training,{} images for validation.".format(train_num,val_num))
    
    flower_list = train_dataset.class_to_idx
    cla_dict = dict((val,key) for key ,val in flower_list.items())
    json_str = json.dumps(cla_dict,indent=4) #indent=4，格式化json数据,会按照4个空格进行缩进  
    with open("class_indices.json","w") as json_file:
        json_file.write(json_str)
        
    
    batch_size=32
    train_loader = torch.utils.data.DataLoader(train_dataset,
                                               batch_size=batch_size,
                                               shuffle=True,
                                               num_workers=0)
    validate_loader = torch.utils.data.DataLoader(val_dataset,
                                                  batch_size=batch_size,
                                                  shuffle=False,
                                                  num_workers=0)
    
    net = GoogLeNet(num_classes=5,aux_logits=True,init_weights=True)
    net.to(device)
    loss_function = nn.CrossEntropyLoss()
    optimizer = optim.Adam(net.parameters(),lr=0.0003)
    
    epochs = 5
    best_acc = 0.0
    save_path = "./GoogLeNet.pth"
    train_steps = len(train_loader)
    
    # 训练阶段
    for epoch in range(epochs):
        net.train()
        running_loss = 0.0
        train_bar = tqdm(train_loader,file=sys.stdout)    
        for _,data in enumerate(train_bar):
            images,labels = data
            optimizer.zero_grad()
            logits,aux_logits1,aux_logits2 = net(images.to(device)) #共计三个输出
            loss0 = loss_function(logits,labels.to(device))
            loss1 = loss_function(aux_logits1,labels.to(device))
            loss2 = loss_function(aux_logits2,labels.to(device))
            loss = loss0 + loss1*0.5 + loss2*0.5
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            train_bar.desc = "train epoch[{}/{}] loss:{:.3f}".format(epoch+1,epochs,loss)
            
        #验证阶段
        net.eval()
        acc = 0.0
        with torch.no_grad():
            val_bar = tqdm(validate_loader,file=sys.stdout)
            for val_data in val_bar:
                val_images,val_labels = val_data
                outputs = net(val_images.to(device)) 
                predict_y = torch.max(outputs,dim=1)[1] #按行取最大,然后返回索引
                # 先用索引判断是否相等，得到true或者false,然后在进行加和,最后求得预测正确的数量
                acc += torch.eq(predict_y,val_labels.to(device)).sum().item() 
        val_accuracy = acc/val_num
        print("[epoch %d] train_loss:%.3f  val_accuracy:%.3f" %
              (epoch+1,running_loss/train_steps,val_accuracy))

        if val_accuracy > best_acc:
            best_acc = val_accuracy
            torch.save(net.state_dict(),save_path)
    
    print("Finished Training")

if __name__ == '__main__':
    main()
                
                    
                
            