import os
import sys
import torch
import sys
import json
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from tqdm import tqdm
from model import shufflenet_v2_x1_0

def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("using {} device.".format(device))
    
    batch_size = 32
    epochs = 3
    
    data_transform = {
        "train": transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])]),
        "val": transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])])
        }
    data_root = os.path.abspath(os.path.join(os.getcwd(),"../.."))
    image_path = os.path.join(data_root,"data_set","flower_data")
    assert os.path.exists(image_path)
    train_dataset = datasets.ImageFolder(root = os.path.join(image_path,"train"),
                                         transform = data_transform["train"])
    train_num = len(train_dataset)
    
    flower_list = train_dataset.class_to_idx
    cla_dict = dict((val,key) for key ,val in flower_list.items())
    
    json_str = json.dumps(cla_dict,indent=4)
    with open("class_indices.json","w") as json_file:
        json_file.write(json_str)
        
    train_loader = torch.utils.data.DataLoader(train_dataset,
                                               batch_size=batch_size,shuffle=True,
                                               num_workers=0)
    val_dataset = datasets.ImageFolder(root=os.path.join(image_path,"val"),
                                       transform=data_transform["val"])
    val_num = len(val_dataset)
    val_loader = torch.utils.data.DataLoader(val_dataset,
                                             batch_size=batch_size,
                                             shuffle=False,
                                             num_workers=0)
    print("using {} images for training,{} images for validation.".format(train_num,val_num))
    
    net = shufflenet_v2_x1_0(num_classes=5)
    
    #"https://download.pytorch.org/models/shufflenetv2_x1-5666bf0f80.pth"
    model_weight_path = "./shufflenetv2_x1.pth"
    assert os.path.exists(model_weight_path), "file {} does not exist.".format(model_weight_path)
    
    pretrained_dict = torch.load(model_weight_path, map_location=device)
    
    # net.state_dict()[k].numel() 计算的是预训练模型中的权重总数，v.numel() 是下载的权重对应的总数，
    # 因为shufflenet的fc层只有一层，只需要对比总数就可以
    pretrained_dict = {k: v for k, v in pretrained_dict.items() if net.state_dict()[k].numel() == v.numel()}
    net.load_state_dict(pretrained_dict,strict=False)  

    # 冻结卷积层
    for name,param in net.named_parameters():
        if "fc" not in name:
            param.requires_grad_(False)
   
        
    net.to(device)
    loss_function = nn.CrossEntropyLoss()
    params = [p for p in net.parameters() if p.requires_grad]
    optimizer = optim.Adam(params,lr=0.001)
    
    best_acc = 0.0
    save_path = "./shufflenetv2_x1.pth"
    train_steps = len(train_loader)
    for epoch in range(epochs):
        net.train()
        running_loss = 0.0
        train_bar = tqdm(train_loader,file=sys.stdout)
        for step,data in enumerate(train_bar):
            images,labels = data
            optimizer.zero_grad()
            logits = net(images.to(device))
            loss = loss_function(logits,labels.to(device))
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            
            train_bar.desc = "train epoch[{}/{}] loss:{:.3f}".format(epoch + 1, epochs, loss)

        net.eval()
        acc = 0.0
        with torch.no_grad():
            val_bar = tqdm(val_loader,file=sys.stdout)
            for val_data in val_bar:
                val_images,val_labels = val_data
                outputs = net(val_images.to(device))
                predict_y = torch.max(outputs,dim=1)[1]
                acc += torch.eq(predict_y,val_labels.to(device)).sum().item()
                
                val_bar.desc = "val epoch[{}/{}]".format(epoch + 1, epochs)
        
        val_accurate = acc/val_num
        print('[epoch %d] train_loss:%.3f  val_accuracy:%.3f' 
              % (epoch+1,running_loss/train_steps,val_accurate))      
        if val_accurate > best_acc:
            best_acc = val_accurate
            torch.save(net.state_dict(),save_path)   
    
    print("Finished Training")
    
if __name__ == '__main__':
    main()       
    
    
        
    
    