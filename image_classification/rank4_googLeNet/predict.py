
import torch
from model import GoogLeNet
import os
from torchvision import transforms  
from PIL import Image 
import matplotlib.pyplot as plt
import json

def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    data_transform = transforms.Compose([
        transforms.Resize((224,224)),
        transforms.ToTensor(),
        transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))])
    
    #load image
    img_path = "./tulip.png"
    assert os.path.exists(img_path), "file: '{}' does not exist.".format(img_path)
    img = Image.open(img_path).convert("RGB")
    plt.imshow(img)
    # [N, C, H, W]
    #预处理测试数据
    img = data_transform(img)
    img = torch.unsqueeze(img,dim=0) #增加一个维度batch
    
    #read class_indict
    json_path = "./class_indices.json"
    assert os.path.exists(json_path),"file:'{}' dose not exist.".format(json_path)
    with open(json_path,"r") as f:
        class_indict = json.load(f)
    
    # create model
    model = GoogLeNet(num_classes=5,aux_logits=False).to(device)
    
    #加载权重文件
    weight_path = "./GoogLeNet.pth"
    assert os.path.exists(weight_path),"file:'{}' dose not exist.".format(weight_path)
    missing_keys,unexpected_keys = model.load_state_dict(torch.load(weight_path,map_location=device),
                                                         strict=False)
    
    model.eval()
    with torch.no_grad():
        output = torch.squeeze(model(img.to(device))).cpu()

        predict = torch.softmax(output,dim=0)    
        print_cla = torch.argmax(predict).numpy()
    
    print_res = "class:{} prob:{:.3}".format(class_indict[str(print_cla)],
                                             predict[print_cla].numpy())
    
    plt.title(print_res)
    for i in range(len(predict)):
        print("class:{:10} prob:{:.3}".format(class_indict[str(i)],
                                              predict[i].numpy()))
    
    plt.show()

if __name__ == '__main__':
    main()
        
    
    
    
    
    
    