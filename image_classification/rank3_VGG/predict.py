import os 
import json
import torch
from PIL import Image
from torchvision import transforms
import matplotlib.pyplot as plt
from model import vgg
import matplotlib
matplotlib.use('Qt5Agg')

def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    data_transform = transforms.Compose([
        transforms.Resize((224,224)),
        transforms.ToTensor(),
        transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))
    ])
  
    # 测试图片  
    img_path = "./tulip.png"  
    print(os.getcwd())
    print(img_path)
    assert os.path.exists(img_path), "file: '{}' dose not exist".format(img_path)
    
    # 对图片进行预处理
    img = Image.open(img_path).convert("RGB")
    plt.imshow(img)
    img = data_transform(img)
    img = torch.unsqueeze(img,dim=0) #增加一个维度 batch
    
    # 加载json文件为字典的形式
    json_path = "./class_indices.json"
    assert os.path.exists(json_path), "file: '{}' dose not exist.".format(json_path)
    with open(json_path,"r") as f:
        class_indict = json.load(f)
    
    #加载模型和参数文件   
    model = vgg(model_name="vgg16",num_classes=5).to(device)
    weight_path = "vgg16Net.pth"
    assert os.path.exists(weight_path), "file: '{}' dose not exist.".format(weight_path)
    model.load_state_dict(torch.load(weight_path,map_location=device),strict=False)  #strict=False忽略不匹配的参数
    
    
    model.eval()
    with torch.no_grad():
        output = torch.squeeze(model(img.to(device))).cpu() #去除维度为1的维度 model(x) 其实forward(x)
        predict = torch.softmax(output,dim=0)
        print(predict)
        predict_cla = torch.argmax(predict).numpy()
    print_res = "class: {} prob:{:.3}".format(class_indict[str(predict_cla)],
                                              predict[predict_cla].numpy())
    
    plt.title(print_res)
    for i in range(len(predict)):
        print("class:{:10} prob:{:.3}".format(class_indict[str(i)],
                                              predict[i].numpy()))
    plt.show()

if __name__ == "__main__":
    main()