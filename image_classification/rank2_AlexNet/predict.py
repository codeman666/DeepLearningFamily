import os
import json

import torch
from PIL import Image
from torchvision import transforms
import matplotlib.pyplot as plt
from model import AlexNet


def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    data_transform = transforms.Compose(
        [transforms.Resize((224, 224)),
         transforms.ToTensor(),
         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    img_path = "tulip.png"
    img = Image.open(img_path).convert("RGB")
    plt.imshow(img)
    img = data_transform(img)
    img = torch.unsqueeze(img, dim=0) #在第0维度上增加一个batch维度


    json_path = 'class_indices.json'
    with open(json_path, "r") as f:
        class_indict = json.load(f) #json转换为字典

    model = AlexNet(num_classes=5).to(device)
    weights_path = "AlexNet.pth"
    model.load_state_dict(torch.load(weights_path))

    model.eval() #开启评估模式
    with torch.no_grad():       
        output = torch.squeeze(model(img.to(device))).cpu()  #去掉维度=1的维度
        predict = torch.softmax(output, dim=0) # 用softmax 改为概率分布,一维计算
        predict_cla = torch.argmax(predict).numpy()

    print_res = "class: {}   prob: {:.3}".format(class_indict[str(predict_cla)],                                             predict[predict_cla].numpy())
    plt.title(print_res)
    
    for i in range(len(predict)):
        print("class: {:10}   prob: {:.3}".format(class_indict[str(i)],
                                                  predict[i].numpy()))
    plt.show()


if __name__ == '__main__':
    main()