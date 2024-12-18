import torch
import torchvision.transforms as transforms
from PIL import Image
from model_official import LeNet


def main():
    transform =transforms.Compose([
      # 这里只调整图片的宽度和长度,对于通道数并不进行改变  
      transforms.Resize((32,32)),
      transforms.ToTensor(),
      transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5)) 
    ])
    image = Image.open("dog.png").convert("RGB")
    image = transform(image)  
    image = torch.unsqueeze(image,dim=0) # [N,c,h,w]
    
    
    classes = ('plane', 'car', 'bird', 'cat',
            'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

    net = LeNet()
    net.load_state_dict(torch.load("Lenet.pth"))
  
    with torch.no_grad():
        outputs = net(image)
        # torch.argmax返回索引  torch.max 返回最大值和索引
        predict = torch.argmax(outputs,dim=1).item()
    
    print(classes[predict])

if __name__ == "__main__":
    main()
        
    
