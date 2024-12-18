import torch
import torch.nn as nn
import torch.nn.functional as F

# class LeNet(nn.Module):
#     def __init__(self):
#         super().__init__() 
#         """
#         分别对应的含义: (in_channels:输入通道数, out_channels:输出通道数, 
#         kernel_size:卷积核的大小并不是数量,stride =1: 步长默认为1可以省略,padding = 0: 默认为0)
#         计算公式: 经过卷积后的矩阵尺寸大小计算公式为 N = (M - F + 2P)/S +1
#         其中 输入图片大小为MxM,Filter大小为FxF,步长s,padding 为p
#         """
#         # nn.Sequential 对于多个输入的网络结构不太适用,只是适合顺序结构的模型
#         self.net = nn.Sequential(
#             nn.Conv2d(3, 6, 5, 1, 0),nn.ReLU(),    # input(3,32,32)   output(6,28,28)
#             nn.MaxPool2d(2, 2, 0),                 # output(6,14,14)
#             nn.Conv2d(6, 16, 5, 1, 0),nn.ReLU(),   # output(16,10,10)
#             nn.MaxPool2d(2, 2, 0),                 # output(16,5,5)            
#             nn.Flatten(),
#             nn.Linear(16*5*5, 120),nn.ReLU(),
#             nn.Linear(120, 84),nn.ReLU(), 
#             nn.Linear(84, 10)
#         )
        
#     def forward(self,x):
#         return self.net(x)  
    

"""
 固定套路的写法:虽然看起来复杂,但是使用所有的模型,因为有的模型是中途有多个输入的
"""
    
class LeNet(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        self.conv1 = nn.Conv2d(3,6,5)        # input(3,32,32)   output(6,28,28)
        self.pool1 = nn.MaxPool2d(2,2,0)     # output(6,14,14)
        self.conv2 = nn.Conv2d(6, 16, 5)     # output(16,10,10)
        self.pool2 = nn.MaxPool2d(2,2,0)     # output(16,5,5)
        self.fc1 = nn.Linear(16*5*5,120)
        self.fc2 = nn.Linear(120,84)
        self.fc3 = nn.Linear(84,10)
    
    def forward(self,x):
        x = F.relu(self.conv1(x))    
        x = self.pool1(x)            
        x = F.relu(self.conv2(x))    
        x = self.pool2(x)            
        x = x.view(-1,16*5*5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x
         

# input = torch.rand([7,3,32,32])  
# model = LeNet()
# print(model)
# output = model(input)
        
        
        
        
            
        


