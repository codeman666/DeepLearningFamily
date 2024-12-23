
import torch
import torch.nn as nn


class AlexNet(nn.Module):
    def __init__(self,num_classes=1000):
        super(AlexNet,self).__init__()
        self.net = nn.Sequential(
            # (3,224,224) --> (48,55,55)  也可以设置成(3,227,227)作为原始图像,就不用设置padding
            nn.Conv2d(in_channels=3,out_channels=48,kernel_size=11,stride=4,padding=2), 
            nn.ReLU(inplace=True),
            #(48,55,55)->(48,27,27)
            nn.MaxPool2d(kernel_size=3,stride=2), 
            #LRN函数 (对5个通道进行归一化(必须是奇数),alpha控制调节的幅度,
            # beta控制归一化后的放大或压缩效果,k为偏移量)
            nn.LocalResponseNorm(size=5,alpha=0.0001,beta=0.75,k=2),  #[论文创新点]
            
            #(48,27,27) -> (128,27,27)
            nn.Conv2d(48,128,5,padding=2),
            nn.ReLU(inplace=True),
            #(128,27,27) -> (128,13,13)
            nn.MaxPool2d(kernel_size=3,stride=2),
            # 同上局部归一化 [论文创新点]
            nn.LocalResponseNorm(size=5,alpha=0.0001,beta=0.75,k=2),
            
            #(128,13,13) --> (192,13,13)
            nn.Conv2d(128,192,3,padding=1),
            nn.ReLU(inplace=True),
            
            #(192,13,13) --> (192,13,13)
            nn.Conv2d(192,192,3,padding=1),
            nn.ReLU(inplace=True),
            
            # (192,13,13) -> (128,13,13)
            nn.Conv2d(192,128,3,padding=1),
            nn.ReLU(inplace=True),
            # (128,13,13) --> (128,6,6)
            nn.MaxPool2d(kernel_size=3,stride=2),            
        ) 
        self.classifier = nn.Sequential(
            nn.Dropout(p=0.5), # 随机失活 [论文创新点]
            nn.Linear(in_features=(128*6*6),out_features=2048),
            nn.ReLU(inplace=True),#就地操作,直接修改张量
            nn.Dropout(p=0.5), 
            nn.Linear(in_features=2048,out_features=2048),
            nn.ReLU(inplace=True),
            nn.Linear(in_features=2048,out_features=num_classes),
        )
    
    def forward(self,x):
        x = self.net(x)
        x = torch.flatten(x,start_dim=1)  #[b,c,h,w] 从第一维度展开
        x = self.classifier(x)
        return x
        

# model = AlexNet()
# input = torch.rand([1000,3,224,224])  
# print(model)
# output = model(input)
# print(output)
                
        