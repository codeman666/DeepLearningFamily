from torch import nn
import torch

#确保通道数（如卷积层的输出通道）能够被divisor整除，从而提高硬件计算效率，调整通道数
def _make_divisible(ch,divisor=8,min_ch=None):
    if min_ch is None:
        min_ch = divisor
    new_ch = max(min_ch,int(ch + divisor / 2) // divisor * divisor)
    if new_ch < 0.9*ch:
        new_ch += divisor
    return new_ch

#conv2d + bn + relu
class ConvBNReLU(nn.Sequential):
    def __init__(self,in_channel,out_channel,kernel_size=3,stride=1,groups=1):
        padding = (kernel_size - 1) // 2 #确保输出的尺寸不变, // 向下取整
        super(ConvBNReLU,self).__init__(
            nn.Conv2d(in_channel,out_channel,kernel_size,stride,padding,groups=groups,bias=False),
            nn.BatchNorm2d(out_channel),
            nn.ReLU6(inplace=True)
        )

#倒残差结构
class InvertedResidual(nn.Module):
    def __init__(self,in_channel,out_channel,stride,expand_ratio):  #expand_ratio为扩展因子，1x1
        super(InvertedResidual,self).__init__()
        hidden_channel = in_channel * expand_ratio
        self.use_shortcut = stride == 1 and in_channel == out_channel # 判断是否是shortcut连接
        
        layers = []
        
        #第一层的bottleneck 是没有1x1卷积的,不需要扩展通道
        if expand_ratio != 1:
            layers.append(ConvBNReLU(in_channel,hidden_channel,kernel_size=1))
        layers.extend([
            #3x3卷积
            ConvBNReLU(hidden_channel,hidden_channel,stride=stride,groups=hidden_channel),
            #1x1卷积
            nn.Conv2d(hidden_channel,out_channel,kernel_size=1,bias=False),
            nn.BatchNorm2d(out_channel)
        ])
        self.conv = nn.Sequential(*layers)
        
    def forward(self,x):
        # 这里只有完全相同才可以相加
        if self.use_shortcut:
            return x + self.conv(x)
        else:
            return self.conv(x)

class MobileNetV2(nn.Module):
    def __init__(self,num_classes=1000,alpha=1.0,round_nearest=8):
        super(MobileNetV2,self).__init__()
        block = InvertedResidual
        #调整输入通道数和最后的通道
        input_channel = _make_divisible(32*alpha,round_nearest)
        last_channel = _make_divisible(1280*alpha,round_nearest)
        
        inverted_residual_setting = [
            # t, c, n, s
            [1,16,1,1],
            [6,24,2,2],
            [6,32,3,2],
            [6,64,4,2],
            [6,96,3,1],
            [6,160,3,2],
            [6,320,1,1],    
        ]
        
        features = []
        features.append(ConvBNReLU(3,input_channel,stride=2))
        for t,c,n,s in inverted_residual_setting:
            out_channel = _make_divisible(c*alpha,round_nearest)
            for i in range(n):
                #参数表中s标出来的值是残差块的第一层卷积核的步长，其他层的步长都是1
                stride = s if i == 0 else 1
                features.append(block(input_channel,out_channel,stride,expand_ratio=t))
                input_channel = out_channel
        
        features.append(ConvBNReLU(input_channel,last_channel,1))
        self.features = nn.Sequential(*features)
        
        self.avgpool = nn.AdaptiveAvgPool2d((1,1))
        #全连接层定义
        self.classifier = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(last_channel,num_classes)
        )
        
        for m in self.modules():
            if isinstance(m,nn.Conv2d):
                nn.init.kaiming_normal_(m.weight,mode="fan_out")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m,nn.BatchNorm2d):  
                nn.init.ones_(m.weight) #缩放系数
                nn.init.zeros_(m.bias)
            elif isinstance(m,nn.Linear):
                nn.init.normal_(m.weight,0,0.01) #用正态分布（高斯分布）初始化
                nn.init.zeros_(m.bias) # 偏置项置为0
    
    def forward(self,x):
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x,1)
        x = self.classifier(x)
        return x
    
# # 用于做测试的的模型
# model = MobileNetV2()
# print(model)
# input_tensor = torch.randn(8, 3, 256, 256)
# output = model(input_tensor)
# print(output)
