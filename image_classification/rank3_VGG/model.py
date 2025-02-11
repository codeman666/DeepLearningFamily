import torch.nn as nn
import torch


class VGG(nn.Module):
    def __init__(self,features,num_classes=1000,init_weights=False):
        super(VGG,self).__init__()
        self.features = features
        self.classifier = nn.Sequential(
            nn.Linear(512*7*7,4096),
            nn.ReLU(True),
            nn.Dropout(p=0.5),
            nn.Linear(4096,4096),
            nn.ReLU(True),
            nn.Dropout(p=0.5),
            nn.Linear(4096,num_classes)
        )
        if init_weights:
            self._initialize_weights()
            
    def forward(self,x):
        x = self.features(x)
        x = torch.flatten(x,start_dim=1)
        x = self.classifier(x)
        return x 
    
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m,nn.Conv2d):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias,0)
                elif isinstance(m,nn.Linear):
                    nn.init.xavier_uniform(m.weight)
                    nn.init.constant_(m.bias,0)
  
                   
def make_layers(cfg: list):  #类型注解为list
    """
    构建VGG网络的卷积层
    """
    layers = []
    in_channels = 3
    for v in cfg:
        if v == "M":
            layers += [nn.MaxPool2d(kernel_size=2,stride=2)] 
        else:
            conv2d = nn.Conv2d(in_channels,v,kernel_size=3,padding=1)
            layers += [conv2d,nn.ReLU(True)]
            in_channels = v
    return nn.Sequential(*layers)  #语法用于解包列表或元组，将其中的每个层作为单独的参数传递给 nn.Sequential
    
cfgs = {
    "vgg11": [64, "M", 128, "M", 256, 256, "M", 512, 512, "M", 512, 512, "M"],
    "vgg13": [64, 64, "M", 128, 128, "M", 256, 256, "M", 512, 512, "M", 512, 512, "M"],
    "vgg16": [64, 64, "M", 128, 128, "M", 256, 256, 256, "M", 512, 512, 512, "M", 512, 512, 512, "M"],
    "vgg19": [64, 64, "M", 128, 128, "M", 256, 256, 256, 256, "M", 512, 512, 512, 512, "M", 512, 512, 512, 512, "M"],
}
    
def vgg(model_name="vgg16", **kwargs):
    assert model_name in cfgs, f"Warning:model number {model_name} not in cfgs dict!"
    cfg = cfgs[model_name]
    
    model=VGG(make_layers(cfg),**kwargs)
    return model

model = vgg()
print(model)
input_tensor = torch.randn(8, 3, 224, 224)
output = model(input_tensor)
print(output)