import torch
from model_official import LeNet
import torchvision
import torch.nn as nn
import torchvision.transforms as transforms
import torch.optim as optim
import torch.utils



def main():
    # 使用gpu进行计算
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    transform = transforms.Compose(
        [transforms.ToTensor(),
        transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))]
    )

    """
    数据集:60000张
    训练集:50,000 张图像，按照 10 个类别平均分配，每类包含 5,000 张。
    测试集:10,000 张图像，每类包含 1,000 张。
    数据格式：
    每张图像存储为 3D 张量，形状为 (3, 32, 32)，对应 3 个通道(RGB)和图像大小。
    标签是 0 到 9 的整数，表示所属类别。
    
    num_workers: 
    小数据集：对于小型数据集，可以设置为 0 或 1，因为并行加载的开销可能大于加速效果。
    大型数据集：对于大型数据集，尤其是需要进行复杂数据处理时，设置更高的 num_workers 会有明显的性能提升。
    num_workers 的值可以根据 CPU 核心数来选择，比如设置为 4、8 或更高。
    """

    train_set = torchvision.datasets.CIFAR10(root="./data",train=True,
                                            download=False,transform=transform)
    train_loader = torch.utils.data.DataLoader(train_set,batch_size=36,
                                            shuffle=True,num_workers=0)

    test_set = torchvision.datasets.CIFAR10(root="./data",train=False,
                                            download=False,transform=transform)
    test_loader = torch.utils.data.DataLoader(test_set,batch_size=5000,
                                            shuffle=True,num_workers=0)

    
    # 构建一个迭代器,用于迭代出测试数据,因为CIFAR10的测试数据只有10000张,并且batch_size设置了10000,
    # 那么直接就可以迭代一次就可以了,准备测试数据,用于后面使用
    dataiter = iter(test_loader)
    test_images, test_labels = next(dataiter)
    # print(f"test_labels: {test_labels}")
    # print(f"test_labels_size:{test_labels.size(0)}")
    test_images = test_images.to(device)
    test_labels = test_labels.to(device)
    
    """
    测试代码:用于测试下载的数据
    """
    # classes = ('plane', 'car', 'bird', 'cat',
    #            'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
    # import matplotlib.pyplot as plt
    # import numpy as np
    # def imshow(img):
    #     img = img / 2 + 0.5     # unnormalize
    #     npimg = img.numpy()
    #     plt.imshow(np.transpose(npimg, (1, 2, 0)))
    #     plt.show()
    # # show images
    # imshow(torchvision.utils.make_grid(images))
    # # print labels
    # print(' '.join(f'{classes[labels[j]]:5s}' for j in range(batch_size)))

    """
    数据进行训练
    """
    #实例化模型
    net = LeNet()
    net = net.to(device)
    
    # 损失函数
    loss_function = nn.CrossEntropyLoss()  
    loss_function = loss_function.to(device) 
    
    # 迭代器  optim 是pytorch的模块,其中Adam是一个优化算法
    optimizer = optim.Adam(net.parameters(), lr=0.001) 
    #训练轮数
    for epoch in range(5):
        
        running_loss = 0.0
        for step , data in enumerate(train_loader, start=0):
    
            inputs, labels = data
            #print(f"Step {step}: inputs_size = {inputs.size()}, labels_size = {labels.size()}")
            inputs = inputs.to(device)
            labels = labels.to(device)
            
            optimizer.zero_grad()
            outputs = net(inputs)
            # print(f"outputs size = {outputs.size()}")
            # print(f"outputs = {outputs}")
            loss = loss_function(outputs,labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()  # item由tensor转为标量
            if step % 300 == 299:
                with torch.no_grad():
                    outputs = net(test_images)
                    predict_y = torch.argmax(outputs, dim=1)  # dim=1 按行找最大值, 一共输出两个值[1]找到最大值对应的索引
                    
                    # test_labels.size(0)返回样本数量 和 test_labels.size()返回张量的形状,
                    accuracy = torch.eq(predict_y,test_labels).sum() / test_labels.size(0)
                    
                    print('[%d, %5d] train_loss: %.3f test_accuracy:%.3f' % 
                          (epoch+1,step+1,running_loss / 300, accuracy))
                    running_loss = 0.0
    print("Finished Training")
    save_path = './Lenet.pth'
    torch.save(net.state_dict(), save_path)

if __name__ == '__main__':
    main()
                


