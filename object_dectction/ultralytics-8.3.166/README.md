# 本项目中的文件对应的是B站中yolo模型改进系列的代码，存放地方，可以根据模型名称找到对应的文件

ultralytics-8.3.166 资料包：
链接: https://pan.baidu.com/s/1qkgjgy2FJgSthvXguXSobA?pwd=r5jd 提取码: r5jd 
--来自百度网盘超级会员v1的分享

本文件夹代码对应B站视频：https://space.bilibili.com/3546610390797058/lists/5973198?type=season
## 文件使用方法
直接下载对应的修改的文件即可，不用全部下载ultralytics-8.3.166文件夹，环境搭建请看B站 手把手带你实战ultralytics系列，使用linux系统运行
https://space.bilibili.com/3546610390797058/lists/5857510?type=season


B站地址：https://space.bilibili.com/3546610390797058

cv/nlp论文研读|代码实操|模块缝合|水论文方法分享 qq群707837795


## ultralytics-8.3.166 安装命令（wsl+linux 环境）
注：对照B站视频《手把手带你实战ultralytics系列》来安装ultralytics-8.3.166版本
这是在linux环境下安装ultralytics-8.3.166版本，当然你也可以直接在windows中安装ultralytics-8.3.166版本 (windows系统下安装ultralytics-8.3.166版本可以直接自行安装)

### WSL
下载Ubuntu命令： wsl   --install (直接下载在了C盘)
查看Ubuntu命令：wsl --list
查看Ubuntu版本：wsl  --list  --online
切换版本：wsl  --set-default Ubuntu-20.4
在C盘整体导出： wsl --export Ubuntu F:\software\wsl\ubuntu.tar
在C盘注销Ubuntu：wsl --unregister Ubuntu
迁移到新建的盘的文件夹中： wsl --export Ubuntu F:\software\wsl\ubuntu.tar    F:\software\wsl\ubuntu2004
在用户名的文件下（c/用户/1）下面新建一个文件 .wslconfig   [可以设置内存大小线程等]

安装miniconda：wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
安装运行脚本：bash Miniconda3-latest-Linux-x86_64.sh
激活环境：conda activate base
退出conda环境： conda deactivate
刷新Ubuntu目录：  sudo apt-get update
把已经安装的包升级到最新版：sudo apt-get upgrade

copy代码到ubuntu中：
mkdir -p ~/projects     新建一个文件夹
cp -r /mnt/f/pytorchPorject/yolov12 ~/projects/    copy代码
cd ~/projects/yolov12    打开代码文件
code .   用vscode打开

这里版本一定要对得上！ cuda11.8 才可以对上

### 用wsl安装Ubuntu的命令：
1、下载 wsl  --install  （直接下载在C盘,并且Ubuntu系统不稳定）此步骤略过
2、查看可以下载的版本： wsl --list --online
3、下载特定版本  wsl --install  -d Ubuntu-20.04
4、设定默认版本：wsl  --set-default Ubuntu-20.04
5、关闭系统：wsl --shutdown
6、迁移系统到其他的盘：（导出的是打包好的Ubuntu，但是原系统盘还在）wsl --export Ubuntu-20.04 F:\software\wsl\Ubuntu-20.04.tar
7、 卸载注销Unbutu：wsl --unregister Ubuntu-20.04  （卸载C盘中的Ubuntu）
8、迁移到新建的盘的文件夹中： wsl --import Ubuntu-20.04   F:\software\wsl\ubuntu2004   F:\software\wsl\Ubuntu-20.04.tar       

在运行时候出现这个情况，如何进入到自己的用户名下
C:\Users\1>wsl
root@DESKTOP-NKA7IDS:/mnt/c/Users/1#
ubuntu2004 config --default-user joey
joey@DESKTOP-NKA7IDS:/mnt/c/Users/1$


### 在Ubuntu中安装conda环境
下载 Miniconda 安装脚本： wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
 文件权限  chmod +x Miniconda3-latest-Linux-x86_64.sh
运行安装脚本：./Miniconda3-latest-Linux-x86_64.sh
安装完后，刷新 shell 配置  source ~/.bashrc
安装conda环境
conda create -n ultralytics python=3.10
conda activate ultralytics 
配置ultralyticsics的环境依赖  pip install ultralytics 
安装cuda版tensor ： pip3 install torch torchvision torchaudio

### 下载ultralytics代码到linux
新建文件夹  mkdir projects
到projects文件下：cd projects
copy文件：cp -r  /mnt/f/ultralytics   ./
解压代码:  unzip  zip文件

连接vscode
vscode下载的插件 python、wsl
