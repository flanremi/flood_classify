from multiprocessing import freeze_support
import os
from torch import nn
import torch

from modules import MarginLoss


class CNet(nn.Module):
    def __init__(self,num_classes=2):
        super(CNet,self).__init__()
        self.conv1=nn.Sequential(
                nn.Conv2d(3,32,4,1,padding=1),
                nn.BatchNorm2d(32),
                nn.ReLU()
        )
        self.conv2 = nn.Sequential(
                nn.Conv2d(32, 64, 4, 1, padding=1),
                nn.BatchNorm2d(64),
                nn.ReLU()
        )
        self.conv3 = nn.Sequential(
                nn.Conv2d(64, 128, 4, 1, padding=1),
                nn.BatchNorm2d(128),
                nn.ReLU()
        )
        self.conv4 = nn.Sequential(
                nn.Conv2d(128, 256, 4, 1, padding=1),
                nn.BatchNorm2d(256),
                nn.ReLU()
        )
        self.conv5 = nn.Sequential(
                nn.Conv2d(256, 512, 4, 1, padding=1),
                nn.BatchNorm2d(512),
                nn.ReLU()
        )
        self.pool = nn.AvgPool2d(2, 2)
        self.fclayer=nn.Sequential(
                nn.Linear(512,1024),
                nn.ReLU(),
                nn.Dropout(0.4),
                nn.Linear(1024, 1024),
                nn.ReLU(),
                nn.Dropout(0.8),
                nn.Linear(1024, 256),
                nn.ReLU(),
                nn.Linear(256,num_classes)
        )
        self.avg_pool=nn.AdaptiveAvgPool2d((1,1)) # 自适应平均池化，指定输出（H，W）
        self.softmax=nn.Softmax(dim=1)
        self.criterion2 = MarginLoss(margin=0.2).cuda()  # 间隔损失函数

    def forward(self, x, x_label):
        x = self.conv1(x) # x=img_batch  torch.Size([4, 32, 1023, 1023])
        x = self.pool(x) # torch.Size([4, 32, 511, 511])
        x = self.conv2(x) # torch.Size([4, 64, 510, 510])
        x = self.pool(x) # torch.Size([4, 64, 255, 255])
        x = self.conv3(x) # torch.Size([4, 128, 254, 254])
        x = self.pool(x) # torch.Size([4, 128, 127, 127])
        x = self.conv4(x) # torch.Size([4, 256, 126, 126])
        x = self.pool(x) # torch.Size([4, 256, 63, 63])
        x = self.conv5(x) # torch.Size([4, 512, 62, 62])
        x = self.pool(x) # torch.Size([4, 512, 31, 31])
        x = self.avg_pool(x) # torch.Size([4, 512, 1, 1])
        x = torch.flatten(x, 1) #按照列来拼接，横向拼接，一个view torch.Size([4, 512])
        logits = self.fclayer(x) # torch.Size([4, 2])
        prob = self.softmax(logits) # torch.Size([4, 2])


        marginLoss=self.criterion2(x, x_label) # 间隔损失
        # return logits, prob
        return logits, prob, marginLoss



if __name__ == '__main__':
    freeze_support()
    net = CNet()
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    net.to(device)

    lr = 0.001
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(net.parameters(), lr=lr)

    import torchvision.transforms as transforms
    # 数据预处理的操作，可以通过torchvision.transforms模块中的函数进行定义，例如对图像进行缩放、裁剪、标准化等操作
    transform = transforms.Compose(
        [transforms.ToTensor()]
    ) # 定义一个数据转换的管道，将输入的数据转换为Tensor类型

    from torchvision.datasets import ImageFolder # 数据集类，可以方便地加载包含多个类别的图像数据集，图像文件夹结构应该符合特殊格式要求
    from torch.utils.data import DataLoader
    batch_size = 4
    # 创建一个ImageFolder对象，其数据集根目录为'./image/data/'，并且对图像数据进行了预处理操作
    train_data = ImageFolder(root= './image/data/', transform=transform)

    # 创建一个数据加载器
    train_loader = DataLoader(dataset=train_data,
    batch_size=batch_size,
    pin_memory=True, num_workers=8, shuffle=True) # 是否将数据存储在固定内存中  用于加载数据的线程数量 是否对数据进行随机排序


    best_acc=0.0
    print("Start training...")
    for i in range(100):
        for img_batch, label_batch in train_loader:
            img_batch = img_batch.to(device)  # torch.Size([4, 3, 1024, 1024])
            label_batch = label_batch.to(device)  # torch.Size(4)
            # train_logits, train_prob = net(img_batch)
            train_logits, train_prob, marginloss = net(img_batch, label_batch)
            train_loss = criterion.forward(train_logits, label_batch)
            loss = train_loss + marginloss
            acc = torch.mean((train_prob.view(-1) == label_batch.view(-1)).type(torch.FloatTensor))

            optimizer.zero_grad()
            loss.backward()
            # train_loss.backward()
            optimizer.step()
            # print(train_loss)

            print("step:"+str(i)+"  |loss=" + loss + "    |acc=" + acc)
            if acc > best_acc:
                print('Best checkpoint')
                torch.save(net.state_dict(),"classifier_net_0.5")
                best_acc = acc

        # print("==============" + str(i) + "================")


    # torch.save(net.state_dict(),"classifier_net_0.5")
    print("\n####################\n")
    print("Finish training ")
# class = self.q_net(state).argmax().item()