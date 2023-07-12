from multiprocessing import freeze_support

from torch import nn
import torch


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
        self.avg_pool=nn.AdaptiveAvgPool2d((1,1))
        self.softmax=nn.Softmax(dim=1)

    def forward(self, x):
        x = self.conv1(x)
        x = self.pool(x)
        x = self.conv2(x)
        x = self.pool(x)
        x = self.conv3(x)
        x = self.pool(x)
        x = self.conv4(x)
        x = self.pool(x)
        x = self.conv5(x)
        x = self.pool(x)
        x = self.avg_pool(x)
        x = torch.flatten(x, 1)
        logits = self.fclayer(x)
        prob = self.softmax(logits)
        return logits, prob

if __name__ == '__main__':
    freeze_support()
    net = CNet()
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    net.to(device)

    lr = 0.001
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(net.parameters(), lr=lr)

    import torchvision.transforms as transforms
    transform = transforms.Compose(
        [transforms.ToTensor()]
    )

    from torchvision.datasets import ImageFolder
    from torch.utils.data import DataLoader
    batch_size = 4
    train_data = ImageFolder(root= './image/data/', transform=transform)
    train_loader = DataLoader(dataset=train_data,
    batch_size=batch_size,
    pin_memory=True, num_workers=8, shuffle=True)
    for i in range(100):
        for img_batch, label_batch in train_loader:
            img_batch = img_batch.to(device)
            label_batch = label_batch.to(device)
            train_logits, train_prob = net(img_batch)
            train_loss = criterion.forward(train_logits, label_batch)
            optimizer.zero_grad()
            train_loss.backward()
            optimizer.step()
            print(train_loss)
        print("==============" + str(i) + "================")
    torch.save(net.state_dict(),"classifier_net_0.5")

# class = self.q_net(state).argmax().item()