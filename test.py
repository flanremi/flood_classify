from multiprocessing import freeze_support

import torch
import torchvision.transforms as transforms

from classifier import CNet

transform = transforms.Compose(
    [transforms.ToTensor()]
)

from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader

batch_size = 4
train_data = ImageFolder(root='./image/test', transform=transform)
train_loader = DataLoader(dataset=train_data,
                          batch_size=batch_size,
                          pin_memory=True, num_workers=1)

if __name__ == '__main__':
    freeze_support()
    net = CNet()
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    net.to(device)
    net.load_state_dict(torch.load("classifier_net_0.8"))

    pos = 0
    for img_batch, label_batch in train_loader:
        img_batch = img_batch.to(device)
        label_batch = label_batch.to(device)
        train_logits, train_prob = net(img_batch)
        a = train_prob.detach().cpu().numpy().tolist()
        for _a in a:
            tag = 0
            if _a[0] < _a[1]:
                tag = 1
            print(train_data.imgs[pos][0] + "===" +  str(tag))
            pos += 1