import os
import time
from multiprocessing import freeze_support

import pymysql
import torch
import torchvision.transforms as transforms

from classifier6 import CNet

transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Resize((1024, 1024))]
)

from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader

batch_size = 4
target_folder = './image/val'
eval_data = ImageFolder(root=target_folder, transform=transform)
eval_loader = DataLoader(dataset=eval_data,
                          batch_size=batch_size,
                          pin_memory=True, num_workers=1)

if __name__ == '__main__':
    while True:
        # freeze_support()
        net = CNet()
        device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        net.to(device)
        net.load_state_dict(torch.load("classifier_net_1.0",map_location='cpu'))

        pos = 0
        eval_data = ImageFolder(root=target_folder, transform=transform)
        eval_loader = DataLoader(dataset=eval_data,
                                 batch_size=batch_size,
                                 pin_memory=True, num_workers=1)

        for img_batch, label_batch in eval_loader:
            img_batch = img_batch.to(device)
            label_batch = label_batch.to(device)
            train_logits, train_prob, _ = net(img_batch, label_batch)

            a = train_prob.detach().cpu().numpy().tolist()


            for _a in a:
                alert_level = 0
                if _a[0] < _a[1]:
                    alert_level = 5
                print(eval_data.imgs[pos][0] + "===" + str(alert_level))

        time.sleep(15)