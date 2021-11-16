import torch
import torch.nn as nn
import torch.nn.functional as F

from batchers.dataset import Batcher
from configs import args
from utils.utils import get_paths


class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(7, 12, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(12, 16, 5)
        self.fc1 = nn.Linear(16 * 53 * 53, 1)
        #self.fc2 = nn.Linear(120, 1)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1)  # flatten all dimensions except batch
        x = F.relu(self.fc1(x))
        #x = F.relu(self.fc2(x))
        return x


net = Net()
net.to('cuda')
import torch.optim as optim

criterion = nn.MSELoss()
optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)
paths_train = get_paths(args.dataset, ['train'], args.fold, args.data_path)
batcher_train = Batcher(paths_train, args.scaler_features_keys, args.ls_bands, args.nl_band, args.label_name,
                        args.nl_label, 'DHS', args.augment, args.batch_size, groupby=args.group,
                        cache='train' in args.cache)
for epoch in range(2):  # loop over the dataset multiple times
    print('GPUS:', net.device)
    running_loss = 0.0
    for i, batch in enumerate(batcher_train, 0):
        # get the inputs; data is a list of [inputs, labels]
        x = torch.tensor(batch['images'], device=net.conv1.weight.device)
        target = torch.tensor(batch['labels'], device=net.conv1.weight.device)
        x = x.reshape(-1, x.shape[-1], x.shape[-3], x.shape[-2])  # [batch_size ,in_channels, H ,W]

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = net(x)
        outputs = outputs.squeeze(dim=-1)
        loss = criterion(outputs, target)
        loss.backward()
        optimizer.step()

        # print statistics
        running_loss += loss.item()
        if i % 2000 == 1999:  # print every 2000 mini-batches
            print('[%d, %5d] loss: %.3f' %
                  (epoch + 1, i + 1, running_loss / 2000))
            running_loss = 0.0

print('Finished Training')
