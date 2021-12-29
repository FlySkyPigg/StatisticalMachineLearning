import torch
from torch.utils.data import DataLoader, Dataset
from glob import glob
import csv
import random
import torch.nn as nn
import numpy as np


class VideoDataset(Dataset):
    def __init__(self, split):
        super(VideoDataset, self).__init__()
        self.paths = []
        self.labels = []
        with open('data/label_train.csv', newline='') as csvfile:
            reader = csv.reader(csvfile)
            next(reader)
            for row in reader:
                self.paths.append(row[0])
                self.labels.append(int(row[1]))

        random.seed(233)
        inds = list(range(len(self)))
        random.shuffle(inds)
        # inds = inds[:int(len(self) * 0.8)] if split == 'train' else inds[int(len(self) * 0.8):]
        self.paths = [self.paths[ind] for ind in inds]
        self.labels = [self.labels[ind] for ind in inds]

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, item):
        path = f'data/train/{self.paths[item]}'
        label = self.labels[item]
        data = np.load(path)

        data = torch.from_numpy(data).float()
        label = torch.tensor(label).long()

        return data, label


class RnnNet(nn.Module):
    def __init__(self):
        super(RnnNet, self).__init__()
        self.rnn_net = nn.Sequential(
            nn.GRU(input_size=15, hidden_size=48, num_layers=3, batch_first=True, dropout=0.2))
        self.fc = nn.Sequential(
            nn.Linear(48, 48),
            nn.BatchNorm1d(48),
            # nn.ReLU(),
            # nn.Linear(32, 32),
            # nn.BatchNorm1d(32),
            nn.Dropout(0.2),
            nn.ReLU(),
            nn.Linear(48, 20)
        )

    def forward(self, data):
        output, hn = self.rnn_net(data)
        feature = output[:, -1, :]
        logits = self.fc(feature)
        return logits


def get_test_data():
    paths = list(glob('data/test/*.npy'))
    paths.sort()
    datas = []
    for path in paths:
        datas.append(np.load(path))
    datas = np.stack(datas, axis=0)
    paths = [path.replace('data/test/', '') for path in paths]

    return datas, paths


if __name__ == "__main__":
    batch_size = 64

    train_dataset = VideoDataset(split='train')
    val_dataset = VideoDataset(split='test')

    train_dataloader = DataLoader(dataset=train_dataset, shuffle=True, batch_size=batch_size)
    val_dataloader = DataLoader(dataset=val_dataset, shuffle=True, batch_size=batch_size)

    net = RnnNet().cuda()

    net.load_state_dict(torch.load('rnn.pth'))
    net.eval()

    test_datas, test_paths = get_test_data()
    test_datas = torch.from_numpy(test_datas).float().cuda()
    logits = net(test_datas)
    test_pred_labels = torch.argmax(logits, dim=1)
    test_pred_labels = test_pred_labels.detach().cpu().numpy()

    with open('data/rnn_label_test.csv', 'w') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['id', 'category'])
        for path, pred_label in zip(test_paths, test_pred_labels):
            writer.writerow([path, pred_label])

    print(logits.shape)
    exit()

    optimizer = torch.optim.Adam(net.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer=optimizer)
    max_epoch = 1000

    for epoch in range(max_epoch):

        total = 0
        correct = 0
        net.train()
        for data, label in train_dataloader:
            data, label = data.cuda(), label.cuda()
            logits = net(data)

            loss = torch.nn.CrossEntropyLoss()(logits, label)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            pred_labels = torch.argmax(logits, dim=1)
            correct += (pred_labels == label).sum()
            total += pred_labels.shape[0]
        print('train precision', correct / total)

        total = 0
        correct = 0
        net.eval()
        for data, label in val_dataloader:
            data, label = data.cuda(), label.cuda()
            with torch.no_grad():
                logits = net(data)
            pred_labels = torch.argmax(logits, dim=1)
            correct += (pred_labels == label).sum()
            total += pred_labels.shape[0]
        print('test precision', correct / total)

        scheduler.step(correct / total)

        torch.save(net.cpu().state_dict(), 'rnn.pth')
        net.cuda()
