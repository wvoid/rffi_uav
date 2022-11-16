import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
import model
from get_data_frommat import get_mat

device = torch.device('cuda')

Classes = [
    'h1', 'h2', 'h3', 'h4', 'v1', 'v2'
]


class DataTrain(Dataset):
    def __init__(self, train_x=None, train_y=None):
        self.data = train_x
        self.label = train_y

    def __getitem__(self, index):
        x = self.data[index]
        y = self.label[index]
        return x, y

    def __len__(self):
        return self.data.shape[0]


def train(train_loader):
    model1 = model.dnn1()
    model1.to('cuda')
    learn_rate = 0.001
    loss = torch.nn.CrossEntropyLoss()
    optim = torch.optim.Adam(model1.parameters(), learn_rate)
    for epoch in range(100):
        print(f'epoch:{epoch}')
        total_train_step = 0
        avg_loss = 0
        model1.train()
        for data in train_loader:
            x, y = data[0].float().to(device), data[1].to(device)
            output = model1(x)
            res_loss = loss(output, y.float())
            optim.zero_grad()
            res_loss.backward()
            optim.step()
            total_train_step += 1
            avg_loss += res_loss
        print(f'epoch:{epoch},loss:{avg_loss/total_train_step}')
    torch.save(model1, 'model/model_ep100_rawiq_nosnr_decfo_11.8')


def eval(test_loader, model):
    model.eval()
    total = 0
    correct = 0
    conf = np.zeros((6, 6))
    confnorm = np.zeros((6, 6))
    for data in test_loader:
        x, y = data[0].float().to(device), data[1].to(device)
        output = model(x)
        pred = output.data.max(1, keepdim=True)[1]
        label = y.data.max(1, keepdim=True)[1]
        total += y.size(0)
        correct = correct + (pred == label).sum()
        for i in range(len(label)):
            j = label[i]
            k = pred[i]
            conf[j, k] = conf[j, k] + 1
        for i in range(6):
            confnorm[i, :] = conf[i, :] / np.sum(conf[i, :])
    print(correct / total)
    print(confnorm)
    draw_confnorm(confnorm)


def complex2iq(data):
    len = data.shape[0]
    x = np.empty((len, 1, 1600, 2))
    for i in range(len):
        x[i][0, :, 0] = data[i].real
        x[i][0, :, 1] = data[i].imag
    return x


def draw_confnorm(confnorm):
    plt.figure(figsize=(3, 3))
    plt.imshow(confnorm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title('confusion matrix')
    plt.colorbar()
    p = np.arange(len(Classes))
    plt.xticks(p, Classes, rotation=45)
    plt.yticks(p, Classes)
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.show()


if __name__ == '__main__':
    data = np.load('data_train/data_6rf_rawiq_onehot_nosnr_decfo_11.9_与11.8相同usrp型号.npz')
    data_pb = data['data']
    data_pb = complex2iq(data_pb)
    label = data['label']
    dataset = DataTrain(train_x=data_pb, train_y=label)
    train_dataset, test_dataset = torch.utils.data.random_split(dataset, [int(0.9 * dataset.__len__()),
                                                                          int(0.1 * dataset.__len__())],generator=torch.Generator().manual_seed(1))
    train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=128)
    # train(train_loader)
    mod = torch.load('model/model_ep100_rawiq_nosnr_decfo_11.8')
    # x=get_mat('pb_mat_4')[0:30]
    # x=complex2iq(x)
    # pred=mod(torch.tensor(x).float().to(device))
    # print(pred)
    eval(train_loader, mod)

    # result = mod(torch.tensor(test_dataset[1:20][0]).float().to(device))
    # print(result)
    # print(test_dataset[1:20][1])
