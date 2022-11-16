import torch
import torch.nn.functional as F
from torchviz import make_dot
import graphviz


class dnn1(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(in_channels=1, out_channels=8, kernel_size=(1, 2), stride=50)
        self.conv2 = torch.nn.Conv2d(in_channels=8, out_channels=8, kernel_size=(2, 1), stride=2)
        self.conv3 = torch.nn.Conv2d(in_channels=8, out_channels=8, kernel_size=(2, 1))
        self.conv4 = torch.nn.Conv2d(in_channels=8, out_channels=8, kernel_size=(2, 1))
        self.conv5 = torch.nn.Conv2d(in_channels=8, out_channels=8, kernel_size=(2, 1))
        self.zeropad = torch.nn.ZeroPad2d((0, 0, 0, 1))
        self.fc1 = torch.nn.Linear(8 * 16, 32)
        self.fc2 = torch.nn.Linear(32, 6)

    def forward(self, x):
        x = self.conv1(x)
        # X=self.zeropad(x)
        # x = F.max_pool2d(x,(25,1))
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = torch.flatten(x, start_dim=1, end_dim=3)
        x = self.fc1(x)
        x = F.selu(x)
        x = self.fc2(x)
        x = F.selu(x)
        return F.softmax(x, dim=1)
        # return x


# m = dnn1()
# a = torch.rand(5, 1, 1600, 2)
# b = m(a)
# print(b.shape)
# from torchsummary import summary
# summary(m.cuda(), input_size=(1,1600,2))
# print(b.shape)
