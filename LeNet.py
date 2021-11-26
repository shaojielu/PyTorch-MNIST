import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import ConcatDataset
import matplotlib.pyplot as plt



# 数据加载与预处理

train_tfm = transforms.Compose(
    [
        transforms.Resize((28, 28)),
        transforms.ToTensor(),
        transforms.Normalize([0.5],[0.5])
    ])
# train_tfm = transforms.Compose(
#     [
#         transforms.Resize((28, 28)),
#         transforms.ToTensor(),
#         transforms.Normalize((0.5), (0.5)),
#         transforms.RandomHorizontalFlip(),                            # 随机水平镜像
#         transforms.RandomErasing(scale=(0.04, 0.1), ratio=(0.3, 2)),  # 随机遮挡
#         transforms.RandomCrop(28, padding=2)                          # 随机中心裁剪
#     ])

log_interval = 10
n_epochs = 10
batch_size = 128
random_seed = 1
torch.manual_seed(random_seed)

train_loader = torch.utils.data.DataLoader(
  torchvision.datasets.MNIST('./data/', train=True, download=True,
                             transform=train_tfm),
  batch_size=batch_size, shuffle=True)
test_loader = torch.utils.data.DataLoader(
  torchvision.datasets.MNIST('./data/', train=False, download=True,
                             transform=train_tfm),
  batch_size=batch_size, shuffle=False)

examples = enumerate(test_loader)
batch_idx, (example_data, example_targets) = next(examples)
# print(example_targets)
# print(example_data.shape)

#  查看数据集
# fig = plt.figure()
# for i in range(6):
#   plt.subplot(2,3,i+1)
#   plt.tight_layout()
#   plt.imshow(example_data[i][0], cmap='gray', interpolation='none')
#   plt.title("Ground Truth: {}".format(example_targets[i]))
#   plt.xticks([])
#   plt.yticks([])
# plt.show()

# 构造卷积网络模型结构
class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        # self.conv1 = nn.Conv2d(1, 10, 3)           # 1 通道 20 输出  3卷积大小
        # self.conv2 = nn.Conv2d(10, 20, 3)
        # self.conv2_drop = nn.Dropout2d()
        # self.fc1 = nn.Linear(20 * 5 * 5, 256)
        # self.fc2 = nn.Linear(256, 128)
        # self.fc3 = nn.Linear(128, 10)
        self.cnn_layers = nn.Sequential(
            nn.Conv2d(1, 20, 3),
            nn.BatchNorm2d(20),
            nn.MaxPool2d(2, 2),
            nn.ReLU(),

            nn.Conv2d(20, 40, 3),
            nn.BatchNorm2d(40),
            nn.Dropout2d(),
            nn.MaxPool2d(2, 2),
            nn.ReLU(),

        )
        self.fc_layers = nn.Sequential(
            nn.Linear(40 * 5 * 5, 150),
            nn.ReLU(),
            nn.Linear(150, 50),
            nn.ReLU(),
            nn.Linear(50, 10)
        )




    def forward(self, x):
        # x = F.max_pool2d(F.relu(self.conv1(x)), (2, 2))
        # x = F.max_pool2d(F.relu(self.conv2_drop(self.conv2(x))), 2)
        # x = x.view(-1, self.num_flat_features(x))
        # x = F.relu(self.fc1(x))
        # x = F.relu(self.fc2(x))
        # x = self.fc3(x)

        x = self.cnn_layers(x)
        x = x.view(-1, self.num_flat_features(x))
        # x = x.flatten(1)
        x = self.fc_layers(x)
        return F.log_softmax(x)  # 防止CrossEntropyLoss为负

    def num_flat_features(self, x):
        size = x.size()[1:]  # 除批量外的所有尺寸
        num_features = 1
        for s in size:
            num_features *= s
        return num_features



net = Net()



# 损失函数
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(net.parameters(), lr=0.0008,weight_decay=1e-5)

# 遍历数据迭代器，然后将输入馈送到网络并进行优化

train_losses = []
train_counter = []
test_losses = []
test_counter = [i*len(train_loader.dataset) for i in range(n_epochs + 1)]
def train(epoch):
    net.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        optimizer.zero_grad()
        output = net(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                       100. * batch_idx / len(train_loader), loss.item()))
            train_losses.append(loss.item())
            train_counter.append(
                (batch_idx * 64) + ((epoch - 1) * len(train_loader.dataset)))
            torch.save(net.state_dict(), './model.pth') # 保存模型的参数
            torch.save(optimizer.state_dict(), './optimizer.pth')


# 测试训练好的模型

def test():
    net.eval() #测试的时候关掉nn.BatchNorm2d(40),nn.Dropout2d(),
    test_loss = 0
    correct = 0
    prediction = []
    with torch.no_grad():   # 阻止自动求导
        for data, target in test_loader:
            output = net(data)
            test_loss += F.nll_loss(output, target, size_average=False).item()
            pred = output.data.max(1, keepdim=True)[1]
            correct += pred.eq(target.data.view_as(pred)).sum()
    test_loss /= len(test_loader.dataset)
    test_losses.append(test_loss)
    test_label = np.argmax(pred.data.numpy(), axis=1)
    for y in test_label:
        prediction.append(y)
    print('\nTest set: Avg. loss: {:.4f}, Accuracy: {}/{} ({:.3f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))
    with open("predict.csv", 'w') as f:
        f.write('Id,Category\n')
        for i, y in enumerate(prediction):
            f.write('{},{}\n'.format(i, y))



test()
for epoch in range(1, n_epochs + 1):
  train(epoch)
  test()


# 可视化loss的变化，方便用于调整参数
fig = plt.figure()
plt.plot(train_counter, train_losses, color='blue')
plt.scatter(test_counter, test_losses, color='red')
plt.legend(['Train Loss', 'Test Loss'], loc='upper right')
plt.xlabel('number of training examples seen')
plt.ylabel('negative log likelihood loss')
plt.show()


continued_network = Net()
continued_optimizer = torch.optim.Adam(net.parameters(), lr=0.0008,weight_decay=1e-5)
network_state_dict = torch.load('model.pth')
continued_network.load_state_dict(network_state_dict)
optimizer_state_dict = torch.load('optimizer.pth')
continued_optimizer.load_state_dict(optimizer_state_dict)
for i in range(11,28):
  test_counter.append(i*len(train_loader.dataset))
  train(i)
  test()


fig = plt.figure()
plt.plot(train_counter, train_losses, color='blue')
plt.scatter(test_counter, test_losses, color='red')
plt.legend(['Train Loss', 'Test Loss'], loc='upper right')
plt.xlabel('number of training examples seen')
plt.ylabel('negative log likelihood loss')
plt.show()

print('Finished Training')

