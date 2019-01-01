import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import torchvision.models as models
import numpy as np
import model

IMAGE_SIZE = (32, 32)
BATCH_SIZE = 64

def load_dataset():
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    transform = transforms.Compose([
        transforms.Resize(IMAGE_SIZE),
        transforms.ToTensor(),
        normalize,
    ])

    dataset = torchvision.datasets.ImageFolder('dataset/', transform=transform)

    print(dataset.classes)
    print(dataset.__len__())
    return dataset
    # x, y = dataset[0]
    # print x.size()
    # print y


def train(dataset):
    train_size = int(dataset.__len__() * 0.8)
    test_size = dataset.__len__() - train_size
    print(train_size, test_size)

    trainset, testset = torch.utils.data.random_split(dataset, [train_size, test_size])

    trainloader = torch.utils.data.DataLoader(trainset, batch_size=BATCH_SIZE, shuffle=False, num_workers=1)
    testloader = torch.utils.data.DataLoader(testset, batch_size=BATCH_SIZE, shuffle=False, num_workers=1)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(device)

    net = model.Net(dataset.classes)
    net = net.to(device)

    lr = float(input('Enter learning rate: '))
    epoch_num = int(input('Enter number of epoch: '))

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(net.parameters(), lr=lr, weight_decay=5e-3)

    print('Learning rate: %f' % (lr))
    print('Number of epoch: %f' % (epoch_num))

    for epoch in range(epoch_num):
        
        # loop over the dataset multiple times

        running_loss = 0.0
        for i, data in enumerate(trainloader, 0):
            # get the inputs
            inputs, labels = data
            inputs, labels = inputs.to(device, dtype=torch.float), labels.to(device)  # Send to GPU
            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()
        print('Epoch %d loss: %.3f' %
            (epoch + 1, running_loss))

    print('Finished Training')

    ########################################################################
    # Calculate accuracy

    correct_train = 0
    correct_test = 0
    total_train = 0
    total_test = 0

    with torch.no_grad():
        for data in trainloader:
            images, labels = data
            images, labels = images.to(device, dtype=torch.float), labels.to(device)  # Send to GPU
            outputs = net(images)
            _, predicted = torch.max(outputs.data, 1)
            total_train += labels.size(0)
            correct_train += (predicted == labels).sum().item()

        for data in testloader:
            images, labels = data
            images, labels = images.to(device, dtype=torch.float), labels.to(device)  # Send to GPU
            outputs = net(images)
            _, predicted = torch.max(outputs.data, 1)
            total_test += labels.size(0)
            correct_test += (predicted == labels).sum().item()

    print('Train accuracy: %d %%' % (100 * correct_train / total_train))
    print('Test accuracy: %d %%' % (100 * correct_test / total_test))

    save = input('Save model? [y/N]: ')
    if save.lower() == 'y':
        torch.save(net.state_dict(), 'model')


if __name__ == "__main__":
    dataset = load_dataset()
    train(dataset)
