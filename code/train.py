import argparse
from statistics import mean
import torch.nn as nn
import torch.optim as optim
import torch
import torchvision.models as models
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from data_utils import ImageAndPathDataset, ImageAndPathDataLoader
from torchvision.datasets.folder import ImageFolder, default_loader, IMG_EXTENSIONS 
#from model import Classifieur
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold
import os
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import numpy as np


def train(net, optimizer, loader, writer, epochs=10):
    criterion = nn.CrossEntropyLoss()
    for epoch in range(epochs):
        running_loss = []
        t = tqdm(loader)
        for x, y in t:
            x, y = x.to(device), y.to(device)
            outputs = net(x)
            loss = criterion(outputs, y)
            running_loss.append(loss.item())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            t.set_description(f'training loss: {mean(running_loss)}')
        writer.add_scalar('training loss', mean(running_loss), epoch)

def test(model, dataloader, writer):
    test_corrects = 0
    total = 0
    with torch.no_grad():
        for x, y in dataloader:
            x = x.to(device)
            y = y.to(device)
            y_hat = model(x).argmax(1)
            test_corrects += y_hat.eq(y).sum().item()
            total += y.size(0)
    return test_corrects / total


if __name__=='__main__':
    parser = argparse.ArgumentParser()
	
    parser.add_argument('--exp_name', type=str, default = 'Training_Movie', help='experiment name')
    parser.add_argument('--batch_size', type=int, default=16, help='batch size')
    parser.add_argument('--lr', type=float, default=0.01, help='learning rate')
    parser.add_argument('--epochs', type=int, default=10, help='number of epochs')

    args = parser.parse_args()
    exp_name = args.exp_name
    epochs = args.epochs
    batch_size = args.batch_size
    lr = args.lr

    writer = SummaryWriter(f'runs/{exp_name}')

    # transforms
    transform = transforms.Compose(
        [transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,)),
        transforms.Resize((224, 224))
        ])

    # dataset
    dataset = ImageFolder(root='./../MovieGenre/content/sorted_movie_posters_paligema', transform=transform)
    #trainset, testset = train_test_split(dataset, test_size=0.2, random_state=42)
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    train_idx, test_idx = next(skf.split(dataset.samples, dataset.targets))
    trainset = torch.utils.data.Subset(dataset, train_idx)
    testset = torch.utils.data.Subset(dataset, test_idx)


    # dataloaders
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=2)
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=2)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # model
    net = models.mobilenet_v3_small(pretrained=True)
    net.classifier = nn.Linear(576, 10)


    #settign net on device
    net.to(device)    
    optimizer = optim.Adam(net.parameters(), lr=lr)

    train(net, optimizer, trainloader, writer, epochs=epochs)
    test_acc = test(net, testloader, writer)
    print(f'test accuracy: {test_acc}')

    torch.save(net.state_dict(), 'weights/movie_net.pth')

    """
    #add embeddings to tensorboard
    perm = torch.randperm(len(trainset.dataset.samples))
    images, labels = trainset.dataset.samples[perm][:256], trainset.dataset.targets[perm][:256]
    images = images.unsqueeze(1).float().to(device)
    with torch.no_grad():
        embeddings = net.get_features(images)
    writer.add_embedding(embeddings, 
            metadata=labels, 
            label_img=images, global_step=1)

    #save networks computational graph in tensorboard
    writer.add_graph(net, images)
    #save a dataset sample in tensorboard
    img_grid = torchvision.utils.make_grid(images[:64])
    writer.add_image('movie_images', img_grid)
    """
