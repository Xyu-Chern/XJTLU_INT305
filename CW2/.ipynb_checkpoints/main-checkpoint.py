import wandb
import torch
import torchvision
import torchvision.transforms as transforms

from PIL import Image 

import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

criterion = nn.CrossEntropyLoss()

def set(net_name, optimizer_name):
    if net_name =="base":
        from model.base import Net
    elif net_name =="resnet":
        from model.resnet import Net
    elif net_name =="deep":
        from model.deepnet import Net
    net = Net().to(device)

    if optimizer_name =="Adam":
        optimizer = optim.Adam(net.parameters(), lr=0.01, betas=(0.9, 0.999), eps=1e-08, weight_decay=0, amsgrad=False)
    else:
        optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=10, min_lr=0)

    return net , optimizer, scheduler

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='Train CIFAR-10 classifier')
    parser.add_argument('--net', type=str, default='base', help='Model name (base/resnet/deep)')
    parser.add_argument('--optimizer', type=str, default='Adam', help='Optimizer (SGD/Adam)')
    args = parser.parse_args()

    net , optimizer, scheduler = set(args.net, args.optimizer)
    
    train_transform = transforms.Compose(
    [transforms.RandomHorizontalFlip(p=0.5),
     transforms.RandomAffine(degrees=(-5, 5), translate=(0.1, 0.1), scale=(0.9, 1.1), interpolation=Image.BILINEAR),
     transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    transform = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    if args.net =="deep":
        dataset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                                download=True, transform=train_transform)
        print("new transform")
    else:
        dataset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                        download=True, transform=transform)
        
    trainset, valset = torch.utils.data.random_split(dataset, [40000, 10000])
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=1024,
                                              shuffle=True, num_workers=2)
    valloader = torch.utils.data.DataLoader(valset, batch_size=1024,
                                              shuffle=False, num_workers=2)

    testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                           download=True, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=1024,
                                             shuffle=False, num_workers=2)

    classes = ('plane', 'car', 'bird', 'cat',
               'deer', 'dog', 'frog', 'horse', 'ship', 'truck')


    # Initialize wandb
    wandb.init(project="cifar10-classification", 
            config={
                "net": args.net,
                "optimizer": args.optimizer,
                "batch_size": 1024,
                "epochs": 100
            })
    
    # Log model architecture to wandb
    wandb.watch(net, criterion, log="all", log_freq=10)

    for epoch in range(100):
        net.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        for data in trainloader:
            batch, labels = data
            batch, labels = batch.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = net(batch)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # compute training statistics
            running_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        avg_loss = running_loss / len(trainloader)
        avg_acc = correct / total
        
        # Validation
        net.eval()
        loss_val = 0.0
        correct_val = 0
        total_val = 0
        
        with torch.no_grad():
            for data in valloader:
                batch, labels = data
                batch, labels = batch.to(device), labels.to(device)
                outputs = net(batch)
                loss = criterion(outputs, labels)
                loss_val += loss.item()
                _, predicted = torch.max(outputs, 1)
                total_val += labels.size(0)
                correct_val += (predicted == labels).sum().item()

        avg_loss_val = loss_val / len(valloader)
        avg_acc_val = correct_val / total_val
        
        # Update scheduler
        scheduler.step(avg_loss_val)
        
        # Log to wandb
        wandb.log({
            "epoch": epoch + 1,
            "train_loss": avg_loss,
            "train_acc": avg_acc,
            "val_loss": avg_loss_val,
            "val_acc": avg_acc_val,
            "learning_rate": optimizer.param_groups[0]['lr']
        })
        
        print('[epoch %d] loss: %.5f accuracy: %.4f val loss: %.5f val accuracy: %.4f lr: %.6f' % 
              (epoch + 1, avg_loss, avg_acc, avg_loss_val, avg_acc_val, optimizer.param_groups[0]['lr']))
        
        # Save model checkpoint
        torch.save(net.state_dict(), f'{args.net}.pt')
        wandb.save(f'{args.net}.pt')
    
    # Final test evaluation
    net.eval()
    test_correct = 0
    test_total = 0
    
    with torch.no_grad():
        for data in testloader:
            batch, labels = data
            batch, labels = batch.to(device), labels.to(device)
            outputs = net(batch)
            _, predicted = torch.max(outputs, 1)
            test_total += labels.size(0)
            test_correct += (predicted == labels).sum().item()
    
    test_acc = test_correct / test_total
    print(f'\nTest Accuracy: {test_acc:.4f}')
    
    # Log test accuracy
    wandb.log({"test_acc": test_acc})
    wandb.finish()