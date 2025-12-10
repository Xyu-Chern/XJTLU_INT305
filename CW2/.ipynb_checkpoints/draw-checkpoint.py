import wandb
import torch
import torchvision
import torchvision.transforms as transforms
import numpy as np
import matplotlib.pyplot as plt

transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

class_names =['airplane', 'automobile', 'bird', 'cat', 'deer',
           'dog', 'frog', 'horse', 'ship', 'truck']

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

train_on_gpu = torch.cuda.is_available() 

def imshow(img):
    img = img / 2 + 0.5    
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))


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

    model , optimizer, scheduler = set(args.net, args.optimizer)
    
    model.load_state_dict(torch.load(f'{args.net}.pt')) 

    test_loss = 0.0
    class_correct = list(0. for i in range(10))
    class_total = list(0. for i in range(10))

    correctly_classified_samples = []
    misclassified_samples = []
    # -----------------------------------------------

    model.eval()

    with torch.no_grad(): 
        # iterate over test data
        for data, target in testloader:
            
            # move tensors to GPU if CUDA is available
            data, target = data.to(device), target.to(device)

            # forward pass: compute predicted outputs by passing inputs to the model
            output = model(data)
            
            # calculate the batch loss
            loss = criterion(output, target)
            
            # update test loss 
            test_loss += loss.item()*data.size(0)
            probs = F.softmax(output, dim=1) 
            max_probs, pred = torch.max(probs, 1)
            # ------------------------------------

            # compare predictions to true label
            correct_tensor = pred.eq(target.data.view_as(pred))
            correct = np.squeeze(correct_tensor.cpu().numpy())
            
            for i in range(data.size(0)):
                label = target.data[i].item() 
                predicted_label = pred[i].item() 
                confidence = max_probs[i].item() 
                is_correct = correct[i].item()
                
                class_correct[label] += is_correct
                class_total[label] += 1
                
                sample_info = {
                    'image': data[i].cpu(), 
                    'true_label': label,
                    'pred_label': predicted_label,
                    'confidence': confidence
                }

                if is_correct and len(correctly_classified_samples) < 10:
                    correctly_classified_samples.append(sample_info)
                elif not is_correct and len(misclassified_samples) < 10:
                    misclassified_samples.append(sample_info)

        # average test loss
        test_loss = test_loss/len(testloader.dataset)
        print('Test Loss: {:.6f}\n'.format(test_loss))

        print("--- Classification Metrics ---")
        for i in range(10):
            if class_total[i] > 0:
                accuracy = 100 * class_correct[i] / class_total[i]
                print('Test Accuracy of %10s: %5.2f%% (%4d/%4d)' % (
                    class_names[i], accuracy, 
                    np.sum(class_correct[i]), np.sum(class_total[i])))
            else:
                print('Test Accuracy of %10s: N/A' % (class_names[i]))

        overall_acc = 100. * np.sum(class_correct) / np.sum(class_total)
        print('\nTest Accuracy (Overall): %5.2f%% (%4d/%4d)' % (
            overall_acc, np.sum(class_correct), np.sum(class_total)))


    fig_correct = plt.figure(figsize=(15, 3))
    fig_correct.suptitle(f'Well Classified Samples (Total Accuracy: {overall_acc:.2f}%)', fontsize=16)
    for idx, sample in enumerate(correctly_classified_samples):
        ax = fig_correct.add_subplot(1, len(correctly_classified_samples), idx + 1, xticks=[], yticks=[])
        imshow(sample['image'])
        title = f"P: {class_names[sample['pred_label']]}\nC: {sample['confidence']:.2f}"
        ax.set_title(title, color="green", fontsize=8)
    plt.tight_layout()
    fig_correct.savefig('well_classified_samples.png', bbox_inches='tight')


    fig_misclass = plt.figure(figsize=(15, 3))
    fig_misclass.suptitle(f'Misclassified Samples (Total Accuracy: {overall_acc:.2f}%)', fontsize=16)
    for idx, sample in enumerate(misclassified_samples):
        ax = fig_misclass.add_subplot(1, len(misclassified_samples), idx + 1, xticks=[], yticks=[])
        imshow(sample['image'])
        title = f"P: {class_names[sample['pred_label']]}\nC: {sample['confidence']:.2f}\nGT: {class_names[sample['true_label']]}"
        ax.set_title(title, color="red", fontsize=8)
    plt.tight_layout()
    fig_misclass.savefig('misclassified_samples.png', bbox_inches='tight')