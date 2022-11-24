from semg_dataloader import get_loader
from models.multi_view_cnn import Multi_View_CNN

import os
import numpy as np
import matplotlib
matplotlib.use('agg')
from matplotlib import pyplot as plt

import torch
from torch.nn import CrossEntropyLoss
from torch.optim import Adam
from torch.optim.lr_scheduler import MultiStepLR

from torchmetrics import Accuracy

def train_epoch(model, dataloader, loss_f, optimizer, epoch, acc_computer, device):

    model.train()
    
    epoch_loss = 0.
    
    for batch, (datas, labels) in enumerate(dataloader):
        
        optimizer.zero_grad()
        
        datas = datas.to(device)
        labels = labels.to(device)
        
        output = model(datas)
        output_prob = torch.softmax(output, dim=1)
        
        loss = loss_f(output, labels)
        
        loss.backward()
        
        optimizer.step()
        
        acc = acc_computer(output, labels)
        epoch_loss += loss.cpu().item()
        
    acc = acc_computer.compute().cpu().item()
    print('Train Epoch[{}]: Loss[{:.04f}] Acc[{:.04f}]'.format(epoch, epoch_loss, acc))
    
    return epoch_loss, acc

def validation(model, dataloader, loss_f, acc_computer, device):
    
    model.eval()
    
    val_loss = 0.
    for batch, (datas, labels) in enumerate(dataloader):
        
        datas = datas.to(device)
        labels = labels.to(device)
        
        output = model(datas)
        output_prob = torch.softmax(output, dim=1)
        
        val_loss += loss_f(output, labels).cpu().item()
        acc = acc_computer(output, labels)
        
    acc = acc_computer.compute().cpu().item()
    
    print('Validation: Loss[{:.04f}] Acc[{:.04f}]\n'.format(val_loss, acc))
    
    return val_loss, acc

def test(model, dataloader, loss_f, acc_computer, acc_computer_none, device, weights_path):
    
    model.load_state_dict(torch.load(weights_path))
    
    model.eval()
    
    test_loss = 0.
    
    for batch, (datas, labels) in enumerate(dataloader):
        
        datas = datas.to(device)
        labels = labels.to(device)
        
        output = model(datas)
        output_prob = torch.softmax(output, dim=1)
        
        test_loss += loss_f(output, labels).cpu().item()
        acc = acc_computer(output, labels)
        acc_ = acc_computer_none(output, labels)
        
    acc_ = acc_computer_none.compute().cpu()
    print(acc_)
    acc = acc_computer.compute().cpu().item()
    print('Test: Loss[{:.04f}] Acc[{:.04f}]'.format(test_loss, acc))
    
    return test_loss, acc

def pipeline(num_epochs, device_idx, experiment_tag):
    
    if not os.path.exists(os.path.join('experiments', experiment_tag)):
        os.makedirs(os.path.join('experiments', experiment_tag))
    
    os.environ['CUDA_VISIBLE_DEVICES'] = device_idx
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    model = Multi_View_CNN(16, 11).to(device)
    
    acc_computer = Accuracy().to(device)
    acc_computer_none = Accuracy(num_classes=11, average=None).to(device)
    lr = 1e-3
    
    optimizer = Adam(model.parameters(), lr)
    
    loss_f = CrossEntropyLoss()
    
    train_loader = get_loader(dataroot=r'Data\train_data', batch_size=256, mode='train', num_workers=0)
    val_loader = get_loader(dataroot=r'Data\val_data', batch_size=256, mode='val', num_workers=0)
    test_loader = get_loader(dataroot=r'Data\test_data', batch_size=256, mode='test', num_workers=0)
    
    train_loss_record = {'epoch': [], 'loss': []}
    val_loss_record = {'epoch': [], 'loss': []}
    train_acc_record = {'epoch': [], 'acc': []}
    val_acc_record = {'epoch': [], 'acc': []}
    
    best_acc = 0.
    best_epoch = 1
    
    epoch = 1
    while epoch <= num_epochs:
        
        train_loss, train_acc = train_epoch(model,train_loader, loss_f, optimizer, epoch, acc_computer, device)
        train_loss_record['epoch'].append(epoch)
        train_loss_record['loss'].append(train_loss)
        train_acc_record['epoch'].append(epoch)
        train_acc_record['acc'].append(train_acc)
        
        val_loss, val_acc = validation(model, val_loader, loss_f, acc_computer, device)
        val_loss_record['epoch'].append(epoch)
        val_loss_record['loss'].append(val_loss)
        val_acc_record['epoch'].append(epoch)
        val_acc_record['acc'].append(val_acc)
        
        torch.save(model.state_dict(), os.path.join('experiments', experiment_tag, 'epoch_{:03d}.pt'.format(epoch)))
        
        if val_acc > best_acc:
            best_acc = val_acc
            best_epoch = epoch
        
        test_loss, test_acc = test(model, test_loader, loss_f, acc_computer, acc_computer_none, device, weights_path=os.path.join(os.path.join('experiments', experiment_tag, 'epoch_{:03d}.pt'.format(best_epoch))))
        
        epoch += 1
    
        
    test_loss, test_acc = test(model, test_loader, loss_f, acc_computer, acc_computer_none, device, weights_path=os.path.join(os.path.join('experiments', experiment_tag, 'epoch_{:03d}.pt'.format(best_epoch))))    
    
    
    # plot curve
    fig, ax = plt.subplots()
    x = np.linspace(1, num_epochs, num_epochs)
    train_loss = np.array(train_loss_record['loss'])
    val_loss = np.array(val_loss_record['loss'])
    ax.plot(x, train_loss, label='train_loss')
    ax.plot(x, val_loss, label='val_loss')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.set_title('test_loss: {:.4f}'.format(test_loss))
    ax.legend()
    plt.savefig(os.path.join('experiments', experiment_tag, 'loss.png'))
    
    fig, ax = plt.subplots()
    train_acc = np.array(train_acc_record['acc'])
    val_acc = np.array(val_acc_record['acc'])
    ax.plot(x, train_acc, label='train_acc')
    ax.plot(x, val_acc, label='val_acc')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Accuracy')
    ax.set_title('test_acc: {:.4f}'.format(test_acc))
    ax.legend()
    plt.savefig(os.path.join('experiments', experiment_tag, 'acc.png'))
    

if __name__ == '__main__':
    
    pipeline(600, "0", "600epoch-100-overla0.3")

    print()