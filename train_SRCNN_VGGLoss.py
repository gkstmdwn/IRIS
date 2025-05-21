import os
import copy

import torch
import torch.optim as optim
import torch.backends.cudnn as cudnn
from torch import nn
from torch.utils.data.dataloader import DataLoader
from tqdm import tqdm

from SRCNN.models import SRCNN
from SRCNN.datasets import TrainDataset, EvalDataset
from SRCNN.utils import AverageMeter, calc_psnr
from LossFunctions.VGGLossNetwork import VGGPerceptual

if __name__ == "__main__":
    outputs_dir = r"./output/SRCNN_VGGLoss"
    eval_file_dir = r"./data/HIT-UAV Dataset/images/val"
    train_file_dir = r"./data/HIT-UAV Dataset/images/train"
    lr = 1e-4
    batch_size = 16
    num_epochs = 400
    num_workers = 8
    lambda_vgg = 0.01
    
    
    if not os.path.exists(outputs_dir):
        os.makedirs(outputs_dir)
    
    cudnn.benchmark = True
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    
    model = SRCNN().to(device)
    vgg_loss = VGGPerceptual.VGGLoss(device = device)
    pixel_criterion = nn.MSELoss()
    optimizer = optim.Adam([
        {'params': model.conv1.parameters()},
        {'params': model.conv2.parameters()},
        {'params': model.conv3.parameters(), 'lr': lr * 0.1}
    ], lr=lr)
        
    train_dataset = TrainDataset(train_file_dir)
    train_dataloader = DataLoader(dataset=train_dataset,
                                  batch_size=batch_size,
                                  shuffle=True,
                                  num_workers=num_workers,
                                  pin_memory=True,
                                  drop_last=True)
    eval_dataset = EvalDataset(eval_file_dir)
    eval_dataloader = DataLoader(dataset=eval_dataset, batch_size=1)
    
    best_weights = copy.deepcopy(model.state_dict())
    best_epoch = 0
    best_psnr = 0.0
    
    for epoch in tqdm(range(num_epochs)):
        model.train()
        epoch_losses = AverageMeter()
        
        for data in train_dataloader:
            inputs, labels = data
            
            inputs = inputs.to(device)
            labels = labels.to(device)
            
            preds = model(inputs)

            pixel_loss = pixel_criterion(preds, labels)
            perceptual_loss = vgg_loss(preds, labels)
            loss = pixel_loss + lambda_vgg * perceptual_loss

            epoch_losses.update(loss.item(), len(inputs))
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
        torch.save(model.state_dict(), os.path.join(outputs_dir, 'epoch_{}.pth'.format(epoch)))
        
        
        model.eval()
        epoch_psnr = AverageMeter()

        for data in eval_dataloader:
            inputs, labels = data

            inputs = inputs.to(device)
            labels = labels.to(device)

            with torch.no_grad():
                preds = model(inputs).clamp(0.0, 1.0)

            epoch_psnr.update(calc_psnr(preds, labels), len(inputs))

        print('eval psnr: {:.2f}'.format(epoch_psnr.avg))

        if epoch_psnr.avg > best_psnr:
            best_epoch = epoch
            best_psnr = epoch_psnr.avg
            best_weights = copy.deepcopy(model.state_dict())

    print('best epoch: {}, psnr: {:.2f}'.format(best_epoch, best_psnr))
    torch.save(best_weights, os.path.join(outputs_dir, 'best.pth'))