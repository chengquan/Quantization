import logging
import os
import sys
import time

import torch
import torch.nn as nn
from torch.optim import SGD
from torch.utils.data import DataLoader
from torchvision.datasets.cifar import CIFAR10
from torchvision.transforms import Compose, ToTensor, Normalize, Resize
from cfg import *
from models.nas.mixed_precision import SymMixedPrecisionConv2d, SymMixedPrecisionLinear

def loss_rt(loss, model):
    loss_temp = 0
    flops = 0
    for i, p in model.named_modules():
        if isinstance(p, SymMixedPrecisionConv2d) or isinstance(p, SymMixedPrecisionLinear):
            loss_temp += p.branch_cost() * p.flops
            flops += p.flops
    if flops: 
        return loss + loss_temp / flops
    else:
        return loss


def train(model, train_set, test_set, SAVE_PATH):
    os.makedirs(SAVE_PATH, exist_ok=True)
    os.makedirs(os.path.join(SAVE_PATH, "code"), exist_ok=True)
    os.makedirs(os.path.join(SAVE_PATH, "ckpt"), exist_ok=True)
    #os.system('cp -r ./experiment ./models ./quant %s' % os.path.join(SAVE_PATH, 'code'))

    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)
    sh = logging.StreamHandler(stream=sys.stdout)
    sh.setLevel(logging.DEBUG)
    fh = logging.FileHandler(os.path.join(SAVE_PATH, './log.log'))
    fh.setLevel(logging.DEBUG)
    logger.addHandler(sh)
    logger.addHandler(fh)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    if torch.cuda.device_count() > 1:
        model = nn.parallel.DataParallel(model)

    # transformer = Compose([
    #     Resize(INPUT_SIZE),
    #     ToTensor(),
    #     Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
    # train_set = CIFAR10('./data/cifar10', train=True, download=True, transform=transformer)
    # test_set = CIFAR10('./data/cifar10', train=True, download=True, transform=transformer)
    train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True)
    test_loader = DataLoader(test_set, batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True)

    loss_fn = nn.CrossEntropyLoss()
    optimizer = OPTIMIZER(model.parameters(), lr=LR, momentum=MOMENTUM)

    for epoch in range(EPOCHS):
        logger.info("Epoch = %d" % epoch)
        for step, (x, target) in enumerate(train_loader):
            optimizer.zero_grad()
            x = x.float().to(device)
            target = target.long().to(device)
            output = model(x)
            loss = loss_fn(output, target)
            logger.debug("[Train %d:%d] loss=%.6f" % (epoch, step, float(loss.detach().cpu())))
            loss.backward()
            optimizer.step()

        if epoch % SAVE_INTERVAL == SAVE_INTERVAL - 1:
            if isinstance(model, nn.DataParallel):
                torch.save(model.module.state_dict(), os.path.join(SAVE_PATH, 'ckpt', "%d.pth" % epoch))
            else:
                torch.save(model, os.path.join(SAVE_PATH, 'ckpt', "%d.pth" % epoch))
                #torch.save(model.state_dict(), os.path.join(SAVE_PATH, 'ckpt', "%d.pth" % epoch))

        if epoch % EVAL_INTERVAL == EVAL_INTERVAL - 1:
            model.eval()
            correct = 0
            total = 0
            for step, (x, target) in enumerate(test_loader):
                x = x.float().to(device)
                target = target.long().to(device)
                output = model(x)
                loss = loss_fn(output, target)
                logger.info("[Test %d:%d] loss=%.6f" % (epoch, step, float(loss.detach().cpu())))
                correct += int((output.argmax(dim=1) == target).sum().cpu())
                total += target.shape[0]
            logger.info("[Test %d] acc=%.6f" % (epoch, correct * 1.0 / total))
            optimizer.zero_grad()
            model.train()


def eval(model, train_set, test_set, SAVE_PATH):
    os.makedirs(SAVE_PATH, exist_ok=True)
    os.makedirs(os.path.join(SAVE_PATH, "code"), exist_ok=True)
    os.makedirs(os.path.join(SAVE_PATH, "ckpt"), exist_ok=True)

    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)
    sh = logging.StreamHandler(stream=sys.stdout)
    sh.setLevel(logging.DEBUG)
    fh = logging.FileHandler(os.path.join(SAVE_PATH, './log.log'))
    fh.setLevel(logging.DEBUG)
    logger.addHandler(sh)
    logger.addHandler(fh)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    if torch.cuda.device_count() > 1:
        model = nn.parallel.DataParallel(model)
    train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True)
    test_loader = DataLoader(test_set, batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True)
    loss_fn = nn.CrossEntropyLoss()
    for epoch in range(EPOCHS):
        logger.info("Epoch = %d" % epoch)
        if epoch % EVAL_INTERVAL == EVAL_INTERVAL - 1:
            model.eval()
            correct = 0
            total = 0
            for step, (x, target) in enumerate(test_loader):
                x = x.float().to(device)
                target = target.long().to(device)
                output = model(x)
                loss = loss_fn(output, target)
                logger.info("[Test %d:%d] loss=%.6f" % (epoch, step, float(loss.detach().cpu())))
                correct += int((output.argmax(dim=1) == target).sum().cpu())
                total += target.shape[0]
            logger.info("[Test %d] acc=%.6f" % (epoch, correct * 1.0 / total))
