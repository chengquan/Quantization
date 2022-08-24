from re import X
import numpy

from datasets import get_dataset
from trainer import train, eval
from quant.quant_layers import QuantifiedLinear, QuantifiedConv2d, gathered
import os
import time
import torch
import cfg
from cfg import *
#os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2,3'
#import sys; print('Python %s on %s' % (sys.version, sys.platform))
#sys.path.extend('/home/QuantNAS')


os.environ['CUDA_VISIBLE_DEVICES'] = '0'
SAVE_PATH = './run/mnist_nas_%s' % time.strftime("%Y%m%d_%H%M%S")


def model_train():
    from models.multprec.stub import LeNet
    model = LeNet()
    #model.load_state_dict(torch.load('/home/qcheng/***/ckpt/99.pth').state_dict())
    train_set, test_set = get_dataset()
    train(model, train_set, test_set, SAVE_PATH)
    global gathered
    #please set a debug point here and dump the 'gathered' var using pickle
    hook = 0

def model_eval():
    from models.multprec.stub_test import LeNet
    model = LeNet()
    model.load_state_dict(torch.load('/home/TCASII/Quantization/experiment/mnist_nas/run/mnist_nas_***/ckpt/29.pth').state_dict())
    train_set, test_set = get_dataset()
    cfg.EPOCHS = 1
    cfg.LR = 1e-6
    #for k in modell.keys():
    #    print(k)
    # train(model, train_set, test_set, SAVE_PATH)
    eval(model, train_set, test_set, SAVE_PATH)
    global gathered
    hook=0
    # train(model, test_set, test_set)


def dump_file():
    import pickle
    global gathered
    fw = open("name.dumpfile", "wb")
    pickle.dump(gathered, fw)
    fw.close()
    parse_hand_dumped_file('name.dumpfile', './data/export')

def parse_hand_dumped_file(dumpfile_path, export_path):
    import pickle
    f = open(dumpfile_path, 'rb')
    os.makedirs(export_path, exist_ok=True)
    scales = open(os.path.join(export_path, "cfg.txt"), 'w')
    scales.write("layer\t\tname\t\tbit_width\t\tscale\n")
    d = pickle.load(f)
    for i, k in enumerate(d.keys()):
        v = d[k]
        for t in v:
            name, bw, scale, data, ori_data = t
            # name, bw, scale, data = t
            scale = -int(scale.detach().cpu())
            data = data.detach().cpu().numpy().astype('int')
            ori_data = ori_data.detach().cpu().numpy()
            if 'input' in name or 'output' in name:
                data = data[0:1, ...]
                ori_data = ori_data[0:1, ...]
            scales.write("%d\t\t%s\t\t%d\t\t%d\n" % (i, name, bw, scale))
            numpy.savetxt(os.path.join(export_path, '%d_%s.txt' % (i, name)), data.reshape(-1), fmt='%d')
            numpy.savetxt(os.path.join(export_path, '%d_%s_fp.txt' % (i, name)), ori_data.reshape(-1))


if __name__ == '__main__':
    #parse_hand_dumped_file('./***.dump', './data/')
    SEED = 1
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    # 1st step
    model_train()
    # 2nd step
    #model_eval()
