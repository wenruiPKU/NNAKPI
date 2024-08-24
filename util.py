import numpy as np
import torch
import os


# 根据1组预训练权重复制到不同的分支，使得每个分支都是经过预训练的并且权重完全相同
def init_net(model, n_layers, n_neurons, n_TACs):
    pth = torch.load('result/wt/init/molde_%s_n_layers_%d_n_neurons_%d(init).pth' % (model, n_layers, n_neurons), map_location=torch.device("cpu"))
    print(pth)

    for layer in range(n_layers):
        weight = torch.zeros_like(pth['branch_2.%d.weight' % layer]).repeat(n_TACs, 1, 1)
        bias = torch.zeros_like(pth['branch_2.%d.bias' % layer]).repeat(n_TACs)
        for TAC in range(n_TACs):
            if layer != n_layers - 1:
                weight[TAC * n_neurons: (TAC + 1) * n_neurons, :, :] = pth['layers_2.%d.weight' % layer]
                bias[TAC * n_neurons: (TAC + 1) * n_neurons] = pth['layers_2.%d.bias' % layer]
            else:
                weight[TAC * 3: (TAC + 1) * 3, :, :] = pth['layers_2.%d.weight' % layer]
                bias[TAC * 3: (TAC + 1) * 3] = pth['layers_2.%d.bias' % layer]
        # 将原来pth变量中的数值更换
        pth['branch_2.%d.weight' % layer] = weight
        pth['branch_2.%d.bias' % layer] = bias

    parameters = torch.load('result/kps/init/molde_%s_n_layers_%d_n_neurons_%d_iter_%d(init).pth' % (model, n_layers, n_neurons, iter), map_location=torch.device("cpu")).repeat(1, n_TACs)
    return pth, parameters

def generate_folder(path):
    folder = os.path.exists(path)
    if not folder:
        os.makedirs(path)

def print_error(value):
    print("error: ", value)