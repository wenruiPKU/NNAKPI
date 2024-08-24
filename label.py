import torch
import numpy as np
from model import neural_network
from train import select_index, coefficient_matrix_generator
from util import generate_folder
import os
import re
from tqdm import tqdm
import matplotlib.pyplot as mp
import random

# 自动确定有多少批次多少GPU参与到其中
def find_max_min(file_list):
    # 正则表达式匹配文件名
    pattern = re.compile(r'(\d+)_(\d+)\.pth')
    max_x = -1
    max_y = -1
    for filename in file_list:
        match = pattern.match(filename)
        if match:
            x = int(match.group(1))
            y = int(match.group(2))
            if x > max_x:
                max_x = x
            if y > max_y:
                max_y = y
    return max_x + 1, max_y + 1



if __name__ == '__main__':

    serial_numbers = ['1.0.0']#['0.0.0', '0.1.0']
    n_cols = 1726
    device = torch.device('cuda:0')
    t_mid = torch.load('../input/t_mid/0.0.0.pth')  # 读取包含0的midpoint数据
    W = coefficient_matrix_generator(0, t_mid[-1], n_cols)

    for serial_number in serial_numbers:
    
        segment = torch.load('../../data/segment/%s.pth' % serial_number)
        n_voxels = int(torch.sum(segment))
        C_0_col = torch.load('../input/C_0/%s(col).pth' % serial_number)
        scale = torch.max(C_0_col)
        C_Ts_mid = torch.load('../input/C_T/%s(mid).pth' % serial_number)

        S_0s_col = torch.zeros(size=[n_cols, n_voxels])
        S_1s_col = torch.zeros(size=[n_cols, n_voxels])
        S_2s_col = torch.zeros(size=[n_cols, n_voxels])
        parameters = torch.zeros(size=[4, n_voxels])
        
        n_batches, n_gpus = find_max_min(os.listdir('weight/%s' % serial_number))

        # tm: the moment
        f = 0
        for batch in tqdm(range(n_batches), desc='Processing labels for data %s' % serial_number):
            for gpu in range(n_gpus):

                # 确认一下这一批有多少voxel被计算进来，末轮的数量与前面的数量是不同的
                parameters_tm = torch.load('parameters/%s/%d_%d.pth' % (serial_number, batch, gpu), map_location=device)
                n_voxels_tm = parameters_tm.shape[1]
                g = f + n_voxels_tm
                t_col = torch.linspace(start=0, end=t_mid[-1], steps=n_cols, device=device).reshape(-1, 1, 1)
                C_T_index_list, C_1_index_list, C_2_index_list = select_index(n_voxels_tm)

                net = neural_network(4, 64, n_voxels_tm).to(device)
                weight = torch.load('weight/%s/%d_%d.pth' % (serial_number, batch, gpu), map_location=device)
                net.load_state_dict({k.replace('module.', ''): v for k, v in weight.items()})
                pre_1 = torch.squeeze(net(t_col, 1), dim=2)
                pre_2 = torch.squeeze(net(t_col.repeat(1, n_voxels_tm, 1), 2), dim=2)
                S_0s_col[:, f: g] = pre_1.repeat(1, n_voxels_tm).cpu().detach()  # 需要加上detach()，否则会爆显存
                S_1s_col[:, f: g] = pre_2[:, C_1_index_list].cpu().detach()  
                S_2s_col[:, f: g] = pre_2[:, C_2_index_list].cpu().detach()
                parameters[:, f: g] = parameters_tm[0: 4].cpu().detach()
                f = g
        
        S_0s_col *= scale
        S_1s_col *= scale
        S_2s_col *= scale
        
        # 可视化部分结果(我们并不关心S_1和S_2长什么样，我们只关心拟合C_T的结果以及动力学系统是否自洽)
        generate_folder('demo/%s' % serial_number)
        t_col = torch.linspace(start=0, end=t_mid[-1], steps=n_cols)
        for i in range(100):
            j = random.randint(0, n_voxels - 1)
            [fv, K1, k2, k3] = parameters[:, j]
            mp.scatter(t_mid, C_Ts_mid[j, :], label='meas')
            mp.plot(t_col, fv * S_0s_col[:, j] + (1 - fv) * (S_1s_col[:, j] + S_2s_col[:, j]), label='S_T')
            mp.plot(t_col, torch.mm(W, S_1s_col[:, j: j + 1]), label='dS_1(left)')
            mp.plot(t_col, torch.mm(W, S_2s_col[:, j: j + 1]), label='dS_2(left)')
            mp.plot(t_col, K1 * S_0s_col[:, j] - (k2 + k3) * S_1s_col[:, j], label='dS_1(right)', linestyle='--')
            mp.plot(t_col, k3 * S_1s_col[:, j], label='dS_2(right)', linestyle='--')
            mp.title('fv=%.4f, K1=%.4f, k2=%.4f, k3=%.4f' % (fv, K1, k2, k3))
            mp.legend()
            mp.savefig('demo/%s/%d.png' % (serial_number, i), bbox_inches='tight')
            mp.close()

    
        # 保存文件
        generate_folder('../input/kps')
        generate_folder('../input/C_1/%s' % serial_number)
        generate_folder('../input/C_2/%s' % serial_number)

        torch.save(parameters, '../input/kps/%s.pth' % serial_number)
        S_1s_col = np.array(S_1s_col.cpu())
        S_2s_col = np.array(S_2s_col.cpu())
        for i_col in tqdm(range(n_cols), desc='Saving labels for data %s' % serial_number):
            np.save('../input/C_1/%s/%d.npy' % (serial_number, i_col), S_1s_col[i_col, :])
            np.save('../input/C_2/%s/%d.npy' % (serial_number, i_col), S_2s_col[i_col, :])
            

