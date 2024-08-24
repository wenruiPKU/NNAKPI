import numpy as np
import time
import multiprocessing as mpc
from tqdm import tqdm
from util import generate_folder, print_error
from train import train
import torch


if __name__ == '__main__':
    mpc.set_start_method('spawn')

    # 训练参数的设置
    unit_and_serial_numbers = {'UCD': ['0.0.0', '0.1.0', '1.0.0', '1.1.0', '2.0.0', '2.1.0', '3.0.0', '3.1.0', '4.0.0', '4.1.0']}
    n_layers = 4
    n_neurons = 64
    n_iters = 1
    lr = 0.01
    lr_decay_step = n_iters
    lr_decay_gamma = 0.5
    standard_len = 8192
    max_gpus = 1  # 设置使用 GPU 的上限
    model = '2.0'
    kernel_size = 5

    term = 30  # UCD: 30
    n_cols = 1726  # UCD: 1726

    # 获取可用 GPU 数量，并设置使用 GPU 的上限
    available_gpus = torch.cuda.device_count()
    gpus_to_use = min(max_gpus, available_gpus)
    print(f'Using {gpus_to_use} out of {available_gpus} available GPUs')

    # 创建保存权重与动力学参数的文件夹
    generate_folder('result/wt/init')
    generate_folder('result/kps/init')

    # 预训练
    '''
    t_mid = torch.load('pretrain_data/t_mid.pth')
    C_0 = torch.load('pretrain_data/C_0_pretrain.pth')
    C_T = torch.load('pretrain_data/C_T_pretrain(%s).pth' % model)
    train(torch.device('cuda:0'), t_mid.clone(), C_0.clone(), C_T.clone(),
          model, n_layers, n_neurons, 'pretrain', 
          term, 20000, lr, 20000, lr_decay_gamma, n_cols)
    '''


    for unit in unit_and_serial_numbers:
        serial_numbers = unit_and_serial_numbers[unit]
        for serial_number in serial_numbers:

            # 生成保存权重与参数的文件夹
            generate_folder('result/wt/%s' % serial_number)
            generate_folder('result/kps/%s' % serial_number)

            # 读取数据并提取有意义的部分
            t_mid = torch.load('../../data/t_mid/%s.pth' % serial_number)
            imgs = torch.load('../../data/pth/%s.pth' % serial_number)
            C_0 = torch.load('../../data/TAC/C_0/%s.pth' % serial_number)
            segment = torch.load('../../data/segment/%s.pth' % serial_number)
            n_frames, z_size, x_size, y_size = imgs.shape
            C_Ts = imgs.reshape(n_frames, z_size * x_size * y_size)
            segment = torch.nonzero(segment.reshape(z_size * x_size * y_size))[:, 0]
            C_Ts = C_Ts[:, segment]

            # 对数据集进行划分，分批次喂入网络继续计算
            n_voxels = C_Ts.shape[1]
            n_batches = n_voxels // (standard_len * gpus_to_use) + 1  # 分为多少批次处理完，最后多出来的一个批次需要进行细节的处理

            for batch in tqdm(range(n_batches), desc='%s: %s' % (unit, serial_number)):
                C_Ts_list = []
                if batch != (n_batches - 1):
                    for i in range(gpus_to_use):
                        start = batch * (standard_len * gpus_to_use) + i * standard_len
                        end = start + standard_len
                        C_Ts_list.append(C_Ts[:, start: end])
                else:
                    start_last_batch = n_voxels - n_voxels % (standard_len * gpus_to_use)
                    standard_len_last_batch = (n_voxels % (standard_len * gpus_to_use)) // gpus_to_use
                    for i in range(gpus_to_use):
                        if i != gpus_to_use - 1:
                            start = start_last_batch + i * standard_len_last_batch
                            end = start + standard_len_last_batch
                            C_Ts_list.append(C_Ts[:, start: end])
                        else:
                            C_Ts_list.append(C_Ts[:, end:])
                
                t_mid = torch.load('pretrain_data/t_mid.pth')
                C_0 = torch.load('pretrain_data/C_0_pretrain.pth')
                C_T = torch.load('pretrain_data/C_T_pretrain(%s).pth' % model)
                train(torch.device('cuda:0'), t_mid.clone(), C_0.clone(), C_T.clone(),
                        model, n_layers, n_neurons, '0.0.0/0_0', 
                        term, 20000, lr, 20000, lr_decay_gamma, n_cols)

                pool = mpc.Pool(gpus_to_use)
                for i in range(gpus_to_use):  # 这里的i是相对索引，是指在每个batch中的索引
                    pool.apply_async(func=train, args=(torch.device('cuda:%d' % i), t_mid.clone(), C_0.clone(), C_Ts_list[i],
                                                       '1.0', n_layers, n_neurons, '%s/%d_%d' % (serial_number, batch, i), 
                                                       term, n_iters, lr, lr_decay_step, lr_decay_gamma, n_cols), error_callback=print_error)
                pool.close()
                pool.join()
