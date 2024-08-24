import numpy as np
import matplotlib.pyplot as mp
import os
import torch
from util import generate_folder
import time


data_indices = [3]
device = torch.device('cpu')
cmap = 'binary'  # 'binary'#'Reds'#



def find_max_a_b(file_list):
    max_a = float('-inf')
    max_b = float('-inf')
    for file_name in file_list:
        if file_name.startswith("parameters(") and file_name.endswith(").pth"):
            parts = file_name.split("(")[1].split(")")[0].split("_")
            if len(parts) == 2:
                a, b = parts
                if a.isdigit() and b.isdigit():
                    max_a = max(max_a, int(a))
                    max_b = max(max_b, int(b))
    return max_a + 1, max_b + 1  # 因为索引从0开始

for data_index in data_indices:
    for delay_correct in delay_corrects:
        
        if data_index == 0:
            truncation_list = {'fv': 1, 'K1': 0.6, 'k2': 2, 'k3': 0.3, 'k4': 0, 'Ki': 0.05}
            # 左肺肿瘤，血管不清晰，x方向索引68
            z_start, z_end = 180, 196
            y_start, y_end = 100, 116
        if data_index == 1:
            truncation_list = {'fv': 1, 'K1': 0.5, 'k2': 2, 'k3': 0.3, 'k4': 0, 'Ki': 0.05}
            z_start, z_end = 156, 196
            y_start, y_end = 95, 135
        if data_index == 2:
            truncation_list = {'fv': 1, 'K1': 0.5, 'k2': 2, 'k3': 0.3, 'k4': 0, 'Ki': 0.05}
            z_start, z_end = 180, 196
            y_start, y_end = 100, 116
        if data_index == 3:
            truncation_list = {'fv': 1, 'K1': 0.5, 'k2': 2, 'k3': 0.3, 'k4': 0, 'Ki': 0.2}
            z_start, z_end = 50, 110
            y_start, y_end = 60, 125
        if data_index == 4:
            truncation_list = {'fv': 0.13, 'K1': 0.5, 'k2': 2, 'k3': 0.5, 'k4': 0, 'Ki': 0.1}
            z_start, z_end = 50, 110
            y_start, y_end = 60, 125
        if data_index == 5:
            truncation_list = {'fv': 1, 'K1': 0.5, 'k2': 2, 'k3': 0.5, 'k4': 0, 'Ki': 0.1}
            z_start, z_end = 20, 90
            y_start, y_end = 70, 130

        segment = torch.load('data/process_data/segment/segment(%d).pth' % data_index, map_location=device)
        z_size, x_size, y_size = segment.shape
        n_voxels = int(torch.sum(segment))
        segment = torch.nonzero(segment.reshape(z_size * x_size * y_size))[:, 0]

        # 数据整合
        n_batches, n_gpus = 5, 2#find_max_a_b(os.listdir('result/%d/%s' % (data_index, str(delay_correct))))
        pi_img_compress = torch.zeros(size=[5, n_voxels], device=device)
        f = 0
        for batch in range(n_batches):
            for gpu in range(n_gpus):
                result = torch.load('result/%d/%s/parameters(%d_%d).pth' % (data_index, str(delay_correct), batch, gpu), map_location=device)
                pi_img_compress[:, f: f + result.shape[1]] = result
                f += result.shape[1]

        pi_img = torch.zeros(size=[5, z_size * x_size * y_size], device=device)
        pi_img[:, segment] = pi_img_compress
        pi_img = pi_img.reshape(5, z_size, x_size, y_size)
        Ki_img = pi_img[1: 2, :, :, :] * pi_img[3: 4, :, :, :] / (pi_img[1: 2, :, :, :] + pi_img[2: 3, :, :, :])
        Ki_img = torch.nan_to_num(Ki_img, 0)
        pi_img = torch.cat([pi_img, Ki_img], dim=0)
        pi_img = pi_img.cpu().detach().numpy()

        for parameter_index, parameter in enumerate(truncation_list):
            
            if parameter_index == 0:
                vmin, vmax = 0, 1
            elif parameter_index == 1:
                vmin, vmax = 0, 0.5
            elif parameter_index == 2:
                vmin, vmax = 0, 2
            elif parameter_index == 3:
                vmin, vmax = 0, 0.6
            elif parameter_index == 4:
                vmin, vmax = 0, 0.1
            else:
                vmin, vmax = 0, 0.1
                
        
            print(time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time())), data_index, delay_correct, parameter)
            
            generate_folder('fig/%d/pi/%s/MIP' % (data_index, str(delay_correct)))
            generate_folder('fig/%d/pi/%s/slice/intact/%s' % (data_index, str(delay_correct), parameter))
            generate_folder('fig/%d/pi/%s/slice/capture/%s' % (data_index, str(delay_correct), parameter))
            
            _ = pi_img[parameter_index, :, :, :].copy()
            _[_ >= truncation_list[parameter]] = truncation_list[parameter]
    
            # MIP
            # coronal
            mp.figure(dpi=500)
            mp.axis('off')
            mp.xticks([])
            mp.yticks([])
            #mp.title(parameter)
            mp.imshow(np.max(_, axis=1), cmap=cmap)
            mp.colorbar()
            mp.savefig('fig/%d/pi/%s/MIP/coronal(%s).png' % (data_index, str(delay_correct), parameter), bbox_inches='tight')
            mp.cla()
    
            # sagittal
            mp.figure(dpi=500)
            mp.axis('off')
            mp.xticks([])
            mp.yticks([])
            #mp.title(parameter)
            mp.imshow(np.max(_, axis=2), cmap=cmap)
            mp.colorbar()
            mp.savefig('fig/%d/pi/%s/MIP/sagittal(%s).png' % (data_index, str(delay_correct), parameter), bbox_inches='tight')
            mp.cla()
    
    
            # slice(只保存coronal)
            for x_index in [80, 85, 90, 95, 100, 105, 110]:
                # coronal(intact)
                mp.figure(dpi=500)
                mp.axis('off')
                mp.xticks([])
                mp.yticks([])
                #mp.title(parameter)
                mp.imshow(_[:, x_index, :], cmap=cmap, vmin=vmin, vmax=vmax)
                # mp.colorbar(fraction=0.02, orientation='horizontal', pad=0)
                mp.colorbar()
                mp.savefig('fig/%d/pi/%s/slice/intact/%s/%d.png' % (data_index, str(delay_correct), parameter, x_index), bbox_inches='tight')
                mp.cla()
            
                # coronal(capture)
                mp.figure(dpi=500)
                mp.axis('off')
                mp.xticks([])
                mp.yticks([])
                #mp.title(parameter)
                mp.imshow(_[z_start: z_end, x_index, y_start: y_end], cmap=cmap, vmin=vmin, vmax=vmax)  # 0 左侧肺部肿瘤
                # mp.colorbar(fraction=0.02, orientation='horizontal', pad=0)
                mp.colorbar() 
                mp.savefig('fig/%d/pi/%s/slice/capture/%s/%d.png' % (data_index, str(delay_correct), parameter, x_index), bbox_inches='tight')
                mp.cla()


