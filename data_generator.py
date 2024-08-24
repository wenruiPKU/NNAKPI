import torch
import matplotlib.pyplot as mp
from util import generate_folder

# 目前只支持2TCM，即model_1
# rf: response function
# At this stage, we do not reshape the data; all data is represented as vecs.
# objective: C_0, C_1, C_2, C_T.
# Default to using the GPU.
# The default scan duration is 1 hour.

def exception_handling(x):
    x[x < 0.0] = 0.0
    x[torch.isnan(x)] = 0


def f_C_0(t, IFps):
    A_1, A_2, A_3, lambda_1, lambda_2, lambda_3, alpha = IFps
    C_0 = (A_1 * t ** alpha - A_2 - A_3) * torch.exp(-lambda_1 * t) + \
           A_2 * torch.exp(-lambda_2 * t) + \
           A_3 * torch.exp(-lambda_3 * t)
    return C_0


def t_data_generator(delta_frames):
    # unit of delta_t: s
    frame_split = torch.cumsum(torch.cat([torch.zeros(size=(1,)), delta_frames], dim=0), dim=0)
    t_mid = torch.zeros(size=(frame_split.shape[0] - 1, ))
    for i in range(t_mid.shape[0]):
        t_mid[i] = (frame_split[i] + frame_split[i + 1]) / 2
    return frame_split / 60, t_mid / 60  # unit conversion to min


def generate_t_col_mat(end, t_step):
    # t_step must be divisible by 1
    # unit of t_step: s
    # unit of end: min
    n_cols = int(end / (t_step / 60)) + 1
    t_col = torch.linspace(start=0, end=end, steps=n_cols)
    t_col_mat = torch.zeros(size=[n_cols, n_cols])
    t_col_mat_r = torch.zeros(size=[n_cols, n_cols])
    for i in range(n_cols):
        t_col_mat[i, 0: i + 1] = t_col[0: i + 1]
        t_col_mat_r[i, 0: i + 1] = torch.flip(t_col[0: i + 1], dims=[0])
    return t_col, t_col_mat, t_col_mat_r  # unit: min


def TAC_generator(IFps, kps, t_step, t_col, t_col_mat, t_col_mat_r):

    fv, K1, k2, k3, k4 = kps

    k2_k3_k4 = k2 + k3 + k4
    k2_k4 = k2 * k4
    a1 = 0.5 * (k2_k3_k4 - (k2_k3_k4 ** 2 - 4 * k2_k4) ** 0.5)
    a2 = 0.5 * (k2_k3_k4 + (k2_k3_k4 ** 2 - 4 * k2_k4) ** 0.5)

    C_0_mat = f_C_0(t_col_mat, IFps) * (1 - 0.5 * torch.eye(t_col_mat.shape[0], device=t_col_mat.device))
    rf_1_r = torch.exp(-a1 * t_col_mat_r)
    rf_2_r = torch.exp(-a2 * t_col_mat_r)

    # covolution
    _ = C_0_mat * rf_1_r
    _1 = (torch.sum(_, dim=1) * t_step)
    _ = C_0_mat * rf_2_r
    _2 = (torch.sum(_, dim=1) * t_step)
    
    C_0_col = f_C_0(t_col, IFps)
    C_1_col = K1 / (a2 - a1) * ((k4 - a1) * _1 + (a2 - k4) * _2)
    C_2_col = K1 * k3 / (a2 - a1) * (_1 - _2)
    C_T_col = fv * C_0_col + (1 - fv) * (C_1_col + C_2_col)

    return C_0_col, C_T_col, C_1_col, C_2_col


def from_col_to_mid_vec(TAC_col_vec, t_step, frame_split):
    # unit of t_step: s
    # unit of frame_split: min
    n_frames = frame_split.shape[0] - 1
    TAC_mid_vec = torch.zeros(size=[n_frames], device=TAC_col_vec.device)
    for i in range(n_frames - 1):
        index_start, index_end = int(frame_split[i] * 60 / t_step), int(frame_split[i + 1] * 60 / t_step)
        TAC_mid_vec[i] = torch.mean(TAC_col_vec[index_start: index_end])
    TAC_mid_vec[-1] = torch.mean(TAC_col_vec[int(frame_split[-2] * 60 / t_step):])
    exception_handling(TAC_mid_vec)
    return TAC_mid_vec


if __name__ == '__main__':

    generate_folder('pretrain_data')
    
    # generation parameters
    d_0 = 0  # unit: s
    t_step = 1  # unit: s  # 相邻两次采样的时间间隔
    end = 60  # unit: min
    n_cols = int(end / (t_step / 60)) + 1
    device = torch.device('cuda:0')
    model = '2.0'
    delta_frames = torch.cat([torch.ones(size=(30,)) * 2,  # 1 min
                              torch.ones(size=(12,)) * 10,  # 3 min
                              torch.ones(size=(6,)) * 30,  # 6 min
                              torch.ones(size=(12,)) * 120,  # 30 min
                              torch.ones(size=(6,)) * 300], dim=0)
    frame_split, t_mid = t_data_generator(delta_frames)
    torch.save(t_mid, 'pretrain_data/t_mid')
    n_mids = t_mid.shape[0]

    # preparation for time input
    t_col, t_col_mat, t_col_mat_r = generate_t_col_mat(end, t_step)
    t_col = t_col.to(device)
    t_col_mat = t_col_mat.to(device)
    t_col_mat_r = t_col_mat_r.to(device)
    
    # IFps reference: feng + x
    # kps reference: Efficient Delay Correction for Total-Body PET Kinetic
    C_0_pretrain, C_T_pretrain, C_1_pretrain, C_2_pretrain = TAC_generator(IFps=[851.12, 20.81, 21.88, 4.13, 0.01, 0.12, 1.0],
                                                                           kps=[0.076, 0.529, 1.171, 0.037, 0.0],  # all region
                                                                           t_step=t_step / 60, t_col=t_col,
                                                                           t_col_mat=t_col_mat, t_col_mat_r=t_col_mat_r)
    
    C_0_pretrain = from_col_to_mid_vec(C_0_pretrain, t_step, frame_split)
    C_T_pretrain = from_col_to_mid_vec(C_T_pretrain, t_step, frame_split)

    # 展示
    mp.figure(dpi=100)
    mp.plot(t_mid, C_0_pretrain.cpu(), label='C_0')
    mp.plot(t_mid, C_T_pretrain.cpu(), label='C_T')
    mp.legend()
    mp.savefig('pretrain_data/pretrain_data.png')
    
    # save
    torch.save(t_mid, 'pretrain_data/t_mid.pth')
    torch.save(C_0_pretrain, 'pretrain_data/C_0_pretrain.pth')
    torch.save(C_T_pretrain.reshape(-1, 1), 'pretrain_data/C_T_pretrain(%s).pth' % model)