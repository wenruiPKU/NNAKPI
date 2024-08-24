import torch
from util import init_net
from model import model_1


# 输出端每个分支的对应索引
def select_indices_2(n_TACs):
    S_T_indices, S_1_indices, S_2_indices = [], [], []
    for i in range(n_TACs):
        S_T_indices.append(3 * i)
        S_1_indices.append(3 * i + 1)
        S_2_indices.append(3 * i + 2)
    return S_T_indices, S_1_indices, S_2_indices


def delay_locate(t_mid, TAC, d_0_index, term, e=0.00001, n_windows=1):
    TAC[0: d_0_index] = 0  # 根据C_0结果抹除早期噪声的误导，C_0噪声相对较小
    integration = torch.zeros_like(t_mid)
    integration[0] = 0.5 * t_mid[0] * TAC[0] + e
    for i in range(1, t_mid.shape[0]):
        integration[i] = integration[i - 1] + 0.5 * (t_mid[i] - t_mid[i - 1]) * (TAC[i] + TAC[i - 1])
    ratio = ((integration[n_windows:] - integration[: -n_windows]) / (t_mid[n_windows:] - t_mid[: -n_windows])) / \
            (integration[n_windows:] / t_mid[n_windows:])
    d_T_index = torch.argmax(ratio[d_0_index: term], dim=0) + d_0_index
    d_T_value = t_mid[d_T_index - 1]  # 粗略地认为上一帧才是示踪剂出现的时间点，噪声太显著，取消牛顿法迭代微调
    return d_T_index, d_T_value


# 每一组TAC应当对应的时间进行位移
def delay_shift(t_mid, C_0, C_Ts, term):
    n_TACs = C_Ts.shape[1]
    shifted_t_C_Ts = torch.zeros(size=[t_mid.shape[0], n_TACs])
    d_0_index, d_0_value = delay_locate(t_mid, C_0, 0, term)

    shifted_t_C_0 = t_mid - d_0_value
    for i in range(n_TACs):
        d_T_index, d_T_value = delay_locate(t_mid, C_Ts[:, i], d_0_index, term)
        shifted_t_C_Ts[:, i] = t_mid - d_T_value
    return shifted_t_C_0, shifted_t_C_Ts


# 数值微分系数矩阵的生成
def coefficient_matrix_generator(start, end, n_cols):
    # Finite difference method: coefficient matrix
    W = torch.zeros(size=[n_cols, n_cols])
    grid_space = (end - start) / (n_cols - 1)
    W[0, 0], W[0, 1], W[0, 2] = -3 / (2 * grid_space), 4 / (2 * grid_space), -1 / (2 * grid_space)  # second-order
    W[-1, -3], W[-1, -2], W[-1, -1] = 1 / (2 * grid_space), -4 / (2 * grid_space), 3 / (2 * grid_space)  # second-order
    W[1, 0], W[1, 1], W[1, 2], W[1, 3] = -2 / (6 * grid_space),  -3 / (6 * grid_space), 6 / (6 * grid_space), -1 / (6 * grid_space)  # third-order
    W[-2, -4], W[-2, -3], W[-2, -2], W[-2, -1] = 1 / (6 * grid_space), -6 / (6 * grid_space), 3 / (6 * grid_space), 2 / (6 * grid_space)  # third-order
    for i in range(2, n_cols - 2):  # forth-order
        W[i, i - 2] = 1 / (12 * grid_space)
        W[i, i - 1] = -8 / (12 * grid_space)
        W[i, i + 1] = 8 / (12 * grid_space)
        W[i, i + 2] = -1 / (12 * grid_space)
    return W


def norm(C_0, C_Ts):
    scale = torch.max(C_0)
    C_0 /= scale
    C_Ts /= scale
    return scale, C_0, C_Ts


def bound(x, upper_bound, low_bound):
    x = torch.minimum(x, upper_bound * torch.ones_like(x, device=x.device))
    x = torch.max(x, low_bound * torch.ones_like(x, device=x.device))
    return x


def train(device,
          t_mid, C_0, C_Ts,
          model, n_layers, n_neurons, type,
          term, n_iters, lr, lr_decay_step, lr_decay_gamma, n_cols):  
    
    print(type)

    n_TACs = C_Ts.shape[1]

    # 用作不同分支的输入
    t_col_tr = torch.linspace(start=0, end=t_mid[-1], steps=n_cols, device=device).reshape(-1, 1, 1)
    t_col_br = torch.linspace(start=0, end=t_mid[-1], steps=n_cols, device=device).reshape(-1, 1, 1).repeat(1, n_TACs, 1)

    # 数据归一化
    scale, C_0, C_Ts = norm(C_0, C_Ts)
    t_mid = t_mid.to(device)
    C_0 = C_0.reshape(-1, 1).to(device)
    C_Ts = C_Ts.to(device)
    
    # 延迟位移与整型
    if type == 'pretrain':
        t_mid_C_0 = t_mid.reshape(-1, 1, 1).to(device)
        t_mid_C_Ts = t_mid.reshape(-1, n_TACs, 1).to(device)
    else:
        t_mid_C_0, t_mid_C_Ts = delay_shift(t_mid, C_0, C_Ts, term)
        t_mid_C_0 = t_mid_C_0.reshape(-1, 1, 1).to(device)
        t_mid_C_Ts = t_mid_C_Ts.reshape(-1, n_TACs, 1).to(device)

    # 设置数值微分系数矩阵
    W = coefficient_matrix_generator(start=0, end=t_mid[-1], n_cols=n_cols)
    W = W.to(device)

    # 载入预训练模型
    if model == '1.0':
        S_T_indices, S_1_indices, S_2_indices = select_indices_2(n_TACs)
        
        S = model_1(n_layers, n_neurons, n_TACs).to(device)
        if type == 'pretrain':
            fv = 0.1 * torch.ones(size=[1, n_TACs], device=device)
            K1 = 0.1 * torch.ones(size=[1, n_TACs], device=device)
            k2 = 0.1 * torch.ones(size=[1, n_TACs], device=device)
            k3 = 0.1 * torch.ones(size=[1, n_TACs], device=device)
            k4 = torch.zeros(size=[1, n_TACs], device=device)
        else:
            wt_init, kps_init = init_net(model, n_layers, n_neurons, n_TACs)
            S.load_state_dict({k.replace('module.', ''): v for k, v in wt_init.items()})
            fv = torch.FloatTensor(kps_init[0: 1, :]).to(device)
            K1 = torch.FloatTensor(kps_init[1: 2, :]).to(device)
            k2 = torch.FloatTensor(kps_init[2: 3, :]).to(device)
            k3 = torch.FloatTensor(kps_init[3: 4, :]).to(device)
            k4 = torch.FloatTensor(kps_init[4: 5, :]).to(device)
            
        fv.requires_grad = True
        K1.requires_grad = True
        k2.requires_grad = True
        k3.requires_grad = True
        k4.requires_grad = False
        kps = [fv, K1, k2, k3, k4]
        
    # 优化器设置
    loss_fun = torch.nn.MSELoss(reduction='mean')
    optimizer = torch.optim.Adam([{'params': S.parameters(), 'lr': lr},
                                  {'params': kps, 'lr': lr}])
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer=optimizer, 
                                                step_size=lr_decay_step, 
                                                gamma=lr_decay_gamma, last_epoch=-1)

    # 初始输入
    tr_init = torch.zeros(size=[1, 1, 1], device=device)
    br_init = torch.zeros(size=[1, n_TACs, 1], device=device)

    for iter in range(1, n_iters + 1):
        # init
        S_tk_init = torch.squeeze(S(tr_init, 'trunk'), dim=2)
        S_br_init = torch.squeeze(S(br_init, 'branch'), dim=2)
        # midpoint
        S_tk_mid = torch.squeeze(S(t_mid_C_0, 'trunk'), dim=2)
        S_br_T_mid = torch.squeeze(S(t_mid_C_Ts, 'branch'), dim=2)[:, S_T_indices]
        # col
        S_tk_col = torch.squeeze(S(t_col_tr, 1), dim=2)
        S_br_col = torch.squeeze(S(t_col_br, 2), dim=2)
        # d and loss
        if model == '2.0':
            loss_init = torch.mean(S_tk_init) + torch.mean(S_br_init)
            loss_mid = loss_fun(S_tk_mid, C_0) + loss_fun(S_br_T_mid, C_Ts)
            loss_col = loss_fun(torch.mm(W, S_br_col[:, S_1_indices]), 
                                torch.abs(K1) * S_tk_col - (torch.abs(k2) + torch.abs(k3)) * S_br_col[:, S_1_indices]) + \
                       loss_fun(torch.mm(W, S_br_col[:, S_2_indices]), 
                                torch.abs(k3) * S_br_col[:, S_1_indices]) + \
                       loss_fun(S_br_col[:, S_T_indices], 
                                torch.abs(fv) * S_tk_col + (1 - torch.abs(fv)) * (S_br_col[:, S_1_indices] + S_br_col[:, S_2_indices]))
            loss = loss_init + loss_mid + loss_col

        # optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()

        if iter % 100 == 0:
            if model == '2.0':
                if type == 'pretrain':
                    print('pretraining(iter-%d): fv=%.4f, K1=%.4f, k2=%.4f, k3=%.4f' 
                          % (iter, torch.abs(fv)[0, 0].item(), torch.abs(K1)[0, 0].item(), torch.abs(k2)[0, 0].item(), torch.abs(k3)[0, 0].item()))
                    torch.save(S.state_dict(), 
                               'result/wt/init/molde_%s_n_layers_%d_n_neurons_%d(init).pth' % (model, n_layers, n_neurons))
                    torch.save(torch.cat([torch.abs(fv), torch.abs(K1), torch.abs(k2), torch.abs(k3), torch.abs(k4)], dim=0).detach(), 
                               'result/kps/init/molde_%s_n_layers_%d_n_neurons_%d(init).pth' % (model, n_layers, n_neurons))
                else:
                    torch.save(S.state_dict(), 
                               'result/wt/%smolde_%s_n_layers_%d_n_neurons_%d_iter_%d(init).pth' % (type, model, n_layers, n_neurons, iter))
                    torch.save(torch.cat([torch.abs(fv), torch.abs(K1), torch.abs(k2), torch.abs(k3), torch.abs(k4)], dim=0).detach(), 
                               'result/kps/%smolde_%s_n_layers_%d_n_neurons_%d_iter_%d(init).pth' % (type, model, n_layers, n_neurons, iter))