import torch

# 下标数字是指有多少种状态

# class model_1(torch.nn.Module):


class model_2(torch.nn.Module):
    def __init__(self, n_layers, n_neurons, n_TACs):
        super(model_2, self).__init__()

        trunk = []
        branch = []
        for i in range(n_layers):
            if i == 0:
                trunk.append(torch.nn.Conv1d(in_channels=1, out_channels=n_neurons, kernel_size=1, stride=1, groups=1, bias=True))
                branch.append(torch.nn.Conv1d(in_channels=n_TACs, out_channels=n_neurons * n_TACs, kernel_size=1, stride=1, groups=n_TACs, bias=True))
            elif i == n_layers - 1:
                trunk.append(torch.nn.Conv1d(in_channels=n_neurons, out_channels=1, kernel_size=1, stride=1, groups=1, bias=True))
                branch.append(torch.nn.Conv1d(in_channels=n_neurons * n_TACs, out_channels=3 * n_TACs, kernel_size=1, stride=1, groups=n_TACs, bias=True))
            else:
                trunk.append(torch.nn.Conv1d(in_channels=n_neurons, out_channels=n_neurons, kernel_size=1, stride=1, groups=1, bias=True))
                branch.append(torch.nn.Conv1d(in_channels=n_neurons * n_TACs, out_channels=n_neurons * n_TACs, kernel_size=1, stride=1, groups=n_TACs, bias=True))

        self.act = torch.nn.Sigmoid()
        self.n_layers = n_layers
        self.n_neurons = n_neurons
        self.trunk = torch.nn.ModuleList(trunk)
        self.branch = torch.nn.ModuleList(branch)

    def forward(self, x, _):
        if _ == 'trunk':  # 对应C_0
            for i in range(self.n_layers - 1):
                x = self.act(self.trunk[i](x))
            x = self.trunk[-1](x)
        if _ == 'branch':  # 对应C_T, C_1, C_2 
            for i in range(self.n_layers - 1):
                x = self.act(self.branch[i](x))
            x = self.branch[-1](x)
        return torch.abs(x)
