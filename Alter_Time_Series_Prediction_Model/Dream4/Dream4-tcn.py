import numpy as np
import torch
import torch.nn as nn
from sklearn.model_selection import ParameterGrid
from sklearn.preprocessing import StandardScaler
from ComputeROC import compute_roc
from src.efficient_kan import KAN
from tool import dream_read_label, dream_read_data
import scipy.io as sio

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class TCNBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, dilation=1):
        super(TCNBlock, self).__init__()
        padding = (kernel_size - 1) * dilation
        self.net = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size,
                      padding=padding, dilation=dilation),
            nn.ReLU(),
            nn.Conv1d(out_channels, out_channels, kernel_size,
                      padding=padding, dilation=dilation),
            nn.ReLU()
        )
        self.downsample = nn.Conv1d(in_channels, out_channels, 1) if in_channels != out_channels else None

    def forward(self, x):
        out = self.net(x)
        if self.downsample:
            x = self.downsample(x)
        return out[:, :, :x.size(2)] + x

class TCNModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, levels=3, kernel_size=3):
        super(TCNModel, self).__init__()
        layers = []
        for i in range(levels):
            dilation_size = 2 ** i
            in_ch = input_dim if i == 0 else hidden_dim
            layers.append(TCNBlock(in_ch, hidden_dim, kernel_size, dilation=dilation_size))
        self.network = nn.Sequential(*layers)
        self.linear = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        # x shape: (batch, seq_len, features)
        x = x.transpose(1, 2)  # (batch, features, seq_len)
        out = self.network(x)
        out = out.transpose(1, 2)  # (batch, seq_len, hidden_dim)
        return self.linear(out)


def off_diagonal(x):
    mask = ~np.eye(x.shape[0], dtype=bool)
    non_diag_elements = x[mask]
    new_arr = non_diag_elements.reshape(100, 99)
    return new_arr

def read_dream4(size, type):
    GC = dream_read_label(
        r"E:\IntegratedGradients\DREAM4 in-silico challenge"
        r"\DREAM4 gold standards\insilico_size" + str(size) + "_" + str(type) + "_goldstandard.tsv",
        size)
    data = sio.loadmat(r'E:\IntegratedGradients\DREAM4 in-silico challenge'
                       r"\DREAM4 training data\insilico_size" + str(size) + "_" + str(type) + '_timeseries.mat')
    data = data['data']
    return GC, data

def infer_Grangercausality(P, type, epoch, hidden_size, learning_rate, lam):

    global_seed = 1
    torch.manual_seed(global_seed)
    torch.cuda.manual_seed(global_seed)
    np.random.seed(global_seed)

    best_score = 0

    GC, X = read_dream4(P, type)
    GC = off_diagonal(GC)

    # X = data.reshape(210, 100)
    length = X.shape[0]

    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    test_x = X[:length-1, :]
    test_y = X[1:length, :]

    input_seq = torch.tensor(test_x, dtype=torch.float32).unsqueeze(0).cuda()
    target_seq = torch.tensor(test_y, dtype=torch.float32).cuda()

    model = TCNModel(P, hidden_size, P).to(device)
    loss_fn = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    for i in range(epoch):
        losses = []
        input_current = input_seq.squeeze(0).detach().clone().requires_grad_().to(device)
        output_current = model(input_current.unsqueeze(0)).squeeze(0)
        predict_loss = loss_fn(output_current, target_seq)

        # === Gradient × Input 可导稀疏归因 ===
        grad_input_matrix = torch.zeros((P, P)).to(device)
        for output_idx in range(P):
            model.zero_grad()
            out = output_current[:, output_idx].sum()
            grad = torch.autograd.grad(out, input_current, retain_graph=True, create_graph=True)[0]
            grad_input = (grad).abs().mean(dim=0)
            loss_i = grad_input.abs().sum()
            losses.append(loss_i)
            grad_input_matrix[output_idx] = grad_input

        GI_loss = lam * sum(losses)
        # GI_loss = lam * torch.norm(grad_input_matrix, dim=1).sum()

        # ROC/AUPR计算（评估指标）
        GI_np = grad_input_matrix.detach().cpu().numpy()

        GI_np = off_diagonal(GI_np)

        score_gi, aupr_gi = compute_roc(GC, GI_np, False)

        # === 总损失 ===
        total_loss = predict_loss + GI_loss
        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()


        # if best_score < score_gi and score_gi > 0.65:
        #     best_score = score_gi
        #     # np.savetxt("simulation=" + simulation + "subject=" + str(F) + ",true.txt", GC, fmt='%.5f')
        #     np.savetxt(
        #         f"type={type}, "
        #         f"score={score_gi},learning_rate = {learning_rate}, "
        #         f"lam = {lam},epoch={i}.txt",
        #         GI_np, fmt=f'%.5f')


        if (i + 1) % 1 == 0:
            print(f"Epoch {i+1}/{epoch} | Loss: {total_loss:.4f} | GI_loss:{GI_loss:.4f} | "
                  f"[GI-GC] ROC-AUC: {score_gi:.4f} | AUPR: {aupr_gi:.4f}")



def grid_search(param_grid):
    results = []
    param_list = list(ParameterGrid(param_grid))

    for params in param_list:
        print(f"Training with params: {params}")

        infer_Grangercausality(100, 2, 200, hidden_size=params['hidden_size'], lam=params['lam'],
                                            learning_rate=params['learning_rate']
                                           )

    best_params = max(results, key=lambda x: x[1])
    print(f"Best params: {best_params[0]} with avg score: {best_params[1]}")
    return best_params



if __name__ == '__main__':

    infer_Grangercausality(
        P=100,             # 输入维度
        type=2,             # Lorenz 系统参数0
        epoch=200,        # 训练轮数
        hidden_size=128,  # KAN隐藏层大小（未使用，但可以保留以扩展）
        learning_rate=0.01,
        lam = 0.008,
    )
    #0.701 0.618 0.630 0.583 0.590


    # param_grid = {
    #     'hidden_size': [64,128,200,256,300],
    #     'lam': [0.005,0.006, 0.007,0.008,0.009,0.01,0.02,0.03,0.04,0.05],
    #     'learning_rate': [0.01]
    # } ###  0.734,0.673,0.59,0.53,0.566
    #
    # best_params = grid_search(param_grid)