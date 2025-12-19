import numpy as np
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler
from ComputeROC import compute_roc
from src.efficient_kan import KAN
from synthetic import simulate_lorenz_96

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# === TCN 定义部分 ===
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

# === 主训练流程 ===
def infer_Grangercausality(P, F, T, epoch, hidden_size, learning_rate, lam):
    global_seed = 1
    torch.manual_seed(global_seed)
    torch.cuda.manual_seed(global_seed)
    np.random.seed(global_seed)

    # Lorenz-96 dataset
    X, GC = simulate_lorenz_96(P, T, F, delta_t=0.1, sd=0.1, burn_in=1000, seed=0)
    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    length = X.shape[0]
    test_x = X[:length - 1]
    test_y = X[1:length]

    input_seq = torch.tensor(test_x, dtype=torch.float32).unsqueeze(0).to(device)  # (1, T-1, P)
    target_seq = torch.tensor(test_y, dtype=torch.float32).to(device)             # (T-1, P)

    model = TCNModel(input_dim=P, hidden_dim=hidden_size, output_dim=P).to(device)
    loss_fn = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    for i in range(epoch):
        losses = []
        input_current = input_seq.clone().detach().requires_grad_()  # (1, T-1, P)
        output_current = model(input_current).squeeze(0)             # (T-1, P)
        predict_loss = loss_fn(output_current, target_seq)

        grad_input_matrix = torch.zeros((P, P)).to(device)
        for output_idx in range(P):
            model.zero_grad()
            out = output_current[:, output_idx].sum()
            grad = torch.autograd.grad(out, input_current, retain_graph=True, create_graph=True)[0]  # (1, T-1, P)
            grad_input = grad.abs().mean(dim=1).squeeze(0)  # shape: (P,)
            loss_i = grad_input.abs().sum()
            losses.append(loss_i)
            grad_input_matrix[output_idx] = grad_input

        GI_loss = lam * sum(losses)

        GI_np = grad_input_matrix.detach().cpu().numpy()
        score_gi, aupr_gi = compute_roc(GC, GI_np, False)

        total_loss = predict_loss + GI_loss
        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()

        if (i + 1) % 1 == 0:
            print(f"Epoch {i+1}/{epoch} | Loss: {total_loss:.4f} | GI_loss:{GI_loss:.4f} | "
                  f"[GI-GC] ROC-AUC: {score_gi:.4f} | AUPR: {aupr_gi:.4f}")


if __name__ == '__main__':
    infer_Grangercausality(
        P=100,             # 输入维度
        F=10,
        T=1000,
        epoch=200,        # 训练轮数
        hidden_size=512,
        learning_rate=0.01,
        lam=0.0009,
    )
