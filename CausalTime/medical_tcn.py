import numpy as np
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler
from ComputeROC import compute_roc
from load_data import read_causaltime

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
        x = x.transpose(1, 2)
        out = self.network(x)
        out = out.transpose(1, 2)
        return self.linear(out)

def infer_Grangercausality(type, epoch, hidden_size, learning_rate, lam):

    global_seed = 1
    torch.manual_seed(global_seed)
    torch.cuda.manual_seed(global_seed)
    np.random.seed(global_seed)

    X, GC = read_causaltime(type)
    P = X.shape[1]
    P_true = int(P/2)
    length = X.shape[0]

    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    test_x = X[:length - 1, :]
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

        grad_input_matrix = torch.zeros((P, P)).to(device)
        for output_idx in range(P):
            model.zero_grad()
            out = output_current[:, output_idx].sum()
            grad = torch.autograd.grad(out, input_current, retain_graph=True, create_graph=True)[0]
            grad_input = (grad).abs().mean(dim=0)
            loss_i = grad_input.abs().sum()
            losses.append(loss_i)
            grad_input_matrix[output_idx] = grad_input

        L1_loss = lam * sum(losses)

        GC_est = grad_input_matrix.detach().cpu().numpy()
        GC_est = GC_est[:P_true, :P_true]

        score_gi, aupr_gi = compute_roc(GC, GC_est, False)

        total_loss = predict_loss + L1_loss

        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()

        if (i + 1) % 1 == 0:
            print(f"Epoch {i + 1}/{epoch} | Loss: {total_loss:.4f} | L1_loss:{L1_loss:.4f} | "
                  f"AUROC: {score_gi:.4f} | AUPRC: {aupr_gi:.4f}")

if __name__ == '__main__':

    infer_Grangercausality(
        type='medical',
        epoch=500,
        hidden_size=512,
        learning_rate=0.001,
        lam = 0.01,
    )

    # infer_Grangercausality(
    #     type='pm25',
    #     epoch=500,
    #     hidden_size=128,
    #     learning_rate=0.01,
    #     lam = 0.0021,
    # )
