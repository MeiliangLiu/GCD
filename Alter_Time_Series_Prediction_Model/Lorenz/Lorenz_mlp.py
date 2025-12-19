import numpy as np
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler
from ComputeROC import compute_roc
from src.efficient_kan import KAN
from synthetic import simulate_lorenz_96

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(MLP, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),                      # 可以替换为其它激活函数
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, x):
        return self.net(x)

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

    input_seq = torch.tensor(test_x, dtype=torch.float32).unsqueeze(0).cuda()
    target_seq = torch.tensor(test_y, dtype=torch.float32).cuda()

    model = MLP(P,hidden_size,P).to(device)
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
        F=1.0,
        T=1000,
        epoch=200,        # 训练轮数
        hidden_size=512,  # KAN隐藏层大小（未使用，但可以保留以扩展）
        learning_rate=0.01,
        lam = 0.0015,
    )