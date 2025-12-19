import numpy as np
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler
from ComputeROC import compute_roc
from src.efficient_kan import KAN
from tool import dream_read_label, dream_read_data

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def off_diagonal(x):
    mask = ~np.eye(x.shape[0], dtype=bool)
    non_diag_elements = x[mask]
    new_arr = non_diag_elements.reshape(100, 99)
    return new_arr

def read_dream3(size, type):
    name_list = ["Ecoli1", "Ecoli2", "Yeast1", "Yeast2", "Yeast3"]
    GC = dream_read_label(
        r"E:\efficient-kan\DREAM3 in silico challenge"
        r"\DREAM3 gold standards\DREAM3GoldStandard_InSilicoSize" + str(size) + "_" + name_list[type - 1] + ".txt",
        size)
    data = dream_read_data(
        r"E:\efficient-kan\DREAM3 in silico challenge"
        r"\Size" + str(size) + "\DREAM3 data\InSilicoSize" + str(size) + "-" + name_list[
            type - 1] + "-trajectories.tsv")
    return GC, data


def infer_Grangercausality(P, type, epoch, hidden_size, learning_rate, lam):

    global_seed = 1
    torch.manual_seed(global_seed)
    torch.cuda.manual_seed(global_seed)
    np.random.seed(global_seed)

    GC, data = read_dream3(P, type)
    GC = off_diagonal(GC)

    X = data.reshape(966, 100)

    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    test_x = X[:965, :]
    test_y = X[1:966, :]

    input_seq = torch.tensor(test_x, dtype=torch.float32).unsqueeze(0).cuda()
    target_seq = torch.tensor(test_y, dtype=torch.float32).cuda()

    model = KAN([P, hidden_size, P], base_activation=nn.Identity).to(device)
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

        GC_est = off_diagonal(GC_est)

        score_gi, aupr_gi = compute_roc(GC, GC_est, False)

        total_loss = predict_loss + L1_loss
        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()

        if (i + 1) % 1 == 0:
            print(f"Epoch {i+1}/{epoch} | Loss: {total_loss:.4f} | L1_loss:{L1_loss:.4f} | "
                  f" AUROC: {score_gi:.4f} | AUPRC: {aupr_gi:.4f}")

if __name__ == '__main__':

    infer_Grangercausality(
        P=100,             # 输入维度
        type=1,             # Lorenz 系统参数0
        epoch=200,        # 训练轮数
        hidden_size=128,  # KAN隐藏层大小（未使用，但可以保留以扩展）
        learning_rate=0.01,
        lam = 0.008,
    )
