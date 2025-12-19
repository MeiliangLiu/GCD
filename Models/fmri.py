import numpy as np
import torch
import torch.nn as nn
from sklearn.model_selection import ParameterGrid
from sklearn.preprocessing import StandardScaler
from ComputeROC import compute_roc
from src.efficient_kan import KAN
from tool import fmri_read

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def infer_Grangercausality(simulation, subject, epoch, hidden_size, learning_rate, lam):

    global_seed = 1
    torch.manual_seed(global_seed)
    torch.cuda.manual_seed(global_seed)
    np.random.seed(global_seed)
    best_auroc = 0
    best_aupr = 0

    X, GC, length = fmri_read(simulation, subject)
    P = X.shape[1]

    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    test_x = X[:length - 1, :]
    test_y = X[1:length, :]

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
            grad_input = grad.abs().mean(dim=0)
            loss_i = grad_input.abs().sum()
            losses.append(loss_i)
            grad_input_matrix[output_idx] = grad_input

        L1_loss = lam * sum(losses)

        GC_est = grad_input_matrix.detach().cpu().numpy()

        score_gi, aupr_gi = compute_roc(GC, GC_est, False)

        total_loss = predict_loss + L1_loss

        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()

        if (i + 1) % 1 == 0:
            print(f"Epoch {i+1}/{epoch} | Loss: {total_loss:.4f} | L1_loss:{L1_loss:.4f} | "
                  f" AUROC: {score_gi:.4f} | AUPRC: {aupr_gi:.4f}")

    return best_auroc,best_aupr


def grid_search(param_grid):
    results = []
    param_list = list(ParameterGrid(param_grid))

    for params in param_list:
        print(f"Training with params: {params}")

        infer_Grangercausality(3, 0, 600, hidden_size=params['hidden_size'], lam=params['lam'],
                                            learning_rate=params['learning_rate']
                                           )

    best_params = max(results, key=lambda x: x[1])
    print(f"Best params: {best_params[0]} with avg score: {best_params[1]}")
    return best_params




if __name__ == '__main__':

    # infer_Grangercausality(
    #     simulation=4,             # 输入维度
    #     subject=0,
    #     epoch=300,        # 训练轮数
    #     hidden_size=512,  # KAN隐藏层大小（未使用，但可以保留以扩展）
    #     learning_rate=0.0001,
    #     lam = 0.001,
    # )


    param_grid = {
        'hidden_size': [128,200,256,300,400,512],
        'lam': [0.005,0.006,0.007,0.0075,0.0076,0.0079,0.008,0.0081,0.0085,0.009],
        'learning_rate': [0.0001,0.0002,0.0003,0.0005]
    } ###  0.734,0.673,0.59,0.53,0.566

    best_params = grid_search(param_grid)
