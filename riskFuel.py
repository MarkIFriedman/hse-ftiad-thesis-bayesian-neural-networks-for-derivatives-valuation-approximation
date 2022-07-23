import torch
import numpy as np
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter

if torch.cuda.is_available():
    device = "cuda:0"
    print(f"GPU detected. Running on {device}")
else:
    device = "cpu"
    print("No GPU detected. Running on CPU")

to_tensor = lambda x: torch.from_numpy(x).float().to(device)

class RiskFuelNet(torch.nn.Module):
    def __init__(self, n_feature, n_hidden, n_layers, n_output):
        super(RiskFuelNet, self).__init__()

        self.n_hidden = n_hidden
        self.n_layers = n_layers

        self.linears = torch.nn.ModuleList([torch.nn.Linear(n_feature, n_hidden)])
        self.linears.extend([torch.nn.Linear(n_hidden, n_hidden) for i in range(1, n_layers)])
        self.linears.append(torch.nn.Linear(n_hidden, n_output))

    def forward(self, x):
        for lin in self.linears:
            x = F.relu(lin(x))  # Activation function for all layers (prices can't be negative)
        return x


def fit_net(net: RiskFuelNet, n_epochs: int, x_train: np.ndarray, y_train: np.ndarray,
            x_test: np.ndarray, y_test: np.ndarray, device=device, log_folder: str = 'riskFuel'):
    net.to(device)

    x_ = to_tensor(x_train)
    y_ = to_tensor(y_train)

    x_test_ = to_tensor(x_test)
    y_test_ = to_tensor(y_test)

    writer = SummaryWriter(log_dir='./logs/' + log_folder)

    optimizer = torch.optim.Adam(net.parameters(), lr=0.01)
    loss_func = torch.nn.MSELoss()

    l = 10 ** 5
    best_l = l
    checkpoint = {}
    l_train = []
    l_test = []

    for e in range(n_epochs):
        prediction = net(x_)
        loss = loss_func(prediction, y_)
        l_train.append(loss.data.cpu().numpy())

        prediction_test = net(x_test_)
        loss_test = loss_func(prediction_test, y_test_)
        writer.add_scalar('train_loss', loss, global_step=e)
        writer.add_scalar('test_loss', loss_test, global_step=e)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        l = loss_test.data.cpu().numpy()
        l_test.append(l)
        if l.item() < best_l:
            best_l = l.item()
            checkpoint = {
                "n_hidden": net.n_hidden,
                "n_layers": net.n_layers,
                "model_state_dict": net.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
            }
        if (e + 1) % 100 == 0:
            print(f"\tEpoch: {e + 1}\tL2 Loss = {loss.data.cpu().numpy()}")

    return best_l, checkpoint, l_train, l_test
