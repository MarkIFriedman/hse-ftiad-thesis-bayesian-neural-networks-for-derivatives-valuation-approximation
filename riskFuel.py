import numpy
import torch
import torch.nn.functional as F


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


def fit_net(net: RiskFuelNet, n_epochs: int, x_train: numpy.ndarray, y_train: numpy.ndarray,
            x_test: numpy.ndarray, y_test: numpy.ndarray, device: str = 'cpu'):
    # n = y.size()[0]

    n_train = x_train.shape[0]
    n_test = x_test.shape[0]

    net.to(device)
    x_ = torch.from_numpy(x_train).to(device)
    y_ = torch.from_numpy(y_train).to(device)

    x_test_ = torch.from_numpy(x_test).to(device)
    y_test_ = torch.from_numpy(y_test).to(device)

    optimizer = torch.optim.Adam(net.parameters(), lr=0.01)
    loss_func = torch.nn.MSELoss()

    cur_loss = 10 ** 5
    best_l = 1e-3
    checkpoint = {}
    l_train = []
    l_test = []

    for e in range(n_epochs):
        prediction = net(x_)
        loss = loss_func(prediction, y_)
        l_train.append(loss.data.cpu().numpy())

        prediction_test = net(x_test_)
        loss_test = loss_func(prediction_test, y_test_)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        cur_loss = loss_test.data.cpu().numpy()
        l_test.append(cur_loss)
        if cur_loss.item() < best_l:
            best_l = cur_loss.item()
            checkpoint = {
                "n_hidden": net.n_hidden,
                "n_layers": net.n_layers,
                "model_state_dict": net.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
            }
        if (e + 1) % 100 == 0:
            print(f"\tEpoch: {e + 1}\tL2 Loss = {loss.data.cpu().numpy()}")

    return best_l, checkpoint, l_train, l_test
