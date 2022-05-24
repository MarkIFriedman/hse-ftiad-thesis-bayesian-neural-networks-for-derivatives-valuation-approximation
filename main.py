from utils import BlackScholes
from riskFuel import RiskFuelNet, fit_net
import numpy as np
import torch

seed = 314
np.random.seed(seed)
torch.manual_seed(seed)

n_samples = 10000
generator = BlackScholes()
x_train, y_train, dydx_train = generator.trainingSet(n_samples, seed=seed)
x_test, x_axis, y_test, dydx_test, vegas = generator.testSet(0.2 * n_samples, seed=seed)

net = RiskFuelNet(n_feature=1, n_hidden=100, n_layers=4, n_output=1)
n_epochs = 10
ls, checkpoint, l_train, l_test = fit_net(net, n_epochs, x_train, y_train,
                                          x_test, y_test)

