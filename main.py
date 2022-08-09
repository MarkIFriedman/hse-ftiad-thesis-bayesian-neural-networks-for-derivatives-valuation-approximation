import BlackScholes
from riskFuel import *
from differentialML import *
import numpy as np
import torch
from sklearn.metrics import mean_squared_error as mse

# fixing seed to be able to reproduce results
seed = 42
test_seed = 100
np.random.seed(seed)
torch.manual_seed(seed)

# generate train and test
n_samples = 10000
n_test = n_samples // 5
rng = {
    "spot": (0.5, 2),
    "time": (0, 3.0),
    "sigma": (0.1, 0.5),
    "rate": (-0.01, 0.03)
}

generator = BlackScholes.DataGen(rng['spot'], rng['time'], rng['sigma'], rng['rate'])
# xTrain, yTrain, dydxTrain = generator.dataset(n_samples, seed=seed)
# xTest, yTest, dydxTest = generator.dataset(n_test, seed=test_seed)
#
# print('training RiskFuel model')
# net = RiskFuelNet(n_feature=4, n_hidden=100, n_layers=4, n_output=1)
# n_epochs = 10
# ls, checkpoint, l_train, l_test = fit_net(net, n_epochs, xTrain, yTrain,
#                                           xTest, yTest)
#
# print(f"Best loss ={ls}")
#
# model = RiskFuelNet(n_feature=4,
#                     n_hidden=checkpoint["n_hidden"],
#                     n_layers=checkpoint["n_layers"],
#                     n_output=1)
# model.load_state_dict(checkpoint['model_state_dict'])
# model.eval()
# model.to(device)
#
# y_pred = model(to_tensor(xTest).to(device)).flatten().data.cpu().numpy()
# mse_err = mse(yTest, y_pred)
# print("MSE error on test set for RiskFuel is %.5f" % mse_err)

print("Training Differential ML")
weightSeed = None
xAxis, yTest, dydxTest, predvalues, preddiffs = \
    test(generator, n_samples, n_test, seed, test_seed, weightSeed, differential=True)

mse_err_val = mse(yTest, predvalues)
print("MSE error on test set for DiffML on values is %.9f" % mse_err_val)

mse_err_dy = mse(dydxTest, preddiffs)
print("mse error on test set for DiffML on derivs is %.9f" % mse_err_dy)
