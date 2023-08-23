import torch
import pickle
import numpy as np
from main import ESA
torch.set_default_dtype(torch.float64)

dataset = 'Sensor'
if dataset == 'Satellite':
    with open('models/LR_model_Satellite.pt', 'rb') as f:
        model_data = pickle.load(f)
    STR = 'Satellite'

elif dataset == 'Sensor':
    with open('models/LR_model_robot.pt', 'rb') as f:
        model_data = pickle.load(f)
    STR = 'Sensor_'

else:
    with open('models/LR_model_bank.pt', 'rb') as f:
        model_data = pickle.load(f)
    STR = 'Bank_'

LOSS = 'ESA'
init = 'zero'
filename = STR+'LR_'+LOSS+'_'+init+'_init.pckl'


k0_stp, i_stp, Num_of_Predictions = 1, 1, 2000
Version = 'Extended'
Truncate ='yes'

Weights = np.array(list(model_data[0].named_parameters())[0][1].detach().cpu(), dtype = np.float64)
Biases = np.array(list(model_data[0].named_parameters())[1][1].detach().cpu(), dtype = np.float64)
X, Y = model_data[3], model_data[4]


ESA(Weights, Biases, X, Y, k0_stp, i_stp, Num_of_Predictions, filename, STR, Truncate, Version)
