import numpy as np
import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data as Data
import matplotlib.pyplot as plt
import math
from model import Extractor, Regressor

def predict(loader, E, R):
	for num, (batch_x, batch_y) in enumerate(loader):
		batch_x, batch_y = batch_x.to(device), batch_y.to(device)
		batch_prediction = R(E(batch_x)).detach().cpu().numpy() * math.sqrt(1000)
		if num == 0:
			prediction = batch_prediction
			reference = batch_y.detach().cpu().numpy() * math.sqrt(1000)
		else:
			prediction = np.concatenate((prediction, batch_prediction))
			reference = np.concatenate((reference, batch_y.detach().cpu().numpy()*math.sqrt(1000)))
	return prediction, reference
# Preprocessing
class Norm_params():
	def __init__(self, data):
		self.mean = np.zeros(np.shape(data)[-1])
		self.std = np.zeros(np.shape(data)[-1])
		for i in range(np.shape(data)[-1]):
			self.mean[i] = np.mean(data[:, :, i])
			self.std[i] = np.std(data[:, :, i])

	def norm(self, features):
		for i in range(np.shape(features)[-1]):
			features[:, :, i] = ( features[:, :, i] - self.mean[i] ) / self.std[i]

		return features

def info_transfer2FloatTensor(features, labels):
	features = torch.from_numpy(features).float()
	labels = torch.from_numpy(labels).float()
	labels_gas = labels[:, 0]
	labels_liquid = labels[:, 1]
	return features, labels_gas, labels_liquid





# Data Load and preprocess
dir_load = ['./water_training_features.npy']
dir_load.append(dir_load[0].replace('features.npy', 'labels.npy'))
dir_load.append(dir_load[0].replace('training', 'validation'))
dir_load.append(dir_load[1].replace('training', 'validation'))
dir_load.append('./threephase_features.npy')
dir_load.append(dir_load[-1].replace('features.npy', 'labels.npy'))
dir_save = ['./']

source_training_features = np.load(dir_load[0])
source_training_labels = np.load(dir_load[1])
source_validation_features = np.load(dir_load[2])
source_validation_labels = np.load(dir_load[3])
target_features = np.load(dir_load[4])
target_labels = np.load(dir_load[5])

source_params = Norm_params(np.concatenate((source_training_features, source_validation_features)))
source_training_features = source_params.norm(source_training_features)
source_validation_features = source_params.norm(source_validation_features)
target_features = source_params.norm(target_features)
source_training_features = np.transpose(source_training_features, (0, 2, 1))
source_validation_features = np.transpose(source_validation_features, (0, 2, 1))
target_features = np.transpose(target_features, (0, 2, 1))

source_training_features, source_training_gas_labels, source_training_liquid_labels = info_transfer2FloatTensor(
																source_training_features, source_training_labels)
source_validation_features, source_validation_gas_labels, source_validation_liquid_labels = info_transfer2FloatTensor(
																source_validation_features, source_validation_labels)
target_features, target_gas_labels, target_liquid_labels = info_transfer2FloatTensor(target_features, 
																						target_labels)

source_training_liquid_labels = source_training_liquid_labels / math.sqrt(1000)
source_validation_liquid_labels = source_validation_liquid_labels / math.sqrt(1000)
target_liquid_labels =target_liquid_labels / math.sqrt(1000)
length_target = target_gas_labels.size()[0]

target_dataset = Data.TensorDataset(target_features,target_liquid_labels)
Batch_size = 128
target_loader = Data.DataLoader(dataset=target_dataset, batch_size=Batch_size,
								shuffle=False, num_workers=2)

E = Extractor()
R = Regressor()
E.load_state_dict(torch.load('E_l2.pkl'))
R.load_state_dict(torch.load('R_l2.pkl'))
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
E.to(device)
R.to(device)
prediction_target, reference_target = predict(target_loader, E, R)

#plot
standard_line_x = [0, 6500]
standard_line_y = [0, 6500]
error_line_x_p500 = [0, 6000]
error_line_y_p500 = [500, 6500]
error_line_x_n500 = [500, 6500]
error_line_y_n500 = [0, 6000]
l_standar = plt.plot(standard_line_x, standard_line_y, 'k-', label='standard')
l_p500 = plt.plot(error_line_x_p500, error_line_y_p500, ':', color='lime', label='$\pm500kg/h$')
l_n500 = plt.plot(error_line_x_n500, error_line_y_n500, ':', color='lime')
l_predictions_IDAN_ow = plt.scatter(reference_target, prediction_target, s=20, c='r', marker='+', label='IDAN(Testing on Oil-Water-Air)')
plt.xlim((0, 6500))
plt.ylim((0, 6500))
plt.title('Water-Air'r'$\qquad\to\qquad$Oil-Water-Air')
plt.xlabel('Reference of liquid mass flowrate')
plt.ylabel('Prediction of liquid mass flowrate')
plt.text(50,6200,r'$kg/h$')
plt.text(6000,80,r'$kg/h$')
plt.show()







