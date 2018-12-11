import numpy as np
import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data as Data
import math
import random
from model import Extractor, Regressor, Discriminator

#Optimize
def d_optim(E, D, source_features, target_features, optimizer, n):
	source_features, target_features = source_features.to(device), target_features.to(device)
	optimizer.zero_grad()
	# 1. Compute the gradient penalty loss
	e_source = E(source_features).detach()
	e_target = E(target_features).detach()
	alpha = torch.from_numpy(np.random.random((e_target.size()[0], 1, 1)) + np.zeros(e_target.size())).float().to(device)
	interpolates = (alpha * e_target + (1 - alpha) * e_source).float().requires_grad_(True)
	d_interpolates = D(interpolates)
	gradients = torch.autograd.grad(outputs=d_interpolates, inputs=interpolates,
							grad_outputs=torch.ones(e_target.size()[0], 1).to(device), create_graph=True, 
							retain_graph=True, only_inputs=True)[0]
	gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
	# 2. D loss
	d_loss = -n * (torch.mean(D(e_target)) + torch.mean(D(e_source))) + gradient_penalty
	d_loss.backward()
	E.zero_grad()
	optimizer.step()


def e_optim(E, R, D, source_features, source_labels, target_features, e_optimizer, m):
	source_features, target_features = source_features.to(device), target_features.to(device)
	source_labels = torch.unsqueeze(source_labels, dim=1).to(device)
	E.zero_grad()
	D.zero_grad()
	R.zero_grad()
	# 1. R loss
	r_predictions = R(E(source_features))
	r_loss = torch.mean(torch.max(torch.pow(r_predictions - source_labels, 2) - 50, torch.zeros(r_predictions.size()[0],1).to(device)))

	# 2. D loss
	d_source = D(E(source_features))
	d_target = D(E(target_features))
	d_loss = torch.pow(torch.mean(d_target) - torch.mean(d_source), 2)

	# 3. E loss
	e_loss = m * d_loss + r_loss
	e_loss.backward()
	R.zero_grad()
	e_optimizer.step()

def r_optim(E, R, source_features, source_labels, r_optimizer):
	source_features = source_features.to(device)
	source_labels = torch.unsqueeze(source_labels, dim=1).to(device)
	R.zero_grad()
	# R loss
	r_predictions = R(E(source_features).detach())
	r_loss = torch.mean(torch.max(torch.pow(r_predictions - source_labels, 2) - 50, torch.zeros(r_predictions.size()[0],1).to(device)))
	r_loss.backward()
	E.zero_grad()
	r_optimizer.step()


# Evaluete net
def evaluate(E, R, loader, length):
	for num, (batch_x, batch_y) in enumerate(loader):
		batch_x, batch_y = batch_x.to(device), batch_y.to(device)
		r_batch_loss, r_batch_error = r_evaluate(E, R, batch_x, batch_y)

		if num == 0:
			r_loss, r_error = r_batch_loss, r_batch_error
		else:
			r_loss += r_batch_loss
			r_error += r_batch_error
	r_loss = r_loss / length
	r_error = r_error / length

	return r_loss, r_error

def r_evaluate(E, R, features, labels):
	evaluate_prediction = R(E(features))
	labels = torch.unsqueeze(labels, dim=1)
	evaluate_loss = 0.5 * torch.sum(torch.pow(labels - evaluate_prediction, 2))
	evaluate_error = torch.sum(torch.abs(labels - evaluate_prediction) / torch.abs(labels))

	return evaluate_loss.detach().cpu().numpy(), evaluate_error.detach().cpu().numpy()




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


# Initial Params
def weights_init(m):
	if isinstance(m, nn.Conv1d):
		n = list(m.kernel_size)[0] * m.in_channels
		nn.init.normal(m.weight.data, mean=0, std=math.sqrt(1. / n))
		nn.init.normal(m.bias.data, mean=0, std=math.sqrt(1. / n))

	elif isinstance(m, nn.Linear):
		n = float(m.in_features)
		nn.init.normal(m.weight.data, mean=0, std=math.sqrt(1. / n))
		nn.init.normal(m.bias.data, mean=0, std=math.sqrt(1. / n))


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
length_source_training = source_training_gas_labels.size()[0]
length_source_validation = source_validation_gas_labels.size()[0]
length_target = target_gas_labels.size()[0]



#Parameters
Batch_size_s = int(length_source_training / 25) + 1
Batch_size_t = int(length_target / 25) + 1
num_epochs = 200
m = 1
n = 10 ** (-7)
E = Extractor()
D = Discriminator()
R = Regressor()
E.apply(weights_init)
D.apply(weights_init)
R.apply(weights_init)
e_learning_rate = 0.00003
d_learning_rate = 0.00015
r_learning_rate = 0.0000001
e_optimizer = optim.RMSprop(E.parameters(), lr=e_learning_rate, alpha=0.9)
d_optimizer = optim.RMSprop(D.parameters(), lr=d_learning_rate, alpha=0.9)
r_optimizer = optim.RMSprop(R.parameters(), lr=r_learning_rate, alpha=0.9)
e_steps = 1
d_steps = 1
r_steps = 1

#SAMPLING
source_training_dataset = Data.TensorDataset(source_training_features,
											source_training_liquid_labels)
source_validation_dataset = Data.TensorDataset(source_validation_features,
											source_validation_liquid_labels)
target_dataset = Data.TensorDataset(target_features,
									target_liquid_labels)

source_training_loader_d = Data.DataLoader(dataset=source_training_dataset, batch_size=Batch_size_s,
										shuffle=True, num_workers=2)
source_validation_loader = Data.DataLoader(dataset=source_validation_dataset, batch_size=Batch_size_s,
										shuffle=False, num_workers=2)
target_loader_d = Data.DataLoader(dataset=target_dataset, batch_size=Batch_size_t,
								shuffle=True, num_workers=2)

source_training_loader_e = Data.DataLoader(dataset=source_training_dataset, batch_size=Batch_size_s,
										shuffle=True, num_workers=2)
target_loader_e = Data.DataLoader(dataset=target_dataset, batch_size=Batch_size_t,
								shuffle=True, num_workers=2)

#Training
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
E.to(device)
D.to(device)
R.to(device)
for epoch in range(num_epochs):
	print('Epoch:', epoch+1, 'Training')
	# Evaluate Step
	if epoch%3 == 0:
		# 1. Source_validation
		source_validation_loss, source_validation_error = evaluate(E, R, source_validation_loader, length_source_validation)
		# 2. Source_training
		source_training_loss, source_training_error = evaluate(E, R, source_training_loader_d, length_source_training)
		# 3. Target
		target_loss, target_error = evaluate(E, R, target_loader_d, length_target)
		print('Source training loss:', source_training_loss, 'Source training error:', source_training_error)
		print('Source validation loss:', source_validation_loss, 'Source validation error:', source_validation_error)
		print('Target loss:', target_loss, 'Target error', target_error)
	# Save Step
		if epoch == 0:
			target_loss_minimum = target_loss
			target_error_minimum = target_error
		else:
			if target_loss < target_loss_minimum:
				torch.save(E.state_dict(), 'E_l1.pkl')
				torch.save(D.state_dict(), 'D_l1.pkl')
				torch.save(R.state_dict(), 'R_l1.pkl')
				target_loss_minimum = target_loss
			if target_error < target_error_minimum:
				torch.save(E.state_dict(), 'E_l2.pkl')
				torch.save(D.state_dict(), 'D_l2.pkl')
				torch.save(R.state_dict(), 'R_l2.pkl')
				target_error_minimum = target_error
	# Optim Step
	for num, ((source_batch_x_d, source_batch_y_d), (target_batch_x_d, target_batch_y_d), (source_batch_x_e, source_batch_y_e), (target_batch_x_e, target_batch_y_e)) in enumerate(zip(source_training_loader_d, target_loader_d, source_training_loader_e, target_loader_e)):
		# 1. Train D
		#print(source_batch_x_d.size()[0], target_batch_x_d.size()[0])
		source_batch_x_d = source_batch_x_d[0:target_batch_x_d.size()[0], :, :]
		source_batch_y_d = source_batch_y_d[0:target_batch_x_d.size()[0]]
		source_batch_x_e = source_batch_x_e[0:target_batch_x_d.size()[0], :, :]
		source_batch_y_e = source_batch_y_e[0:target_batch_x_d.size()[0]]

		for d_index in range(d_steps):
			d_optim(E, D, source_batch_x_d, target_batch_x_d, d_optimizer, n)
		# 3. Train E
		for e_index in range(e_steps):
			e_optim(E, R, D, source_batch_x_e, source_batch_y_e, target_batch_x_e, e_optimizer, m)

		for r_index in range(r_steps):
			r_optim(E, R, source_batch_x_e, source_batch_y_e, r_optimizer)






