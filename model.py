import torch
import torch.nn as nn
import torch.nn.functional as F


class Extractor(nn.Module):
	def __init__(self):
		super(Extractor, self).__init__()
		self.conv1 = nn.Conv1d(3, 64, 3, stride=1)
		self.conv2 = nn.Conv1d(64, 64, 3, stride=1)
		self.maxpool1 = nn.MaxPool1d(2, stride=2)
		self.conv3 = nn.Conv1d(64, 128, 3, stride=1)
		self.conv4 = nn.Conv1d(128, 128, 3, stride=1)
		self.maxpool2 = nn.MaxPool1d(2, stride=2, padding=1)
		self.conv5 = nn.Conv1d(128, 256, 3, stride=1)
		self.conv6 = nn.Conv1d(256, 256, 3, stride=1)
		self.conv7 = nn.Conv1d(256, 256, 3, stride=1)
		self.maxpool3 = nn.MaxPool1d(2, stride=2, padding=1)
		self.conv8 = nn.Conv1d(256, 512, 3, stride=1)
		self.conv9 = nn.Conv1d(512, 512, 3, stride=1)
		self.conv10 = nn.Conv1d(512, 512, 3, stride=1)
		self.maxpool4 = nn.MaxPool1d(2, stride=2)
		self.conv11 = nn.Conv1d(512, 512, 3, stride=1)
		self.conv12 = nn.Conv1d(512, 512, 3, stride=1)
		self.conv13 = nn.Conv1d(512, 512, 3, stride=1)
		self.maxpool5 = nn.MaxPool1d(2, stride=2)

	def num_flat_features(self, x):
		size = x.size()[1:]
		num_features = 1
		for s in size:
			num_features *= s
		return num_features

	def forward(self, x):
		x = F.selu(self.conv1(x))
		x = F.selu(self.conv2(x))
		x = self.maxpool1(x)
		x = F.selu(self.conv3(x))
		x = F.selu(self.conv4(x))
		x = self.maxpool2(x)
		x = F.selu(self.conv5(x))
		x = F.selu(self.conv6(x))
		x = F.selu(self.conv7(x))
		x = self.maxpool3(x)
		x = F.selu(self.conv8(x))
		x = F.selu(self.conv9(x))
		x = F.selu(self.conv10(x))
		x = self.maxpool4(x)
		x = F.selu(self.conv11(x))
		x = F.selu(self.conv12(x))
		x = F.selu(self.conv13(x))
		x = self.maxpool5(x)
		return x

class Regressor(nn.Module):
	def __init__(self):
		super(Regressor, self).__init__()
		self.fc1 = nn.Linear(512*51, 512)
		self.fc2 = nn.Linear(512, 64)
		self.fc3 = nn.Linear(64, 1)


	def forward(self, x):
		x = x.view(-1, self.num_flat_features(x))
		x = F.selu(self.fc1(x))
		x = F.selu(self.fc2(x))
		x = self.fc3(x)
		return x

	def num_flat_features(self, x):
		size = x.size()[1:]
		num_features = 1
		for s in size:
			num_features *= s
		return num_features

class Discriminator(nn.Module):
	def __init__(self):
		super(Discriminator, self).__init__()

		self.fc1 = nn.Linear(512*51, 512)
		self.fc2 = nn.Linear(512, 64)
		self.fc3 = nn.Linear(64, 1)

	def forward(self, x):
		x = x.view(-1, self.num_flat_features(x))
		x = F.selu(self.fc1(x))
		x = F.selu(self.fc2(x))
		x = self.fc3(x)
		return x

	def num_flat_features(self, x):
		size = x.size()[1:]
		num_features = 1
		for s in size:
			num_features *= s
		return num_features

