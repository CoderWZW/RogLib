from model.base.GCN import GCN
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import networkx as nx
import numpy as np
import scipy.sparse as sp
from model.attack.nettack import Nettack
from matplotlib import pyplot as plt
from utils import maxminnorm_list

# the accuracy of the output
def accuracy(output, labels):
    pred = output.max(1)[1].type_as(labels)
    correct = pred.eq(labels).double()
    correct = correct.sum()
    
    return correct / len(labels)

def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(
    np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    
    return torch.sparse.FloatTensor(indices, values, shape)

class Attack(object):
	def __init__(self, config, data):
		print("attack initialization")
		self.data = data
		self.attack_node = config['attack_node'] # the node you want to attack
		if self.attack_node in self.data.split_unlabeled:
			print(self.attack_node + "in the test set")
		

		self.model_name = config['model_name']
		self.direct_attack = True # set the attack way
		self.n_influencers = 1 if self.direct_attack else 5
		self.n_perturbations = int(self.data.degrees[int(self.attack_node)]) # How many perturbations to perform. Default: Degree of the node
		self.perturb_features = True # feature attack
		self.perturb_structure = True # structure attack
		self.retrain_iters=5

		self.classification_margins_clean = []
		self.class_distrs_clean = []
		self.classification_margins_corrupted = []
		self.class_distrs_retrain = []
		#print(self.attack_node)

	# train a surrogate model (i.e.  GCN without nonlinear activation) used in train_base_model_without_perturbations & train_base_model_with_perturbations
	def train_surrogate_model(self):
		print("data process")
		if self.model_name == 'nettack':
			A = sparse_mx_to_torch_sparse_tensor(self.data._An)
			X = sparse_mx_to_torch_sparse_tensor(self.data._X_obs)
			Y = torch.LongTensor(self.data._z_obs)
			model = GCN(self.data.sizes, X.shape[1]) # get GCN model
			optimizer = optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)
			epochs = 100
			for epoch in range(epochs):
				optimizer.zero_grad()
				output = model(X, A)
				loss_train = F.nll_loss(output[self.data.split_train], Y[self.data.split_train])
				acc_train = accuracy(output[self.data.split_train], Y[self.data.split_train])
				loss_train.backward()
				optimizer.step()

				loss_val = F.nll_loss(output[self.data.split_val], Y[self.data.split_val])
				acc_val = accuracy(output[self.data.split_val], Y[self.data.split_val])
				
				if (epoch % 10 == 0):
					print("Epoch: {}".format(epoch + 1),
						"loss_train: {:.4f}".format(loss_train.item()),
						"acc_train: {:.4f}".format(acc_train.item()),
						"loss_val: {:.4f}".format(loss_val.item()),
						"acc_val: {:.4f}".format(acc_val.item()))
		self.surrogate_model = model
		self.W1 = model.state_dict()['gcn1.weight'].numpy()
		self.W2 = model.state_dict()['gcn2.weight'].numpy()


		#numpy.ndarray
		print("surrogate_model trains successfully")

	# attack on the surrogate model according the model_name
	def model(self):
		print("model training")
		if self.model_name == 'nettack':
			self.attack_node = int(self.attack_node)
			self.nettack = Nettack(self.data._A_obs, self.data._X_obs, self.data._z_obs, self.W1, self.W2, self.attack_node, verbose=True)
			self.nettack.reset()
			# Perform an attack on the surrogate model.
			self.nettack.attack_surrogate(self.n_perturbations, perturb_structure=self.perturb_structure, perturb_features=self.perturb_features, direct=self.direct_attack, n_influencers=self.n_influencers)
			
	def generate(self):
		print("generate perturbations")
		print(self.nettack.structure_perturbations)
		print(self.nettack.feature_perturbations)

	def train_base_model_without_perturbations(self):
		print("train_base_model_without_perturbations")

		if self.model_name == 'nettack':
			A = sparse_mx_to_torch_sparse_tensor(self.data._An)
			X = sparse_mx_to_torch_sparse_tensor(self.data._X_obs)
			Y = torch.LongTensor(self.data._z_obs)
			model = self.surrogate_model
			optimizer = optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)
			for epoch in range(self.retrain_iters):
				optimizer.zero_grad()
				output = model(X, A)
				loss_train = F.nll_loss(output[self.data.split_train], Y[self.data.split_train])
				acc_train = accuracy(output[self.data.split_train], Y[self.data.split_train])
				loss_train.backward()
				optimizer.step()

				loss_val = F.nll_loss(output[self.data.split_val], Y[self.data.split_val])
				acc_val = accuracy(output[self.data.split_val], Y[self.data.split_val])
				
				print("Epoch: {}".format(epoch + 1),
					"loss_train: {:.4f}".format(loss_train.item()),
					"acc_train: {:.4f}".format(acc_train.item()),
					"loss_val: {:.4f}".format(loss_val.item()),
					"acc_val: {:.4f}".format(acc_val.item()))
				# the probability of the attack node before it be attacked
				probs_before_attack = output[self.attack_node]
				self.class_distrs_clean.append(maxminnorm_list(probs_before_attack.detach().cpu().numpy()))
			self.class_distrs_clean = np.array(self.class_distrs_clean)
			#print(self.class_distrs_clean)


	def train_base_model_with_perturbations(self):
		print("train_base_model_with_perturbations")
		if self.model_name == 'nettack':
			A = sparse_mx_to_torch_sparse_tensor(self.nettack.adj_preprocessed)
			X = sparse_mx_to_torch_sparse_tensor(self.nettack.X_obs.tocsr())
			Y = torch.LongTensor(self.data._z_obs)
			model = self.surrogate_model
			optimizer = optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)
			for epoch in range(self.retrain_iters):
				optimizer.zero_grad()
				output = model(X, A)
				loss_train = F.nll_loss(output[self.data.split_train], Y[self.data.split_train])
				acc_train = accuracy(output[self.data.split_train], Y[self.data.split_train])
				loss_train.backward()
				optimizer.step()

				loss_val = F.nll_loss(output[self.data.split_val], Y[self.data.split_val])
				acc_val = accuracy(output[self.data.split_val], Y[self.data.split_val])
				
				print("Epoch: {}".format(epoch + 1),
					"loss_train: {:.4f}".format(loss_train.item()),
					"acc_train: {:.4f}".format(acc_train.item()),
					"loss_val: {:.4f}".format(loss_val.item()),
					"acc_val: {:.4f}".format(acc_val.item()))
				# the probability of the attack node after it be attacked
				probs_after_attack = output[self.attack_node]
				
				self.class_distrs_retrain.append(maxminnorm_list(probs_after_attack.detach().cpu().numpy()))
			self.class_distrs_retrain = np.array(self.class_distrs_retrain)
			#print(self.class_distrs_retrain)
	
	# draw to show the result
	def show(self):
		def make_xlabel(ix, correct):
			if ix==correct:
				return "Class {}\n(correct)".format(ix)
			return "Class {}".format(ix)

		figure = plt.figure(figsize=(12,4))
		plt.subplot(1, 2, 1)
		center_ixs_clean = []
		for ix, block in enumerate(self.class_distrs_clean.T):
			x_ixs= np.arange(len(block)) + ix*(len(block)+2)
			center_ixs_clean.append(np.mean(x_ixs))
			color = '#555555'
			if ix == self.nettack.label_u:
				color = 'darkgreen'
			plt.bar(x_ixs, block, color=color)

		ax=plt.gca()
		plt.ylim((-.05, 1.05))
		plt.ylabel("Predicted probability")
		ax.set_xticks(center_ixs_clean)
		ax.set_xticklabels([make_xlabel(k, self.nettack.label_u) for k in range(self.data._K)])
		ax.set_title("Predicted class probabilities for node {} on clean data\n({} re-trainings)".format(self.nettack.u, self.retrain_iters))

		fig = plt.subplot(1, 2, 2)
		center_ixs_retrain = []
		for ix, block in enumerate(self.class_distrs_retrain.T):
			x_ixs= np.arange(len(block)) + ix*(len(block)+2)
			center_ixs_retrain.append(np.mean(x_ixs))
			color = '#555555'
			if ix == self.nettack.label_u:
				color = 'darkgreen'
			plt.bar(x_ixs, block, color=color)


		ax=plt.gca()
		plt.ylim((-.05, 1.05))
		ax.set_xticks(center_ixs_retrain)
		ax.set_xticklabels([make_xlabel(k, self.nettack.label_u) for k in range(self.data._K)])
		ax.set_title("Predicted class probabilities for node {} after {} perturbations\n({} re-trainings)".format(self.nettack.u, self.n_perturbations, self.retrain_iters))
		plt.tight_layout()
		#plt.show()
		plt.savefig("./result/comparison.png")

	
