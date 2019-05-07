import torch
import torch.nn as nn
import torch.optim as optim

import os

from models import Model


def os_walk(folder_dir):
	for root, dirs, files in os.walk(folder_dir):
		files = sorted(files, reverse=True)
		dirs = sorted(dirs, reverse=True)
		return root, dirs, files


class Base:

	def __init__(self, config, loaders):

		self.config = config
		self.loaders = loaders

		# Model Configuration
		self.part_num = config.part_num
		self.pid_num = config.pid_num

		# Logger Configuration
		self.max_save_model_num = config.max_save_model_num
		self.output_path = config.output_path
		self.model_path = os.path.join(self.output_path, 'models/')
		self.log_path = os.path.join(self.output_path, 'logs/')

		# Train Configuration
		self.base_learning_rate = config.base_learning_rate
		self.milestones = config.milestones

		# init model
		self._init_device()
		self._init_model()
		self._init_creiteron()
		self._init_optimizer()


	def _init_device(self):
		self.device = torch.device('cuda')


	def _init_model(self):
		self.model = Model(part_num=self.part_num, class_num=self.config.pid_num)
		self.model = nn.DataParallel(self.model).to(self.device)


	def _init_creiteron(self):
		self.ide_creiteron = nn.CrossEntropyLoss()


	## compute average ide loss of all outputs
	def compute_ide_loss(self, logits_list, pids):
		avg_ide_loss = 0
		avg_logits = 0
		for i in xrange(self.part_num):
			logits_i = logits_list[i]
			avg_logits += 1.0 / float(self.part_num) * logits_i
			ide_loss_i = self.ide_creiteron(logits_i, pids)
			avg_ide_loss += 1.0 / float(self.part_num) * ide_loss_i
		return avg_ide_loss, avg_logits


	## init optimizer and lr_lr_scheduler
	def _init_optimizer(self):
		params = [{'params': self.model.module.resnet_conv.parameters(), 'lr': 0.1*self.base_learning_rate}]
		for i in xrange(self.part_num):
			params.append({'params': getattr(self.model.module, 'classifier' + str(i)).parameters(), 'lr': self.base_learning_rate})
		for i in xrange(self.part_num):
			params.append({'params': getattr(self.model.module, 'embedder' + str(i)).parameters(), 'lr': self.base_learning_rate})
		self.optimizer = optim.SGD(params=params, weight_decay=5e-4, momentum=0.9, nesterov=True)
		self.lr_scheduler = optim.lr_scheduler.MultiStepLR(self.optimizer, self.milestones, gamma=0.1)


	## save model as save_epoch
	def save_model(self, save_epoch):

		# save model
		file_path = os.path.join(self.model_path, 'model_{}.pkl'.format(save_epoch))
		torch.save(self.model.state_dict(), file_path)

		# if saved model is more than max num, delete the model with smallest iter
		if self.max_save_model_num > 0:
			root, _, files = os_walk(self.model_path)
			if len(files) > self.max_save_model_num:
				file_iters = sorted([int(file.replace('.pkl', '').split('_')[1]) for file in files], reverse=False)
				file_path = os.path.join(root, 'model_{}.pkl'.format(file_iters[0]))
				os.remove(file_path)


	## resume model from resume_epoch
	def resume_model(self, resume_epoch):
		model_path = os.path.join(self.model_path, 'model_{}.pkl'.format(resume_epoch))
		self.model.load_state_dict(torch.load(model_path))
		print('successfully resume model from {}'.format(model_path))


	## set model as train mode
	def set_train(self):
		self.model = self.model.train()

	## set model as eval mode
	def set_eval(self):
		self.model = self.model.eval()
