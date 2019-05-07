import torch
from tools import *


def train_an_epoch(config, base, loaders):

	base.set_train()
	meter = MultiItemAverageMeter()

	### we assume 100 iterations as an epoch
	for _ in xrange(100):

		### load a batch data
		imgs, pids, _ = loaders.train_iter.next_one()
		imgs, pids = imgs.to(base.device), pids.to(base.device)

		### forward
		_, _, _, logits_list, embeddings_list = base.model(imgs)

		### loss
		ide_loss, avg_logits = base.compute_ide_loss(logits_list, pids)
		source_acc = accuracy(avg_logits, pids, [1])[0]

		### optimize
		base.optimizer.zero_grad()
		ide_loss.backward()
		base.optimizer.step()

		### recored
		meter.update({'ide_loss': ide_loss, 'acc': source_acc})

	return meter.get_val(), meter.get_str()