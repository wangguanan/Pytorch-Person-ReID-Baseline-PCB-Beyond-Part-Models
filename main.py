import torchvision.transforms as transforms

import argparse
import os

from core import *
from tools import *


def main(config):

	# environments
	make_dirs(config.output_path)
	make_dirs(os.path.join(config.output_path, 'logs/'))
	make_dirs(os.path.join(config.output_path, 'models/'))


	# loaders
	transform_train = transforms.Compose([
		transforms.Resize([384, 192], interpolation=3),
		transforms.RandomHorizontalFlip(),
		transforms.ToTensor(),
		transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
	transform_test = transforms.Compose([
		transforms.Resize([384, 192], interpolation=3),
		transforms.ToTensor(),
		transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
	loaders = Loaders(config, transform_train, transform_test)


	# base
	base = Base(config, loaders)


	# logger
	logger = Logger(os.path.join(os.path.join(config.output_path, 'logs/'), 'log.txt'))
	logger('\n'*3)
	logger(config)


	# resume model
	if config.resume_train_epoch >= 0:
		base.resume_model(config.resume_train_epoch)
		start_train_epoch = config.resume_train_epoch
	else:
		start_train_epoch = 0


	# main loop
	for current_epoch in range(start_train_epoch, config.total_train_epochs):

		# train
		base.lr_scheduler.step(current_epoch)
		_, results = train_an_epoch(config, base, loaders)
		logger('Time: {};  Epoch: {};  {}'.format(time_now(), current_epoch, results))

		# save model
		if (current_epoch+1) % 10 == 0:
			base.save_model(current_epoch)

		# test
		if (current_epoch+1) % 10 == 0 and current_epoch+1 >=90:
			market_map, market_rank = test(config, base, loaders, 'market_test')
			duke_map, duke_rank = test(config, base, loaders, 'duke_test')
			logger('Time: {},  Dataset: Market  \nmAP: {} \nRank: {}'.format(time_now(), market_map, market_rank))
			logger('Time: {},  Dataset: Duke  \nmAP: {} \nRank: {}'.format(time_now(), duke_map, duke_rank))
			logger('')


if __name__ == '__main__':


	parser = argparse.ArgumentParser()

	parser.add_argument('--cuda', type=str, default='cuda')

	# dataset configuration
	parser.add_argument('--market_path', type=str, default='/home/wangguanan/datasets/PersonReID/Market/Market-1501-v15.09.15/')
	parser.add_argument('--duke_path', type=str, default='/home/wangguanan/datasets/PersonReID/Duke/DukeMTMC-reID/')

	parser.add_argument('--train_dataset', type=str, default='market_train', help='market_train, market2duke_train, duke_train, duke2market_train')

	# batch size configuration
	parser.add_argument('--p', type=int, default=18, help='person count in a batch')
	parser.add_argument('--k', type=int, default=4, help='images count of a person in a batch')

	# model configuration
	parser.add_argument('--part_num', type=int, default=6)
	parser.add_argument('--pid_num', type=int, default=751)

	# train configuration
	parser.add_argument('--resume_train_epoch', type=int, default=-1, help='-1 for no resuming')
	parser.add_argument('--total_train_epochs', type=int, default=120)
	parser.add_argument('--base_learning_rate', type=float, default=0.5)
	parser.add_argument('--milestones', nargs='+', type=int, default=[50, 80, 100])

	# logger configuration
	parser.add_argument('--output_path', type=str, default='out/base/')
	parser.add_argument('--max_save_model_num', type=int, default=3, help='0 for max num is infinit')

	config = parser.parse_args()
	main(config)



