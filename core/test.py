import torch
import numpy as np
from tools import *


def test(config, base, loaders, test_dataset):

	def compute_feature(images):
		images_f = fliplr(images)
		images = images.to(base.device)
		images_f = images_f.to(base.device)
		_, features, _, _, _ = base.model(images)
		_, features_f, _, _, _ = base.model(images_f)
		features = (features + features_f)
		if base.part_num_c == 1:
			features = torch.unsqueeze(features, -1)
		return features

	def norm_feature(features):
		'''
		three dims, [feature_num, channels, height, width]
		:param features:
		:return:
		'''
		features = normalize(features)
		features = features.reshape([features.shape[0], features.shape[1] * features.shape[2]])
		return features

	def normalize(inputs):
		'''
		normalize 3d matrix along dim 1, there are 3 dims (0, 1, 2)
		:param inputs: 3d
		:return: normalized matrix
		'''
		norm = np.tile(np.sqrt(np.sum(np.square(inputs), axis=1, keepdims=True)), [1, inputs.shape[1], 1])
		return inputs / norm


	base.set_eval()

	# meters
	query_features_meter, query_pids_meter, query_cids_meter = CatMeter(), CatMeter(), CatMeter()
	gallery_features_meter, gallery_pids_meter, gallery_cids_meter = CatMeter(), CatMeter(), CatMeter()

	# init dataset
	if test_dataset == 'market_test':
		loaders = [loaders.market_query_loader, loaders.market_gallery_loader]
	elif test_dataset == 'duke_test':
		loaders = [loaders.duke_query_loader, loaders.duke_gallery_loader]

	# compute query and gallery features
	with torch.no_grad():
		for loader_id, loader in enumerate(loaders):
			for data in loader:
				# compute feautres
				images, pids, cids = data
				features = compute_feature(images)
				# save as query features
				if loader_id == 0:
					query_features_meter.update(features.data)
					query_pids_meter.update(pids)
					query_cids_meter.update(cids)
				# save as gallery features
				elif loader_id == 1:
					gallery_features_meter.update(features.data)
					gallery_pids_meter.update(pids)
					gallery_cids_meter.update(cids)

	print('Time: {}, Successfully Compute Feature'.format(time_now()))

	# norm features
	query_features = norm_feature(query_features_meter.get_val_numpy())
	gallery_features = norm_feature(gallery_features_meter.get_val_numpy())
	print('Time: {}, Successfully Norm Feature'.format(time_now()))

	# compute mAP and rank@k
	result = PersonReIDMAP(query_features, query_cids_meter.get_val_numpy(), query_pids_meter.get_val_numpy(),
						   gallery_features, gallery_cids_meter.get_val_numpy(), gallery_pids_meter.get_val_numpy(), dist='cosine')

	return result.mAP, list(result.CMC[0: 150])


