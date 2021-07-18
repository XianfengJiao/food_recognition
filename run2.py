#!/usr/bin/python
#-*- coding: utf-8 -*-

import os
import time
import random
import numpy as np
import pickle as pkl


import torch
import torch.nn as nn
import torch.nn.parallel
import torch.optim
import torch.utils.data
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models
import torch.backends.cudnn as cudnn

from batch_sampler import BatchSamplerTriplet
from data_loader import MyDataSet
from loss import Triplet
from args import get_parser
from model import full_model


parser = get_parser()
opts = parser.parse_args()

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

def setup_seed(seed):
     torch.manual_seed(seed)
     torch.cuda.manual_seed_all(seed)
     np.random.seed(seed)
     random.seed(seed)
     torch.backends.cudnn.deterministic = True

def main():
	model = full_model(titleDim = opts.titleDim,
						ingrDim = opts.ingrDim,
						wordDim = opts.wordDim,
						wordModelDim = opts.wordModelDim,
						imageDim = opts.imageDim,
						titleMaxlen = opts.titleMaxlen,
						ingrMaxlen = opts.ingrMaxlen,
						wordMaxlen = opts.wordMaxlen,
						imageMaxlen = opts.imageMaxlen,
						margin = opts.margin,
						num_layer = opts.numLayer,
						num_heads = opts.numHeads,
						embedding_dim = opts.embDim,
						dropout_rate = opts.dropout)
	model.cuda()

	criterion = Triplet().cuda()

	img_params = list(map(id, model.image.resnet.parameters()))
	rec_params   = filter(lambda p: id(p) not in img_params, model.parameters())
	optimizer = torch.optim.Adam([
					{'params': rec_params},
					{'params': model.image.resnet.parameters(), 'lr': opts.lr*opts.freeImage }
				], lr=opts.lr*opts.freeRecipe)
	
	if opts.restore:
		if os.path.isfile(opts.restore):
			print("=> loading checkpoint '{}'".format(opts.restore))
			checkpoint = torch.load(opts.restore)

			start_epoch = checkpoint['epoch']
			best_medr = checkpoint['best_medr']
			valtrack = checkpoint['valtrack']
			ckpts = checkpoint['ckpts']

			print('startepoch:', start_epoch)
			print('best medr:', best_medr)
			print('valtrack:', valtrack)
			print('ckpts:', ckpts)

			model.load_state_dict(checkpoint['state_dict'])
			optimizer.load_state_dict(checkpoint['optimizer'])
			
			opts.freeImage = checkpoint['freeImage']
			opts.freeRecipe = checkpoint['freeRecipe']
			#change_lr(optimizer, opts.freeRecipe, opts.freeImage, opts.lr)
		else:
			print("=> no checkpoint found at '{}'".format(opts.restore))
			start_epoch = 0
			best_medr = float('inf')
			valtrack = 0
			ckpts = []
	else:
		start_epoch = 0
		best_medr = float('inf')
		valtrack = 0
		ckpts = []


	normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
										std=[0.229, 0.224, 0.225])	

	# preparing data loader 
	data_loader = torch.utils.data.DataLoader(
			MyDataSet(
				transforms.Compose([
					transforms.Resize(256), # rescale the image keeping the original aspect ratio
					transforms.CenterCrop(224), # we get only the center of that rescaled
					transforms.ToTensor(),
					normalize,
				]),
				partition='all'),
			batch_size= opts.batch_size,
			shuffle=False,
			num_workers=opts.workers,
			pin_memory=True)
	print( 'Test loader prepared.')

	data_time = 0.0
	batch_time = 0.0
	valid_time = 0.0
	test_time = 0.0

	start_time = time.time()
	end = time.time()

	torch.save(model, '/home/pku1616/liny/food/APP/model_1.pth')
	model2 = torch.load('/home/pku1616/liny/food/APP/model_1.pth')

	'''
	validate(model, criterion, data_loader)
	'''

	print('total time:', time.time()-start_time)

def validate(model, criterion, data_loader):
	model.eval()

	with torch.no_grad():
		for i, (inputs, classes) in enumerate(data_loader):
			input_var = [] 
			for j in range(len(inputs)):
				input_var.append(torch.autograd.Variable(inputs[j]).cuda())

			output = model(input_var[0],input_var[1], input_var[2], input_var[3], input_var[4])

			if i==0:
				data0 = output[0].data.cpu().numpy()
				data1 = output[1].data.cpu().numpy()
			else:
				data0 = np.concatenate((data0,output[0].data.cpu().numpy()),axis=0)
				data1 = np.concatenate((data1,output[1].data.cpu().numpy()),axis=0)

			length = data0.shape[0]
			if length % 1000 == 0:
				print(length)
	
	print(data0.shape)
	print(data1.shape)

	torch.save(model, '/home/pku1616/liny/food/APP/model.pth')
	np.save('/home/pku1616/liny/food/APP/img_emb.npy', data0)
	np.save('/home/pku1616/liny/food/APP/rec_emb.npy', data1)



def save_checkpoint(state, ckpts):
	print('saving...')
	if not os.path.exists(opts.checkpoint):
		os.mkdir(opts.checkpoint)


	filename = os.path.join(opts.checkpoint, 
		'model_e%03d_v-%.3f.pth.tar'%(state['epoch'],state['best_medr']) )

	ckpts.append(filename)

	while len(ckpts) > opts.maxCkpt:
		removeFile = ckpts.pop(0)
		if os.path.exists(removeFile):
			os.remove(removeFile)

	state['ckpts'] = ckpts
	torch.save(state, filename)


if __name__ == '__main__':
	setup_seed(2019)
	main()