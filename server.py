#!/usr/bin/python
#-*- coding: utf-8 -*-

import os
import sys
import time
import random
import numpy as np
import pickle as pkl
import logging
import string
import json
import argparse

from PIL import Image

import torch
import torch.nn as nn
import torch.utils.data
import torchvision.transforms as transforms
import torchvision.datasets as datasets

import tornado.ioloop
import tornado.web

from args import get_parser
from model import full_model

parser = get_parser()
opts = parser.parse_args()

def default_loader(path):
    try:
        im = Image.open(path).convert('RGB')
        return im
    except:
        print(path)
        return Image.new('RGB', (224, 224), 'white')

transform = transforms.Compose([
	transforms.Resize(256), # rescale the image keeping the original aspect ratio
	transforms.CenterCrop(224), # we get only the center of that rescaled
	transforms.ToTensor(),
	transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
	])

def get_input(img_path):

	title = torch.FloatTensor(np.ones((20, 64))).unsqueeze(0).repeat(2, 1, 1)
	ingr = torch.FloatTensor(np.ones((30, 64))).unsqueeze(0).repeat(2, 1, 1)
	step = torch.FloatTensor(np.ones((400, 64))).unsqueeze(0).repeat(2, 1, 1)
	stepimg = torch.FloatTensor(np.ones((60, 2048))).unsqueeze(0).repeat(2, 1, 1)

	img = default_loader(img_path)
	img = transform(img)
	img = img.unsqueeze(0).repeat(2, 1, 1, 1)

	return title, ingr, step, stepimg, img


food_dict = pkl.load(open('/home/ubuntu/linyang/APP/food_dict_new2.pkl', 'rb'))
ids = pkl.load(open('/home/ubuntu/linyang/APP/rec_ids.pkl', 'rb'))
rec_emb = np.load('/home/ubuntu/linyang/APP/rec_emb.npy')

#Model = torch.load('/home/ubuntu/linyang/APP/model_1.pth', map_location='cpu')
Model = full_model(titleDim = opts.titleDim,
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
Model.load_state_dict(torch.load('/home/ubuntu/linyang/APP/model_state.pth', map_location='cpu'))
Model.eval()

def get_emb(img_path):
	with torch.no_grad():
		inputs = get_input(img_path)
		input_var = []
		for j in range(len(inputs)):
			input_var.append(torch.autograd.Variable(inputs[j]))

		output = Model(input_var[0],input_var[1], input_var[2], input_var[3], input_var[4])
	img_embedding =  output[0].data.numpy()[0]
	return np.reshape(img_embedding, (1, -1))

def get_recipe(img_path):

	embedding = get_emb(img_path)
	sims = np.dot(rec_emb,embedding.T).reshape((-1))
	sorting = np.argsort(sims)[::-1].tolist()[:1]

	return food_dict[ ids[sorting[0]] ]


class UploadFileHandler(tornado.web.RequestHandler):
	def get(self):
		self.write('''
		<html>
			<head><title>Upload File</title></head>
			<body>
				<form action='food' enctype="multipart/form-data" method='post'>
				<input type='file' name='file'/><br/>
				<input type='submit' value='submit'/>
				</form>
				<br/>
			</body>
		</html>
		''')

	def post(self):
		upload_path=os.path.join('/home/ubuntu/linyang/APP','tmp_files')
		file_metas=self.request.files['file']
		for meta in file_metas:
			filename='img.jpg'
			filepath=os.path.join(upload_path, filename)
			with open(filepath,'wb') as up:
				up.write(meta['body'])

			result = get_recipe(filepath)
			final_json = json.dumps(result, sort_keys=False, indent=4, ensure_ascii=False)
			print(final_json)
			self.finish(final_json)

app=tornado.web.Application([
	(r'/food', UploadFileHandler),
])

if __name__ == '__main__':
	print('start server')
	app.listen(8002)
	tornado.ioloop.IOLoop.instance().start()

