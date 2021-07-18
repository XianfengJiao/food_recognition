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
from model import full_model,image_model

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
	img = default_loader(img_path)
	img = transform(img)
	img = img.unsqueeze(0)

	return img

data_path = '/home/ubuntu/linyang/APP_new/'
#data_path = '../'

ids = pkl.load(open(os.path.join(data_path, 'rec_ids.pkl'), 'rb'))
rec_emb = np.load(os.path.join(data_path, 'rec_emb.npy'))

Model = image_model(imageDim = opts.imageDim,
					embedding_dim = opts.embDim,
					dropout_rate = opts.dropout)
Model.load_state_dict(torch.load(os.path.join(data_path, 'resnet_statedict_globatt.pth'), map_location='cpu'))
Model.eval()

def get_emb(img_path):
	with torch.no_grad():
		image = get_input(img_path)
		image_var = torch.autograd.Variable(image)
		
		output = Model(image_var)
	img_embedding =  output.data.numpy()
	return np.reshape(img_embedding, (1, -1))

def get_recipe(img_path, return_num=1):

	embedding = get_emb(img_path)
	sims = np.dot(rec_emb,embedding.T).reshape((-1))
	sorting = np.argsort(sims)[::-1].tolist()[:return_num]

	if return_num==1:
		idx = ids[sorting[0]]
		recipe = json.load(open(os.path.join(data_path, 'recipe_info', str(idx)+'.json')))
		return {'name':recipe['name'], 'ingredients':recipe['ingredients']}
	
	return_rec = []
	for i in range(return_num):
		idx = ids[sorting[i]]
		return_rec.append( json.load(open(os.path.join(data_path, 'recipe_info', str(idx)+'.json'))) )

	return return_rec


class BaseHandler(tornado.web.RequestHandler):
	def set_default_headers(self):
		self.set_header('Access-Control-Allow-Origin', '*')
		self.set_header('Access-Control-Allow-Headers', '*')
		self.set_header('Access-Control-Max-Age', 1000)
		#self.set_header('Content-type', 'application/json')
		self.set_header('Access-Control-Allow-Methods', 'POST, GET, OPTIONS')
		self.set_header('Access-Control-Allow-Headers',
						'authorization, Authorization, Content-Type, Access-Control-Allow-Origin, Access-Control-Allow-Headers, X-Requested-By, Access-Control-Allow-Methods')


class UploadFileHandler(BaseHandler):
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
		upload_path=os.path.join(data_path,'tmp_files')
		file_metas=self.request.files['file']
		for meta in file_metas:
			filename='img.jpg'
			filepath=os.path.join(upload_path, filename)
			with open(filepath,'wb') as up:
				up.write(meta['body'])

			result = get_recipe(filepath)
			final_json = json.dumps(result, sort_keys=False, indent=4, ensure_ascii=False)
			print(time.strftime("%I:%M:%S %d/%m/%Y"))
			print(final_json)
			self.finish(final_json)

class Get5Handler(BaseHandler):
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
		upload_path=os.path.join(data_path,'tmp_files_get5')
		file_metas=self.request.files['file']
		for meta in file_metas:
			filename='img.jpg'
			filepath=os.path.join(upload_path, filename)
			with open(filepath,'wb') as up:
				up.write(meta['body'])

			result = get_recipe(filepath, 5)
			final_json = json.dumps(result, sort_keys=False, indent=4, ensure_ascii=False)
			print(time.strftime("%I:%M:%S %d/%m/%Y"))
			print(final_json)
			self.finish(final_json)

app=tornado.web.Application([
	(r'/food', UploadFileHandler),
	(r'/recipe', Get5Handler),
])

if __name__ == '__main__':
	print('start server')
	app.listen(8002)
	tornado.ioloop.IOLoop.instance().start()

