#!/usr/bin/python
#-*- coding: utf-8 -*-

from module import *
from layer import *
from args import get_parser

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.optim
import torch.utils.data
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models


def norm(inputs, p=2, dim=1, eps=1e-12):
	return inputs / inputs.norm(p,dim,keepdim=True).clamp(min=eps).expand_as(inputs)


class encoder_model(nn.Module):
	def __init__(self,
				model_dim,
				len_max_seq,
				num_layer=6,
				num_heads=8,
				dropout_rate=0.1,
				use_posenc=True):

		super(encoder_model, self).__init__()

		self.model_dim = model_dim
		self.len_max_seq = len_max_seq
		self.num_heads = num_heads
		self.num_layer = num_layer
		self.dropout_rate = dropout_rate
		self.use_posenc = use_posenc

		n_position = self.len_max_seq + 1

		self.position_enc = nn.Embedding.from_pretrained(
			get_sinusoid_encoding_table(n_position, self.model_dim, padding_idx=0),
			freeze=True)

		self.position_ind = torch.unsqueeze(torch.arange(1, self.len_max_seq +1), 0)

		
		self.layer_stack = nn.ModuleList([
			encoder_layer(self.model_dim, self.num_heads, dropout_rate=self.dropout_rate)
			for _ in range(self.num_layer)])
		

		'''
		# for UT
		self.enc_layer = encoder_layer(self.model_dim, self.num_heads, dropout_rate=self.dropout_rate)
		'''

	def forward(self, inputs):
		N, T, d_word_vec = inputs.size()
		# -- Prepare masks
		slf_attn_mask = get_attn_key_pad_mask(seq_k=inputs, seq_q=inputs)
		non_pad_mask = get_non_pad_mask(inputs)

		pos_ind = (self.position_ind.cpu().repeat(N,1) * non_pad_mask.cpu().squeeze().long()).to(device=torch.device('cuda'))

		
		if self.use_posenc:
			enc_output = inputs + self.position_enc(pos_ind)
		else:
			enc_output = inputs

		for enc_layer in self.layer_stack:
			enc_output = enc_layer(
				inputs = enc_output,
				non_pad_mask=non_pad_mask,
				slf_attn_mask=slf_attn_mask)
		

		'''
		# for UT
		enc_output = inputs
		for lay in range(self.num_layer):
			if self.use_posenc:
				enc_output = enc_output + self.position_enc(pos_ind)
			enc_output = self.enc_layer(
					inputs=enc_output,
					non_pad_mask=non_pad_mask,
					slf_attn_mask=slf_attn_mask)
		'''

		return enc_output

class co_attention_model(nn.Module):
	def __init__(self,
				wordDim,
				wordModelDim,
				imageDim,
				wordMaxlen,
				imageMaxlen,
				num_layer=6, 
				num_heads=8,
				dropout_rate = 0.1):

		super(co_attention_model, self).__init__()

		self.wordDim = wordDim
		self.wordModelDim = wordModelDim
		self.imageDim = imageDim

		self.wordMaxlen = wordMaxlen
		self.imageMaxlen = imageMaxlen

		self.num_layer = num_layer
		self.num_heads = num_heads
		self.dropout_rate = dropout_rate

		self.conv = nn.Conv1d(wordDim, wordModelDim, 1, bias=False)

		self.step_encoder = encoder_model(model_dim = self.wordModelDim,
										len_max_seq = self.wordMaxlen,
										num_heads = self.num_heads,
										num_layer = self.num_layer,
										dropout_rate = self.dropout_rate)


	# def forward(self, inputs_words, inputs_images):
	def forward(self, inputs_words): # delete inputs_images
		# step self-encoder
		inputs = inputs_words.transpose(1, 2)
		inputs = self.conv(inputs)
		inputs = inputs.transpose(1, 2)

		self.step_enc = self.step_encoder(inputs)

		return self.step_enc

class recipe_model(nn.Module):
	def __init__(self,
				titleDim,
				ingrDim,
				wordDim,
				wordModelDim,
				imageDim,
				titleMaxlen,
				ingrMaxlen,
				wordMaxlen,
				imageMaxlen,
				num_layer=6,
				num_heads=8,
				embedding_dim=1024,
				dropout_rate=0.1):
		super(recipe_model, self).__init__()

		self.titleDim = titleDim
		self.ingrDim = ingrDim
		self.wordDim = wordDim
		self.wordModelDim = wordModelDim
		self.imageDim = imageDim

		self.titleMaxlen = titleMaxlen
		self.ingrMaxlen = ingrMaxlen
		self.wordMaxlen = wordMaxlen
		self.imageMaxlen = imageMaxlen

		self.num_layer = num_layer
		self.num_heads = num_heads
		self.embedding_dim = embedding_dim
		self.dropout_rate = dropout_rate

		#self.image_pooling = nn.MaxPool1d(self.imageMaxlen, 1)

		self.title_model = encoder_model(model_dim = self.titleDim,
										len_max_seq = self.titleMaxlen,
										num_heads = self.num_heads,
										num_layer = self.num_layer,
										dropout_rate = self.dropout_rate)
		
		self.ingr_model = encoder_model(model_dim = self.ingrDim,
										len_max_seq = self.ingrMaxlen,
										num_heads = self.num_heads,
										num_layer = self.num_layer,
										dropout_rate = self.dropout_rate)
		
		self.co_step_model = co_attention_model(wordDim = self.wordDim, 
												wordModelDim = self.wordModelDim,
												imageDim = self.imageDim,
												wordMaxlen = self.wordMaxlen,
												imageMaxlen = self.imageMaxlen,
												num_layer = self.num_layer, 
												num_heads = self.num_heads,
												dropout_rate = self.dropout_rate)

		self.title_attn = global_attention(inputs_dim = self.titleDim,
											context_dim = self.titleDim)
		self.ingr_attn = global_attention(inputs_dim = self.ingrDim,
											context_dim = self.ingrDim)
		self.step_attn = global_attention(inputs_dim = self.wordModelDim,
											context_dim = self.wordModelDim)
		
		self.embedding = nn.Sequential(
				nn.Linear(self.ingrDim+self.titleDim+self.wordModelDim, self.embedding_dim),
				nn.Tanh()
			)
		'''
		self.embedding = nn.Sequential(
				nn.Linear(self.wordModelDim, self.embedding_dim),
				nn.Tanh()
			)
		'''
		'''
		self.embedding = nn.Sequential(
				nn.Linear(self.ingrDim, self.embedding_dim),
				nn.Tanh()
			)
		'''
		'''
		self.embedding = nn.Sequential(
				nn.Linear(self.titleDim, self.embedding_dim),
				nn.Tanh()
			)
		'''
		

	# def forward(self, inputs_title, inputs_ingrs, inputs_words, inputs_images):
	def forward(self, inputs_title, inputs_ingrs, inputs_words): # delete input_images
		self.title_featue = self.title_model(inputs_title)
		self.title_featue = self.title_attn(self.title_featue)
		
		self.ingr_feature = self.ingr_model(inputs_ingrs)
		self.ingr_feature = self.ingr_attn(self.ingr_feature)

		# self.co_step_feature = self.co_step_model(inputs_words, inputs_images)
		self.co_step_feature = self.co_step_model(inputs_words) # delete inputs_images
		self.co_step_feature = self.step_attn(self.co_step_feature)

		outputs = torch.cat([self.title_featue, self.ingr_feature, self.co_step_feature], 1)
		#outputs = self.title_featue
		outputs = norm(self.embedding(outputs))
		return outputs

class image_model(nn.Module):
	def __init__(self,
				imageDim,
				embedding_dim=1024,
				dropout_rate = 0.1):
		
		super(image_model, self).__init__()

		self.imageDim = imageDim
		self.embedding_dim = embedding_dim
		self.dropout_rate = dropout_rate

		resnet50 = models.resnet50(pretrained=True)
		modules = list(resnet50.children())[:-1]
		self.resnet = nn.Sequential(*modules)

		self.embedding = nn.Sequential(
				nn.Linear(self.imageDim, self.embedding_dim),
				nn.Tanh()
			)

	def forward(self, inputs_finalimage):
		outputs = self.resnet(inputs_finalimage)
		outputs = outputs.view(outputs.size(0), -1)

		outputs = self.embedding(outputs)
		outputs = norm(outputs)
		return outputs

class full_model(nn.Module):
	def __init__(self,
				titleDim,
				ingrDim,
				wordDim,
				wordModelDim,
				imageDim,
				titleMaxlen,
				ingrMaxlen,
				wordMaxlen,
				imageMaxlen,
				margin,
				num_layer=6,
				num_heads=8,
				embedding_dim=1024,
				dropout_rate = 0.1):
		
		super(full_model, self).__init__()

		self.titleDim = titleDim
		self.ingrDim = ingrDim
		self.wordDim = wordDim
		self.wordModelDim = wordModelDim
		self.imageDim = imageDim

		self.titleMaxlen = titleMaxlen
		self.ingrMaxlen = ingrMaxlen
		self.wordMaxlen = wordMaxlen
		self.imageMaxlen = imageMaxlen


		self.margin = margin
		self.num_layer = num_layer
		self.num_heads = num_heads
		self.embedding_dim = embedding_dim
		self.dropout_rate = dropout_rate

		self.recipe = recipe_model(titleDim = self.titleDim,
								ingrDim = self.ingrDim,
								wordDim = self.wordDim,
								wordModelDim = self.wordModelDim,
								imageDim = self.imageDim,
								titleMaxlen = self.titleMaxlen,
								ingrMaxlen = self.ingrMaxlen,
								wordMaxlen = self.wordMaxlen,
								imageMaxlen = self.imageMaxlen,
								num_layer = self.num_layer,
								num_heads = self.num_heads,
								embedding_dim = self.embedding_dim,
								dropout_rate = self.dropout_rate)
		
		self.image = image_model(imageDim = self.imageDim,
								embedding_dim = self.embedding_dim,
								dropout_rate = self.dropout_rate)

	def get_img_params(self):
		return self.image.resnet.parameters()

	def get_rec_params(self):
		params = []
		params.append({'params': self.recipe.parameters()})
		params.append({'params': self.image.embedding.parameters()})
		return params

	# def forward(self, inputs_title, inputs_ingrs, inputs_words, inputs_images, inputs_finalimage):
	# delete step-image: inputs_images
	def forward(self, inputs_title, inputs_ingrs, inputs_words, inputs_finalimage):
		# self.recipe_embedding = self.recipe(inputs_title, inputs_ingrs, inputs_words, inputs_images)
		self.recipe_embedding = self.recipe(inputs_title, inputs_ingrs, inputs_words)
		self.image_embedding = self.image(inputs_finalimage)

		return [self.image_embedding, self.recipe_embedding]

