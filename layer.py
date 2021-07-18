#!/usr/bin/python
#-*- coding: utf-8 -*-

import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from module import *


class encoder_layer(nn.Module):
	def __init__(self,
				model_dim,
				num_heads=8,
				dropout_rate=0.1):
		super(encoder_layer, self).__init__()

		self.model_dim = model_dim
		self.num_heads = num_heads
		self.dropout_rate = dropout_rate

		assert (self.model_dim % self.num_heads) == 0

		self.d_in = self.model_dim // self.num_heads

		self.slf_attn = MultiHeadAttention(num_heads = self.num_heads,
										q_dim= self.model_dim,
										k_dim = self.model_dim,
										d_in = self.d_in,
										dropout = self.dropout_rate)
		self.pos_ffn = PositionwiseFeedForward(
			self.model_dim, 4*self.model_dim, dropout=self.dropout_rate)

	def forward(self, inputs, non_pad_mask=None, slf_attn_mask=None):
		outputs = self.slf_attn(inputs, inputs, inputs, mask=slf_attn_mask)
		outputs *= non_pad_mask
		outputs = self.pos_ffn(outputs)
		outputs *= non_pad_mask

		return outputs

'''
class decoder_layer(nn.module):
	def __init__(self,
				model_dim,
				key_dim
				num_heads=8,
				dropout_rate=0.1):
		super(decoder_layer, self).__init__()

		self.model_dim = model_dim
		self.key_dim = key_dim
		self.num_heads = num_heads,
		self.dropout_rate = dropout_rate

		assert (self.model_dim % self.num_heads) == 0
		self.d_in = self.model_dim / self.num_heads

		self.slf_attn = MultiHeadAttention(num_heads = self.num_heads,
										q_dim= self.model_dim,
										k_dim = self.model_dim,
										d_in = self.d_in,
										dropout = self.dropout_rate)
		self.key_attn = MultiHeadAttention(num_heads = self.num_heads,
										q_dim= self.model_dim,
										k_dim = self.key_dim,
										d_in = self.d_in,
										dropout = self.dropout_rate)
		self.pos_ffn = PositionwiseFeedForward(
			self.model_dim, 4*self.model_dim, dropout=self.dropout_rate)

	def forward(self, inputs, keys, non_pad_mask=None, slf_attn_mask=None, key_attn_mask=None):
		outputs = self.slf_attn(inputs, inputs, inputs, mask=slf_attn_mask)
        outputs *= non_pad_mask

        outputs = self.key_attn(outputs, keys, keys, mask=key_attn_mask)
        outputs *= non_pad_mask

        outputs = self.pos_ffn(outputs)
        outputs *= non_pad_mask

        return outputs
'''


class word_conv(nn.Module):	
	def __init__(self,
				in_units,
				out_units,
				num_windows = 3,
				pool_size = 3):
		super(word_conv, self).__init__()

		self.out_units = out_units
		self.pool_size = pool_size
		self.num_windows = num_windows

		self.layer_stack0 = nn.ModuleList([nn.Sequential(
				nn.Conv1d(in_units, out_units, w),
				nn.Tanh(),
				) for w in range(1, self.num_windows+1)])

		self.maxpool0 = nn.MaxPool1d(pool_size, pool_size)

		self.layer_stack1 = nn.ModuleList([nn.Sequential(
				nn.Conv1d(out_units, out_units, w),
				nn.Tanh(),
				)for w in range(1, self.num_windows+1)])

		self.maxpool1 = nn.MaxPool1d(pool_size, pool_size)

		
	def forward(self, inputs):
		# phrase level
		phrase_Ngrams = []
		inputs = inputs.transpose(1,2)
		layer_no = 0
		for layer in self.layer_stack0:
			phrase_Ngrams.append( layer(F.pad(inputs, [0,layer_no,0,0])) )
			layer_no += 1

		phrase_outputs, _= torch.max(
			torch.cat([torch.unsqueeze(Ngram, 0) for Ngram in phrase_Ngrams], 0), 0)

		reduced_phrase_outputs = self.maxpool0(phrase_outputs)

		# step level
		step_Ngrams = []
		layer_no = 0
		for layer in self.layer_stack1:
			phrase_Ngrams.append( layer(F.pad(reduced_phrase_outputs, [0,layer_no,0,0])) )
			layer_no += 1

		step_outputs, s_ = torch.max(
			torch.cat([torch.unsqueeze(Ngram, 0) for Ngram in phrase_Ngrams], 0), 0)


		reduced_step_outputs = self.maxpool1(step_outputs)
		reduced_step_outputs = reduced_step_outputs.transpose(1,2)	

		return reduced_step_outputs

class attention(nn.Module):
	def __init__(self,
				inputs_dim,
				context_dim):
		super(attention, self).__init__()

		self.inputs_dim = inputs_dim
		self.context_dim = context_dim

		self.dense = nn.Sequential(
                nn.Linear(self.inputs_dim, self.context_dim),
                nn.Tanh()
            )
		self.softmax = nn.Softmax(dim=1)

	def forward(self, inputs, context):
		'''
		inputs: (N, T, C0)
		context: (N, C)
		mask: (N, T)
		outputs: (N, C0)
		'''
		Us = self.dense(inputs)  #(N, T, C)
		masks = torch.sum(inputs, -1).eq(0).type(torch.float) #(N, T)

		alphas = torch.bmm(Us, torch.unsqueeze(context, 2)) #(N, T, 1)
		alphas = torch.squeeze(alphas) #(N, T)
		alphas = alphas.masked_fill(masks, -np.inf)

		alphas = torch.unsqueeze(self.softmax(alphas), 1) #(N, 1, T)
		outputs = torch.squeeze(torch.bmm(alphas, inputs))

		return outputs

class global_attention(nn.Module):
	def __init__(self,
				inputs_dim,
				context_dim):
		super(global_attention, self).__init__()

		self.inputs_dim = inputs_dim
		self.context_dim = context_dim

		self.context_vec = nn.Parameter(torch.Tensor(self.context_dim))

		self.dense = nn.Sequential(
                nn.Linear(self.inputs_dim, self.context_dim),
                nn.Tanh()
            )
		self.softmax = nn.Softmax(dim=1)
		self.reset_parameters()

	def reset_parameters(self):
		stdv = 1. / math.sqrt(self.context_vec.size(0))
		nn.init.uniform_(self.context_vec, -stdv, stdv)

	def forward(self, inputs):
		'''
		inputs: (N, T, C0)
		mask: (N, T)
		outputs: (N, C0)
		'''
		batchsize = inputs.size(0)
		Us = self.dense(inputs)  #(N, T, C)
		masks = torch.sum(torch.abs(inputs), -1).eq(0)

		Uw = torch.unsqueeze(self.context_vec.expand(batchsize, self.context_dim), 2)	#(N, C, 1)

		alphas = torch.bmm(Us, Uw) #(N, T, 1)
		alphas = torch.squeeze(alphas) #(N, T)
		alphas = alphas.masked_fill(masks, -np.inf)

		alphas = torch.unsqueeze(self.softmax(alphas), -1) #(N, T, 1)
		outputs = torch.sum(inputs*alphas, 1)	#(N, C0)

		return outputs

