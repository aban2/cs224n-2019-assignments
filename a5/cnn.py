#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
CS224N 2018-19: Homework 5
"""

### YOUR CODE HERE for part 1i
import torch
import torch.nn as nn
import torch.nn.functional as F

class CNN(nn.Module):
	def __init__(self, dimchar, dimword, num_characters=21, kernel_size=5):
		super(CNN, self).__init__()
		self.conv = nn.Conv1d(dimchar, dimword, kernel_size) # stride默认是1
		self.pool = nn.MaxPool1d(kernel_size=num_characters-kernel_size+1)

	def forward(self, x): 
		# x: (batch_size, e_char, num_characters)
		conv = self.conv(x) # size是 3,17
		# 横着扫的，扫完变成1,17，3个filter变成3,17

		# max-pool继续横着扫，我们想得到3,1的（3是w_emb）
		conv_out = self.pool(F.relu(conv))

		return conv_out.squeeze(2)


### END YOUR CODE

