#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
CS224N 2018-19: Homework 5
"""

### YOUR CODE HERE for part 1h
import torch
import torch.nn as nn
import torch.nn.functional as F

class Highway(nn.Module):
	def __init__(self, dim):
		super(Highway, self).__init__()
		self.proj = nn.Linear(dim, dim)
		self.gate = nn.Linear(dim, dim)

	def forward(self, x):
		gate = F.relu(self.gate(x))
		proj = torch.sigmoid(self.proj(x))
		return torch.mul(gate, proj) + torch.mul(1-gate, x)

### END YOUR CODE 