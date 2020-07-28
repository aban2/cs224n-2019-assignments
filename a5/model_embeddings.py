#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
CS224N 2018-19: Homework 5
model_embeddings.py: Embeddings for the NMT model
Pencheng Yin <pcyin@cs.cmu.edu>
Sahil Chopra <schopra8@stanford.edu>
Anand Dhoot <anandd@stanford.edu>
Michael Hahn <mhahn2@stanford.edu>
"""

import torch
import torch.nn as nn

# Do not change these imports; your module names should be
#   `CNN` in the file `cnn.py`
#   `Highway` in the file `highway.py`
# Uncomment the following two imports once you're ready to run part 1(j)

from cnn import CNN
from highway import Highway
    
# End "do not change" 

class ModelEmbeddings(nn.Module): 
    """
    Class that converts input words to their CNN-based embeddings.
    """
    def __init__(self, embed_size, vocab):
        """
        Init the Embedding layer for one language
        @param embed_size (int): Embedding size (dimensionality) for the output 
        @param vocab (VocabEntry): VocabEntry object. See vocab.py for documentation.
        """
        super(ModelEmbeddings, self).__init__()

        ## A4 code
        # pad_token_idx = vocab.src['<pad>']
        # self.embeddings = nn.Embedding(len(vocab.src), embed_size, padding_idx=pad_token_idx)
        ## End A4 code

        ### YOUR CODE HERE for part 1j
        self.e_char = 50
        self.embed_size = embed_size # 词向量维度（输出维度ew）
        # 这里vocab预定义了__len__方法，所以直接调用len就行
        self.embedding = nn.Embedding(len(vocab), self.e_char)
        self.cnn_layer = CNN(self.e_char, self.embed_size)
        self.highway_layer = Highway(embed_size)
        self.dropout = nn.Dropout(0.3)

        ### END YOUR CODE

    def forward(self, input):
        """
        Looks up character-based CNN embeddings for the words in a batch of sentences.
        @param input: Tensor of integers of shape (sentence_length, batch_size, max_word_length) where
            each integer is an index into the character vocabulary

        @param output: Tensor of shape (sentence_length, batch_size, embed_size), containing the 
            CNN-based embeddings for each word of the sentences in the batch
        """
        ## A4 code
        # output = self.embeddings(input)
        # return output
        ## End A4 code

        ### YOUR CODE HERE for part 1j

        word_embeddings = []
        # input size: 10, 5, 21
        for words in input: # 从句子里取第一个单词
            # 此时 words (5, 21)是batch_size + 一个单词
            X_emb = self.embedding(words) # 处理后：5，21，50
            X_reshaped = X_emb.transpose(-1, -2) # 变成 5, 50, 21

            X_conv_out = self.cnn_layer(X_reshaped)
            X_highway = self.highway_layer(X_conv_out)
            X_drop = self.dropout(X_highway)
            word_embeddings.append(X_drop)

        word_embeddings = torch.stack(word_embeddings)
        return word_embeddings
        


        ### END YOUR CODE

