import io
import torch
import requests
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import torch.nn as nn
import torch
import cv2
from parti.encoder import CLIPTextEmbedder
import torch.nn.functional as F
from einops import rearrange
from torch import nn, Tensor
from torch.nn import TransformerEncoder, TransformerEncoderLayer
from torch.nn import TransformerDecoder, TransformerDecoderLayer
from einops import rearrange, repeat

import math




class PositionalEncoding(nn.Module):

	def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
		super().__init__()
		self.dropout = nn.Dropout(p=dropout)

		position = torch.arange(max_len).unsqueeze(1)
		div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
		pe = torch.zeros(max_len, 1, d_model)
		pe[:, 0, 0::2] = torch.sin(position * div_term)
		pe[:, 0, 1::2] = torch.cos(position * div_term)
		self.register_buffer('pe', pe)

	def forward(self, x: Tensor) -> Tensor:
		"""
		Arguments:
			x: Tensor, shape ``[seq_len, batch_size, embedding_dim]``
		"""
		x = x + self.pe[:x.size(0)]
		return self.dropout(x)



class TransformerModel(nn.Module):
	""" Container module with an encoder, a recurrent or transformer module, and a decoder.
	"""

	def __init__(self, d_model: int, nhead: int, d_hid: int,
				 nlayers: int, dropout: float = 0.5, n_trg_vocab: int = 8192, d_word_vec: int = 1024):
		super().__init__()
		self.model_type = 'Transformer'
		self.pos_encoder = PositionalEncoding(d_model, dropout)
		self.pos_encoder_dec= PositionalEncoding(512, dropout)
		encoder_layers = TransformerEncoderLayer(
			d_model, nhead, d_hid, dropout)
		self.transformer_encoder = TransformerEncoder(encoder_layers, nlayers)

		self.start_token = nn.Parameter(torch.randn(512))
		self.inp_word_emb  = CLIPTextEmbedder(
			arch='ViT-H-14', version='laion2b_s32b_b79k', freeze=True)
		

		
		self.trg_word_emb = nn.Embedding(n_trg_vocab, 512)
		decoder_layers = TransformerDecoderLayer(
			512, nhead, d_hid, dropout)
		self.transformer_decoder = TransformerDecoder(decoder_layers, nlayers)

		self.d_model = d_model
		self.linear_layer = nn.Linear(d_model, n_trg_vocab)

	def forward(self, src: Tensor, tgt: Tensor) -> Tensor:
		"""
		Arguments:
				src: Tensor, shape ``[seq_len, batch_size]``
				src_mask: Tensor, shape ``[seq_len, seq_len]``

		Returns:
				output Tensor of shape ``[seq_len, batch_size, ntoken]``
		"""

		# Encoder
		src = self.inp_word_emb(src)
		src = rearrange(src, 'b l e -> l b e')
		src = self.pos_encoder(src) 
		src_mask = torch.zeros(src.shape[0], src.shape[0]).cuda()
		enc_output = self.transformer_encoder(src, src_mask)

		# Decoder
		tgt_mask = torch.triu(torch.full(
			(tgt.shape[0]+1, tgt.shape[0]+1), float('-inf')), diagonal=1).cuda()
		tgt = self.trg_word_emb(tgt) 
		tgt = self.pos_encoder_dec(tgt)
		start_token = repeat(self.start_token, 'd -> 1 b d', b=tgt.shape[1])
		tgt = torch.cat([start_token, tgt], dim=0) 
		output = self.transformer_decoder(tgt, enc_output, tgt_mask)
		
		output = self.linear_layer(output)
		return output


# create tensor of shape 77,1,1024
src = ['hi']
target = torch.empty(10,1).long().cuda()

model = TransformerModel(d_model=1024, nhead=2, d_hid=1024, nlayers=1, dropout=0.1, n_trg_vocab=8192, d_word_vec=1024).cuda()
out = model(src, target)
print(out.shape)