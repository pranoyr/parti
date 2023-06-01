import io
import torch
import requests
import numpy as np
from .factory import create_model
from .utils.transform import stage1_transform
from PIL import Image, ImageDraw, ImageFont
import torch.nn as nn
import torch
# import cv2
from .encoder import CLIPTextEmbedder
from axial_positional_embedding import AxialPositionalEmbedding
from functools import partial
import torch.nn.functional as F
from einops import rearrange
from torch import nn, Tensor
from torch.nn import TransformerEncoder, TransformerEncoderLayer
from torch.nn import TransformerDecoder, TransformerDecoderLayer
from einops import rearrange, repeat
# from x_transformers import XTransformer
from parti.transformer import Transformer
from .t5 import TextEncoder, get_encoded_dim

# from parti.t5 import t5_encode_text, get_encoded_dim

import random



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
		# self.pos_encoder = PositionalEncoding(d_model, dropout)

		self.pos_encoder = AxialPositionalEmbedding(
			dim = 1024,
			axial_shape = (32, 32),          # axial shape will multiply up to the maximum sequence length allowed (64 * 64 = 4096)
			axial_dims = (512, 512)          # if not specified, dimensions will default to 'dim' for all axials and summed at the end. if specified, each axial will have the specified dimension and be concatted together. the concatted dimensions needs to sum up to the `dim` (256 + 256 = 512)
		)
		# encoder_layers = TransformerEncoderLayer(
		# 	d_model, nhead, d_hid, dropout)
		# self.transformer_encoder = TransformerEncoder(encoder_layers, nlayers)

		self.start_token = nn.Parameter(torch.randn(d_model))
		# self.text_encoder  = CLIPTextEmbedder(
		# 	arch='ViT-H-14', version='laion2b_s32b_b79k', freeze=True)

		# self.text_encoder = partial(t5_encode_text, name = 't5-large')
		self.text_encoder = TextEncoder(name = 't5-large')
		

		self.trg_word_emb = nn.Embedding(n_trg_vocab, d_word_vec)
		decoder_layers = TransformerDecoderLayer(
			d_model, nhead, d_hid, dropout)
		self.transformer_decoder = TransformerDecoder(decoder_layers, nlayers)

		self.d_model = d_model
		self.linear_layer = nn.Linear(d_model, n_trg_vocab)

	def forward(self, tgt, texts=None, labels=None) -> Tensor:
		"""
		Arguments:
				tgt : the sequence to the decoder (required). shape ``(batch_size, seq_len)``
				text : the sequence to the encoder (required). shape ``(batch_size,)``
				labels : labels for computing the masked language modeling loss (optional). shape ``(batch_size, seq_len)``

		Returns:
				loss ,  logits
		"""

		# Encoder
		src, _ = self.text_encoder(texts)
		enc_output = rearrange(src, 'b l e -> l b e')
		# src = self.pos_encoder(src) 
		# src_mask = torch.zeros(src.shape[0], src.shape[0]).cuda()
		# enc_output = self.transformer_encoder(src, src_mask)
		# enc_output = self.text_encoder(texts)
		# enc_output = rearrange(enc_output, 'b l e -> l b e')

		# print(enc_output.shape)

		tgt = rearrange(tgt, 'b l -> l b')

		# Decoder
		tgt_mask = torch.triu(torch.full(
			(tgt.shape[0]+1, tgt.shape[0]+1), float('-inf')), diagonal=1).cuda()
		tgt = self.trg_word_emb(tgt) 


		# tgt = self.pos_encoder(tgt)
		tgt = self.pos_encoder(rearrange(tgt, 'l b e -> b l e')) + rearrange(tgt, 'l b e -> b l e')
		tgt = rearrange(tgt, 'b l e -> l b e')


		start_token = repeat(self.start_token, 'd -> 1 b d', b=tgt.shape[1])
		tgt = torch.cat([start_token, tgt], dim=0) 
		output = self.transformer_decoder(tgt, enc_output, tgt_mask)
		
		output = self.linear_layer(output)

		# cross entropy loss
		if labels is not None:

			# labels = rearrange(labels, 'b l -> (l b)')
			loss = F.cross_entropy(rearrange(output, 'l b e -> b e l'), labels)
			
			return loss, rearrange(output, 'l b e -> b l e')


		return rearrange(output, 'l b e -> b l e')
	
	# def generate(self, src):
	# 	with torch.no_grad():
	# 		src_mask = torch.zeros(src.shape[0], src.shape[0]).cuda()
	# 		src = self.pos_encoder(src) 
	# 		enc_output = self.transformer_encoder(src, src_mask)

	# 		bos_token = torch.tensor([[8192]]).cuda()
	# 		gen_seq = bos_token

	# 		for step in range(1, 1025):
	# 			dec_input = self.trg_word_emb(gen_seq) 
	# 			dec_input = self.pos_encoder(dec_input)
	# 			dec_output = self.transformer_decoder(dec_input, enc_output)
	# 			dec_output = self.linear_layer(dec_output)[-1]
	# 			dec_output = F.softmax(dec_output, dim=1)  # (B, NUM_CLASSES)
	# 			dec_output = torch.argmax(dec_output, dim=1) 
	# 			gen_seq = torch.cat([gen_seq, dec_output.unsqueeze(0)])
	# 			# if step == 2:
	# 			# 	break
	# 		return gen_seq[1:]
		
	def generate(self, src):
		with torch.no_grad():
			# src_mask = torch.zeros(src.shape[0], src.shape[0]).cuda()

			# src = self.pos_encoder(src)
			enc_output, enc_mask = self.text_encoder(src)
			enc_output = rearrange(enc_output, 'b l e -> l b e')

			start_token = repeat(self.start_token, 'd -> 1 b d', b=enc_output.shape[1])
			#  generate first output
			dec_output = self.transformer_decoder(start_token, enc_output)
			dec_output = self.linear_layer(dec_output)[-1]
			dec_output = F.softmax(dec_output, dim=1)
			dec_output = torch.argmax(dec_output, dim=1) 
			gen_seq = dec_output.unsqueeze(0)

			for step in range(1, 1024):
				dec_input = self.trg_word_emb(gen_seq)


				# dec_input = self.pos_encoder(dec_input)
				dec_input = self.pos_encoder(rearrange(dec_input, 'l b e -> b l e')) + rearrange(dec_input, 'l b e -> b l e')
				dec_input = rearrange(dec_input, 'b l e -> l b e')
	
				# concat start token
				
				dec_input = torch.cat([start_token, dec_input], dim=0)
				
				dec_output = self.transformer_decoder(dec_input, enc_output)
				dec_output = self.linear_layer(dec_output)[-1]
				dec_output = F.softmax(dec_output, dim=1)
				dec_output = torch.argmax(dec_output, dim=1) 
				gen_seq = torch.cat([gen_seq, dec_output.unsqueeze(0)])
				# print(gen_seq)

			gen_seq = rearrange(gen_seq, 't b -> b t')
			return gen_seq


def restore(x):
	x = (x + 1) * 0.5
	x = x.permute(1, 2, 0).detach().cpu().numpy()
	x = (255*x).astype(np.uint8)
	# x = Image.fromarray(x)
	return x


class Parti(nn.Module):
	def __init__(self, cfg):
		super().__init__()
		self.cfg = cfg
		self.vqgan = create_model(
			arch='vqgan', version='vit-s-vqgan', pretrained=True, checkpoint_path="/home/pranoy/code/parti/output/models/vit_vq_step_270000.pt")
		self.vqgan.freeze()

		
		d_hid = 200  # dimension of the feedforward network model in ``nn.TransformerEncoder``
		nlayers = 20  # number of ``nn.TransformerEncoderLayer`` in ``nn.TransformerEncoder``
		nhead = 16  # number of heads in ``nn.MultiheadAttention``
		dropout = 0.2  # dropout probability
		# self.transformer = TransformerModel(
		# 	d_model=cfg.MODEL.D_MODEL, nhead=cfg.MODEL.NUM_HEADS, d_hid=cfg.MODEL.MLP_DIM, nlayers=cfg.MODEL.DEPTH, dropout=cfg.MODEL.DROPOUT, n_trg_vocab=8192, d_word_vec=1024)

		# self.transformer = XTransformer(
		# 			dim = 1024,
		# 			enc_num_tokens = 256, # input vocab size
		# 			enc_depth = 6,
		# 			enc_heads = 8,
		# 			enc_max_seq_len = 128,
		# 			dec_num_tokens = 8192, # target vocab size
		# 			dec_depth = 6,
		# 			dec_heads = 8,
		# 			dec_max_seq_len = 1024,
		# 			tie_token_emb = False      # tie embeddings of encoder and decoder
		# 		)

		# main class

		dim = 512
		seq_len = 1024
		heads = 8
		depth = 6
		vocab_size = 8192

		# self.transformer = Transformer(          # vit vqgan vae
		# 	dim = 512,                # model dimension
		# 	depth = 8,                # depth
		# 	dim_head = 64,            # attention head dimension
		# 	heads = 8,                # attention heads
		# 	dropout = 0.,             # dropout
		# 	cond_drop_prob = 0.25,    # conditional dropout, for classifier free guidance
		# 	ff_mult = 4,              # feedforward expansion factor
		# 	t5_name = 't5-large',     # name of your T5
		# )


		self.transformer = Transformer(
			dim = dim,
			vocab_size = vocab_size,
			seq_len = seq_len,
			dim_out = vocab_size,
			depth = depth,
			heads = heads
		)
			



	def forward(self, text, imgs):
		w, h = 256, 256

		device = imgs.device
		z, _, img_token_idcs = self.vqgan.encode(imgs)

		# encoded_text = self.text_encoder(text)


		# encoded_text = rearrange(encoded_text, 'b l e -> l b e')
		# img_token_idcs = rearrange(img_token_idcs	, 'b l -> l b')  

		# add eos and  bos token 
		# bos_token = torch.tensor([8192] * img_token_idcs.shape[1]).unsqueeze(0).cuda()
		
		# shifted_img_token_idcs = torch.cat((bos_token, img_token_idcs), dim=0)

		# eos_token = torch.tensor([8193] * img_token_idcs.shape[1]).unsqueeze(0).cuda()
		# tgt_img_token_idcs = torch.cat((img_token_idcs, eos_token), dim=0)
		
		# out_indices = self.transformer(text, img_token_idcs[:-1].cuda())
		loss , logits = self.transformer(img_token_idcs[:,:-1].to(device), texts = text, labels = img_token_idcs)
		# loss , logits = self.transformer(image_token_ids = img_token_idcs.to(device), texts = text, return_loss = True)


		# # xformer
		# loss , logits = self.transformer(text, img_token_idcs)

		# try:
		# sc
		# cv2.imwrite(f"results/{text[rand_num]}.png", output_image)
		# cv2.waitKey(1)

		
		# sample_text = ['A woman wearing a hair net cutting a large sheet cake.']
		# img = self.generate(sample_text)[0]
		# img = restore(img)
		# cv2.imwrite("result.png", img)
		






		# out_indices = rearrange(out_indices, 'l b e -> (b l) e')
		# tgt_img_token_idcs = rearrange(img_token_idcs, 'l b -> (b l)')

		# loss = F.cross_entropy(out_indices, tgt_img_token_idcs)
	
		return loss
		
	@ torch.no_grad()
	def generate(self, src):

		# encoded_text = self.text_encoder(text)
		# encoded_text = rearrange(encoded_text, 'b l e -> l b e')
		out_indices = self.transformer.generate(src)
		# print(out_indices.shape)

		# ind = torch.where(out_indices == 8193)
		# print(ind)

		# # print(out_indices.shape)
		# # decode to image
		# out_indices = out_indices.permute(1,0,2).squeeze(0)
		# out_indices = torch.argmax(out_indices, dim=1)
		# print(out_indices)
		# print(out_indices.shape)
		# out_indices = rearrange(out_indices, 't b -> b t')
		output_image = self.vqgan.decode_from_indice(out_indices)
		# print(output_image.shape)
		return output_image
