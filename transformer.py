import math
from random import random
from functools import partial

import torch
import torch.nn.functional as F
from torch import nn, einsum
import pathlib
from pathlib import Path
import torchvision.transforms as T

from typing import Callable, Optional, List

from einops import rearrange, repeat
from einops.layers.torch import Rearrange

from einops import rearrange, repeat

from axial_positional_embedding import AxialPositionalEmbedding


from parti.t5 import t5_encode_text, get_encoded_dim



# helpers

def exists(val):
	return val is not None

def default(val, d):
	return val if exists(val) else d

def eval_decorator(fn):
	def inner(model, *args, **kwargs):
		was_training = model.training
		model.eval()
		out = fn(model, *args, **kwargs)
		model.train(was_training)
		return out
	return inner

def l2norm(t):
	return F.normalize(t, dim = -1)

# tensor helpers

def get_mask_subset_prob(mask, prob, min_mask = 0):
	batch, seq, device = *mask.shape, mask.device
	num_to_mask = (mask.sum(dim = -1, keepdim = True) * prob).clamp(min = min_mask)
	logits = torch.rand((batch, seq), device = device)
	logits = logits.masked_fill(~mask, -1)

	randperm = logits.argsort(dim = -1).float()

	num_padding = (~mask).sum(dim = -1, keepdim = True)
	randperm -= num_padding

	subset_mask = randperm < num_to_mask
	subset_mask.masked_fill_(~mask, False)
	return subset_mask


# 2d relative positional bias

class RelPosBias2d(nn.Module):
	def __init__(self, size, heads):
		super().__init__()
		self.pos_bias = nn.Embedding((2 * size - 1) ** 2, heads)

		arange = torch.arange(size)

		pos = torch.stack(torch.meshgrid(arange, arange, indexing = 'ij'), dim = -1)
		pos = rearrange(pos, '... c -> (...) c')
		rel_pos = rearrange(pos, 'i c -> i 1 c') - rearrange(pos, 'j c -> 1 j c')

		rel_pos = rel_pos + size - 1
		h_rel, w_rel = rel_pos.unbind(dim = -1)
		pos_indices = h_rel * (2 * size - 1) + w_rel
		self.register_buffer('pos_indices', pos_indices)

	def forward(self, qk):
		i, j = qk.shape[-2:]

		bias = self.pos_bias(self.pos_indices[:i, :(j - 1)])
		bias = rearrange(bias, 'i j h -> h i j')

		bias = F.pad(bias, (j - bias.shape[-1], 0), value = 0.) # account for null key / value for classifier free guidance
		return bias

# classes

class LayerNorm(nn.Module):
	def __init__(self, dim):
		super().__init__()
		self.gamma = nn.Parameter(torch.ones(dim))
		self.register_buffer('beta', torch.zeros(dim))

	def forward(self, x):
		return F.layer_norm(x, x.shape[-1:], self.gamma, self.beta)

class GEGLU(nn.Module):
	""" https://arxiv.org/abs/2002.05202 """

	def forward(self, x):
		x, gate = x.chunk(2, dim = -1)
		return gate * F.gelu(x)

def FeedForward(dim, mult = 4):
	""" https://arxiv.org/abs/2110.09456 """

	inner_dim = int(dim * mult * 2 / 3)
	return nn.Sequential(
		LayerNorm(dim),
		nn.Linear(dim, inner_dim * 2, bias = False),
		GEGLU(),
		LayerNorm(inner_dim),
		nn.Linear(inner_dim, dim, bias = False)
	)

# attention

class Attention(nn.Module):
	def __init__(
		self,
		dim,
		*,
		context_dim = None,
		dim_head = 64,
		heads = 8,
		causal = False,
		dropout = 0.,
		norm_context = False,
		rel_pos_bias = False,
		encoded_fmap_size = None
	):
		super().__init__()
		self.causal = causal
		self.scale = dim_head ** -0.5
		self.norm = LayerNorm(dim)


		inner_dim = heads * dim_head
		context_dim = default(context_dim, dim)
		self.norm_context = LayerNorm(context_dim) if norm_context else nn.Identity()

		self.to_q = nn.Sequential(
			nn.Dropout(dropout),
			nn.Linear(dim, inner_dim, bias = False),
			Rearrange('b n (h d) -> b h n d', h = heads)
		)

		# needed for classifier free guidance for transformers
		# by @crowsonkb, adopted by the paper

		self.null_kv = nn.Parameter(torch.randn(dim_head))

		# one-headed key / value attention, from Shazeer's multi-query paper, adopted by Alphacode and PaLM

		self.to_kv = nn.Sequential(
			nn.Dropout(dropout),
			nn.Linear(context_dim, dim_head, bias = False)
		)

		self.to_out = nn.Sequential(
			Rearrange('b h n d -> b n (h d)'),
			nn.Linear(inner_dim, dim, bias = False),
			LayerNorm(dim)
		)

		# positional bias

		self.rel_pos_bias = None

		if rel_pos_bias:
			assert exists(encoded_fmap_size)
			self.rel_pos_bias = RelPosBias2d(encoded_fmap_size, heads)

	def forward(
		self,
		x,
		context = None,
		context_mask = None
	):
		batch, device = x.shape[0], x.device

		x = self.norm(x)

		q = self.to_q(x) * self.scale

		context = default(context, x)
		context = self.norm_context(context)

		kv = self.to_kv(context)

		null_kv = repeat(self.null_kv, 'd -> b 1 d', b = batch)
		kv = torch.cat((null_kv, kv), dim = 1)

		sim = einsum('b h i d, b j d -> b h i j', q, kv)

		if exists(self.rel_pos_bias):
			pos_bias = self.rel_pos_bias(sim)
			sim = sim + pos_bias

		mask_value = -torch.finfo(sim.dtype).max

		if exists(context_mask):
			context_mask = F.pad(context_mask, (1, 0), value = True)
			context_mask = rearrange(context_mask, 'b j -> b 1 1 j')
			sim = sim.masked_fill(~context_mask, mask_value)

		if self.causal:
			i, j = sim.shape[-2:]
			causal_mask = torch.ones((i, j), dtype = torch.bool, device = device).triu(j - i + 1)
			sim = sim.masked_fill(causal_mask, mask_value)

		attn = sim.softmax(dim = -1, dtype = torch.float32)
		out = einsum('b h i j, b j d -> b h i d', attn, kv)

		return self.to_out(out)


class TransformerBlocks(nn.Module):
	def __init__(
		self,
		dim,
		depth,
		dim_head = 64,
		heads = 8,
		ff_mult = 4,
		dropout = 0.,
		text_embed_dim = 512
	):
		super().__init__()
		self.layers = nn.ModuleList([])

		print(dim, dim_head, heads)
		self.image_encoded_dim = 32

		for _ in range(depth):
			self.layers.append(nn.ModuleList([
				Attention(dim, causal = True, encoded_fmap_size = self.image_encoded_dim, rel_pos_bias = True, dim_head = dim_head, heads = heads, dropout = dropout),
				Attention(dim, context_dim = text_embed_dim, dim_head = dim_head, heads = heads, dropout = dropout),
				FeedForward(dim, mult = ff_mult)
			]))

		self.norm = LayerNorm(dim)

	def forward(self, x, context = None, context_mask = None):
		for attn, cross_attn, ff in self.layers:
			x = attn(x) + x

			x = cross_attn(x, context = context, context_mask = context_mask) + x

			x = ff(x) + x

		return self.norm(x)

# transformer - it's all we need

class Transformer(nn.Module):
	def __init__(
		self,
		dim,
		vocab_size,
		seq_len,
		dim_out = 1,
		t5_name = "t5-large",
		**kwargs
	):
		super().__init__()
		self.dim = dim

		self.token_emb = nn.Embedding(vocab_size, dim)
		self.dim_out = dim_out
		self.pos_emb = AxialPositionalEmbedding(
				dim = 512,
				axial_shape = (32, 32),          # axial shape will multiply up to the maximum sequence length allowed (64 * 64 = 4096)
				axial_dims = (256, 256)          # if not specified, dimensions will default to 'dim' for all axials and summed at the end. if specified, each axial will have the specified dimension and be concatted together. the concatted dimensions needs to sum up to the `dim` (256 + 256 = 512)
			)

		self.seq_len = seq_len

		self.transformer_blocks = TransformerBlocks(dim = dim, **kwargs)
		self.norm = LayerNorm(dim)

		self.to_logits = nn.Linear(dim, dim_out, bias = False)

		self.encode_text = partial(t5_encode_text, name = t5_name)

		text_embed_dim = get_encoded_dim(t5_name)

		self.start_token = nn.Parameter(torch.randn(dim))

		self.text_embed_proj = nn.Linear(text_embed_dim, dim, bias = False) if text_embed_dim != dim else nn.Identity() 

		

	def forward(
		self,
		x,            # input image tokens
		labels = None,
		texts: Optional[List[str]] = None,
	):
		device, b, n = x.device, *x.shape

		# encode text
		text_embeds, context_mask = self.encode_text(texts)
		context = self.text_embed_proj(text_embeds)

		# token embedding
		x = self.token_emb(x)
		# add position embedding
		pos_emb = self.pos_emb(x)
		x = x + pos_emb

		# add start token
		start_token = repeat(self.start_token, 'd -> b 1 d', b=b)
		x = torch.cat((start_token, x), dim=1)

		print(x.shape)
		print(context.shape)

		# decoder
		embed = self.transformer_blocks(x, context = context, context_mask = context_mask)
		# to logits
		logits = self.to_logits(embed)
		# calculate loss
		loss = F.cross_entropy(rearrange(logits, 'b n c -> b c n'), labels)

		return loss, logits


# text = b,128, 512
# img = b, 1024, 512

# main class

dim = 512
seq_len = 1024
heads = 20
depth = 6
vocab_size = 8192


parti = Transformer(
	dim = dim,
	vocab_size = vocab_size,
	seq_len = seq_len,
	dim_out = vocab_size,
	depth = depth,
	heads = heads
).cuda()
	



texts = ["ashyly hiu kuttapi", "pranoy kuttan"]
img_indices = torch.randint(1, 100, (2, 1024)).cuda()

loss, logits = parti(img_indices[:,:-1], texts = texts, labels = img_indices)