import torch
from axial_positional_embedding import AxialPositionalEmbedding

pos_emb = AxialPositionalEmbedding(
    dim = 1024,
    axial_shape = (32, 32),          # axial shape will multiply up to the maximum sequence length allowed (64 * 64 = 4096)
    axial_dims = (500, 512)          # if not specified, dimensions will default to 'dim' for all axials and summed at the end. if specified, each axial will have the specified dimension and be concatted together. the concatted dimensions needs to sum up to the `dim` (256 + 256 = 512)
)

tokens = torch.randn(1, 1024, 1024)  # assume are tokens
tokens = pos_emb(tokens) + tokens   