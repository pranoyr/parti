


from x_transformers import XTransformer
import torch

from parti.encoder import CLIPTextEmbedder


model = XTransformer(
    dim = 1024,
    enc_num_tokens = 256,
    enc_depth = 8,
    enc_heads = 6,
    enc_max_seq_len = 1000,
    dec_num_tokens = 8192,
    dec_depth = 6,
    dec_heads = 8,
    dec_max_seq_len = 1024,
    tie_token_emb = False      # tie embeddings of encoder and decoder
).cuda()

# src = torch.randint(0, 256, (2, 77))
# src_mask = torch.ones_like(src).bool()


# text_encoder = CLIPTextEmbedder(
# 			arch='ViT-H-14', version='laion2b_s32b_b79k', freeze=True)

src = ['this is a car']
# src = text_encoder(src)
# src_mask = torch.ones(len(src), 77).bool()

# # src_mask = torch.ones(len(src), 77).bool()

tgt = torch.randint(0, 8192, (1, 1024))


print("input")
print(len(src))
print(tgt.shape)

loss = model(src, tgt.cuda()) # (1, 1024, 512)












# model = XTransformer(
#     dim = 1024,
#     enc_num_tokens = 256,
#     enc_depth = 6,
#     enc_heads = 8,
#     enc_max_seq_len = 77,
#     dec_num_tokens = 8192,
#     dec_depth = 6,
#     dec_heads = 8,
#     dec_max_seq_len = 1024,
#     tie_token_emb = False      # tie embeddings of encoder and decoder
# )


# src = torch.randint(0, 256, (1, 77))
# src_mask = torch.ones_like(src).bool()
# print(src_mask.shape)
# tgt = torch.randint(0, 8192, (1, 1024))

# loss = model(src, tgt, mask = src_mask) # (1, 1024, 512)