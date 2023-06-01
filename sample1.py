import torch.nn as nn
import torch
from transformers import T5Tokenizer, T5EncoderModel

# bos_token = torch.tensor([[8193]])

# # append [334] to bos_token
# gen = torch.tensor([[2295]])
# gen =  torch.cat([bos_token, gen])
# print(gen)



# decoder_layer = nn.TransformerDecoderLayer(d_model=512, nhead=8)
# transformer_decoder = nn.TransformerDecoder(decoder_layer, num_layers=6).cuda()


# memory = torch.rand(10, 32, 512).cuda()
# tgt = torch.rand(20, 32, 512).cuda()



# tgt_mask = torch.triu(torch.full((tgt.shape[0], tgt.shape[0]), float('-inf')), diagonal=1).cuda()



# out = transformer_decoder(tgt, memory, tgt_mask=tgt_mask)

# print(out.shape)
# import argparse

# parser = argparse.ArgumentParser(description='CLIFF')
# parser.add_argument('--cfg', type=str, help='config file path')
# args = parser.parse_args()

# # load config
# args.world_size = torch.cuda.device_count()
# print(args.world_size)



# print(2e-3)
# encoder = nn.Embedding(10, 3)

# src = torch.tensor([[1,2,4,5],[4,3,2,0]])
# src = src.view(4,2)
# output = encoder(src)
# print(output.shape)

def get_model(name):
    print(name)
    model = T5EncoderModel.from_pretrained('t5-large')
    return model






# class T5TextEmbedder(nn.Module):
#     def __init__(self, version="google/flan-t5-xl", device="cuda", max_length=77, freeze=True):  
#         super().__init__()
#         self.tokenizer = T5Tokenizer.from_pretrained(version)
#         self.transformer = T5EncoderModel.from_pretrained(version)
#         self.device = device
#         self.max_length = max_length
#         if freeze:
#             self.freeze()

#     def freeze(self):
#         self.transformer = self.transformer.eval()
#         for param in self.parameters():
#             param.requires_grad = False

#     def forward(self, text):
#         batch_encoding = self.tokenizer(text, truncation=True, max_length=self.max_length, return_length=True,
#                                         return_overflowing_tokens=False, padding="max_length", return_tensors="pt")
#         tokens = batch_encoding["input_ids"].to(self.device)
#         outputs = self.transformer(input_ids=tokens)
#         z = outputs.last_hidden_state
#         return z

#     def encode(self, text):
#         return self(text)


# text_embedd = T5TextEmbedder("t5-large")
# # text_embedd = text_embedd.cuda()
# text = ["I am a student"," hi"]
# text = text_embedd.encode(text)
# print(text.shape)

