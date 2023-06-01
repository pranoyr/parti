import torch
import transformers
from transformers import T5Tokenizer, T5EncoderModel, T5Config

transformers.logging.set_verbosity_error()

def exists(val):
	return val is not None

# config

MAX_LENGTH = 256

DEFAULT_T5_NAME = 'google/t5-v1_1-base'

T5_CONFIGS = {}

# singleton globals

def get_tokenizer(name):
	tokenizer = T5Tokenizer.from_pretrained(name)
	return tokenizer

def get_model(name):
	print(name)
	model = T5EncoderModel.from_pretrained(name)
	return model

def get_model_and_tokenizer(name):
	global T5_CONFIGS

	if name not in T5_CONFIGS:
		T5_CONFIGS[name] = dict()
	if "model" not in T5_CONFIGS[name]:
		T5_CONFIGS[name]["model"] = get_model(name)
	if "tokenizer" not in T5_CONFIGS[name]:
		T5_CONFIGS[name]["tokenizer"] = get_tokenizer(name)


	return T5_CONFIGS[name]['model'], T5_CONFIGS[name]['tokenizer']

def get_encoded_dim(name):
	if name not in T5_CONFIGS:
		# avoids loading the model if we only want to get the dim
		config = T5Config.from_pretrained(name)
		T5_CONFIGS[name] = dict(config=config)
	elif "config" in T5_CONFIGS[name]:
		config = T5_CONFIGS[name]["config"]
	elif "model" in T5_CONFIGS[name]:
		config = T5_CONFIGS[name]["model"].config
	else:
		assert False
	return config.d_model

# encoding text

class TextEncoder(torch.nn.Module):
	def __init__(self, name = DEFAULT_T5_NAME):
		super().__init__()
		self.name = name
	   
		self.t5, self.tokenizer = get_model_and_tokenizer(name)
		self.t5.eval()
	
	def forward(self, texts, output_device =None):
	 
		# if torch.cuda.is_available():
		#     t5 = t5.cuda()

		encoded = self.tokenizer.batch_encode_plus(
			texts,
			return_tensors = "pt",
			padding = 'max_length',
			max_length = MAX_LENGTH,
			truncation = True
		)

		input_ids = encoded.input_ids
		attn_mask = encoded.attention_mask


		with torch.no_grad():
			output = self.t5(input_ids = input_ids, attention_mask = attn_mask)
			encoded_text = output.last_hidden_state.detach()

		attn_mask = attn_mask.bool()

		if not exists(output_device):
			return encoded_text, attn_mask

		encoded_text.to(output_device)
		attn_mask.to(output_device)

		return encoded_text, attn_mask
