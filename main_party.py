import os
import torch
import random
import os
import torch
import matplotlib.pyplot as plt
from parti.trainer import Trainer
from config import get_config
from parti.parti import Parti
import torch.multiprocessing as mp
from torch.distributed import init_process_group, destroy_process_group

# add path for demo utils functions 
import numpy as np
import wandb
import argparse
import torch.distributed as dist



def ddp_setup():
    init_process_group(backend="nccl")
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))

def set_seed(seed):
	torch.manual_seed(seed)
	np.random.seed(seed)
	random.seed(seed)
	torch.backends.cudnn.deterministic = True
	torch.backends.cudnn.benchmark = False
	os.environ['PYTHONHASHSEED'] = str(seed)
	

def main(cfg):
	# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
	# torch.set_float32_matmul_precision('high')

	# if cfg.TRAIN.DISTRIBUTED:
	#ddp_setup()
	# seed = cfg.SEED + get_rank()
	# torch.manual_seed(seed)
	# torch.cuda.manual_seed(seed)
	# np.random.seed(seed)
	# random.seed(seed)
	# init wandb
	wandb.init(project=cfg.MODEL.NAME, config=cfg, name=cfg.EXP_NAME)

	# load model
	# if cfg.MODEL.NAME == 'parti':
	model = Parti(cfg)
	# model = torch.compile(model)
	
	trainer = Trainer(cfg, model)
	trainer.fit()
	#destroy_process_group()


if __name__ == '__main__':
	parser = argparse.ArgumentParser(description='CLIFF')
	parser.add_argument('--cfg', type=str, help='config file path')
	args = parser.parse_args()

	# load config
	cfg = get_config(args)
	main(cfg)

