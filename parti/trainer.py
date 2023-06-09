import torch
import wandb
from parti.utils.meter import AverageMeter, ProgressMeter
import math
from .utils.optim import Lion
from .dataset import CoCo
from timm.scheduler.cosine_lr import CosineLRScheduler
# from timm.optim import Adafactor

from transformers.optimization import Adafactor, AdafactorSchedule

from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
import os
from transformers import get_cosine_schedule_with_warmup




class Trainer():
    """ Trainer class for training and validation"""

    def __init__(self, cfg, model):
        super().__init__()
        self.cfg = cfg
        self.gpu_id = int(os.environ["LOCAL_RANK"])
        self.global_step = 0
        self.th = math.inf

        self.model = model
        self.prepare_data()
        self.get_training_config()

        self.model = model.to(self.gpu_id)
        self.resume_training()
        self.model = DDP(self.model, device_ids=[self.gpu_id], find_unused_parameters=True)
    
        
    def init_meter(self):
        """Init meter for training and validation"""

        self.losses = AverageMeter(f'GPU[{self.gpu_id}], Loss', ':.4f')
        self.progress = ProgressMeter(
            len(self.train_loader) * self.cfg.TRAIN.EPOCHS,
            [self.losses])

    def prepare_data(self):
        """Prepare data for training and validation"""

        train_dataset = CoCo(self.cfg.DATA.TRAIN_PATH)
        val_dataset = CoCo(self.cfg.DATA.VAL_PATH)
        self.train_loader = torch.utils.data.DataLoader(
            train_dataset, batch_size=self.cfg.DATA.BATCH_SIZE, shuffle=False, num_workers=self.cfg.DATA.NUM_WORKERS, sampler=DistributedSampler(train_dataset), pin_memory=True)
        self.val_loader = torch.utils.data.DataLoader(
            val_dataset, batch_size=self.cfg.DATA.BATCH_SIZE, shuffle=False, num_workers=self.cfg.DATA.NUM_WORKERS, sampler=DistributedSampler(val_dataset), pin_memory=True)
        print("train dataset size: ", len(train_dataset))
        print("val dataset size: ", len(val_dataset))
        print("Total Iterations: ", len(
            self.train_loader) * self.cfg.TRAIN.EPOCHS)

        self.init_meter()

    def get_training_config(self):
        """Get training config"""
        # optimizer
        if self.cfg.TRAIN.OPTIMIZER.NAME == 'adam':
            self.optimizer = torch.optim.Adam(self.model.parameters(
            ), lr=self.cfg.TRAIN.BASE_LR, weight_decay=self.cfg.TRAIN.WEIGHT_DECAY)
        elif self.cfg.TRAIN.OPTIMIZER.NAME == 'adamr':
            self.optimizer = torch.optim.AdamR(self.model.parameters(
            ), lr=self.cfg.TRAIN.BASE_LR, weight_decay=self.cfg.TRAIN.WEIGHT_DECAY)
        elif self.cfg.TRAIN.OPTIMIZER.NAME == 'sgd':
            self.optimizer = torch.optim.SGD(self.model.parameters(
            ), lr=self.cfg.TRAIN.BASE_LR, weight_decay=self.cfg.TRAIN.WEIGHT_DECAY)
        elif self.cfg.TRAIN.OPTIMIZER.NAME == 'adamw':
            self.optimizer = torch.optim.AdamW(self.model.parameters(
            ), lr=self.cfg.TRAIN.BASE_LR, weight_decay=self.cfg.TRAIN.WEIGHT_DECAY)
        elif self.cfg.TRAIN.OPTIMIZER.NAME == 'lion':
            self.optimizer = Lion([p for p in self.model.parameters() if p.requires_grad],
                                  lr=self.cfg.TRAIN.BASE_LR, weight_decay=self.cfg.TRAIN.WEIGHT_DECAY)
        elif self.cfg.TRAIN.OPTIMIZER.NAME == 'adafactor':
            # self.optimizer = Adafactor([p for p in self.model.parameters() if p.requires_grad],
            #                       lr=self.cfg.TRAIN.BASE_LR, weight_decay=self.cfg.TRAIN.WEIGHT_DECAY, betas=(0.9, 0.96))
            self.optimizer = Adafactor(self.model.parameters(), scale_parameter=True, relative_step=True, warmup_init=True, lr=None)

        # scheduler
        if self.cfg.TRAIN.LR_SCHEDULER.NAME == 'multistep':
            self.scheduler = torch.optim.lr_scheduler.MultiStepLR(
                self.optimizer, milestones=self.cfg.TRAIN.LR_SCHEDULER.MULTISTEPS, gamma=self.cfg.TRAIN.LR_SCHEDULER.GAMMA)
        elif self.cfg.TRAIN.LR_SCHEDULER.NAME == 'linear':
            self.scheduler = torch.optim.lr_scheduler.LambdaLR(
                self.optimizer, lr_lambda=lambda epoch: 1 - epoch / self.cfg.TRAIN.EPOCHS)
        elif self.cfg.TRAIN.LR_SCHEDULER.NAME == 'cosine':
            # self.scheduler = CosineLRScheduler(self.optimizer, t_initial=self.cfg.TRAIN.LR_SCHEDULER.T_INIT, lr_min=self.cfg.TRAIN.LR_SCHEDULER.LR_MIN,
            #                                    warmup_t=self.cfg.TRAIN.LR_SCHEDULER.WARMUP_T, warmup_lr_init=self.cfg.TRAIN.LR_SCHEDULER.WARMUP_LR_INIT,
            #                                    cycle_limit=1, t_in_epochs=False, warmup_prefix=True)
            self.scheduler = get_cosine_schedule_with_warmup(self.optimizer, num_warmup_steps=self.cfg.TRAIN.LR_SCHEDULER.WARMUP_T, num_training_steps=self.cfg.TRAIN.EPOCHS*len(self.train_loader))
            
        elif self.cfg.TRAIN.LR_SCHEDULER.NAME == 'onecycle':
            self.scheduler = torch.optim.lr_scheduler.OneCycleLR(self.optimizer, max_lr=self.cfg.TRAIN.LR_SCHEDULER.MAX_LR, steps_per_epoch=len(self.train_loader), \
                                                                 epochs=self.cfg.TRAIN.EPOCHS)
        elif self.cfg.TRAIN.LR_SCHEDULER.NAME == 'adafactor_schedule':
            self.scheduler = AdafactorSchedule(self.optimizer)
        elif self.cfg.TRAIN.LR_SCHEDULER.NAME == 'rangetest':
            start_lr=1e-6
            end_lr=0.1
            lr_find_epochs=self.cfg.TRAIN.EPOCHS
            lr_lambda=lambda x: math.exp(
                x * math.log(end_lr / start_lr) / (lr_find_epochs * len(self.train_loader) - 1))
            self.scheduler=torch.optim.lr_scheduler.LambdaLR(
                self.optimizer, lr_lambda)


    def resume_training(self):
        """Resume training from checkpoint"""

        if self.cfg.MODEL.RESUME:
            checkpoint=torch.load(self.cfg.MODEL.RESUME)
            self.model.load_state_dict(checkpoint['state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer'])
            self.scheduler.load_state_dict(checkpoint['scheduler'])
            self.global_step=checkpoint['step']
            print(
                f"==> Loaded checkpoint '{self.cfg.MODEL.RESUME}' (iteration {self.global_step})")
        elif self.cfg.MODEL.PRETRAINED:
            state_dict=torch.load(self.cfg.MODEL.PRETRAINED)['model']
            self.model.load_state_dict(state_dict, strict=False)
            print(
                f"==> Loaded pretrained weights for backbone '{self.cfg.MODEL.PRETRAINED}'")

    def save_checkpoint(self, step, filename, is_best=False):
        """Save checkpoint"""

        if is_best:
            checkpoint={
                'step': step,
                'state_dict': self.model.module.state_dict(),
                'optimizer': self.optimizer.state_dict(),
                'scheduler': self.scheduler.state_dict()
            }
            torch.save(checkpoint, filename)

    def fit(self):
        """Train the model"""

        self.model.train()
        start_epoch=self.global_step//len(self.train_loader)
        for epoch in range(start_epoch, self.cfg.TRAIN.EPOCHS):
            self.train_loader.sampler.set_epoch(epoch)
            for idx, (img, text) in enumerate(self.train_loader):
                img=img.to(self.gpu_id)
                loss=self.model(text, img)
                self.losses.update(loss.item(), img.size(0))

                self.optimizer.zero_grad()
                loss.backward()
                # torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                self.optimizer.step()
                self.scheduler.step()

                # torch.cuda.synchronize()

                # LOGGING
                if self.global_step % self.cfg.PRINT_FREQ == 0:
                    self.progress.display(self.global_step)

                if self.global_step % self.cfg.TRAIN_FREQ == 0:
                    wandb.log({'train_loss': loss.item()},
                              step=self.global_step)
                    lr=self.optimizer.param_groups[0]['lr']
                    wandb.log({'lr': lr}, step=self.global_step)

            # 	if self.global_step % self.cfg.VALID_FREQ == 0:
            # 		self.model.eval()
            # 		val_loss = self.validate()
            # 		self.visualise()
            # 		self.model.train()

            # 		lr = self.optimizer.param_groups[0]['lr']
            # 		wandb.log({'train_loss': self.losses.avg}, step=self.global_step)
            # 		wandb.log({'val_loss': val_loss}, step=self.global_step)
            # 		wandb.log({'lr': lr}, step=self.global_step)

            # 		is_best = val_loss < self.th
            # 		self.th = min(val_loss, self.th)
            # 		checkpoint_path = self.cfg.CKPT_DIR + \
            # 			f"/best_weights_{self.cfg.EXP_NAME}.pth"
            # 		self.save_checkpoint(
            # 			self.global_step,  checkpoint_path, is_best=is_best)

                if self.global_step % self.cfg.SAVE_FREQ == 0:
                    checkpoint_name=self.cfg.CKPT_DIR + \
                        f"/checkpoint_iter{self.global_step}_{self.cfg.EXP_NAME}.pth"
                    self.save_checkpoint(
                        self.global_step, checkpoint_name,  is_best=True)
                    print("model saved to {}".format(checkpoint_name))
                    print()
                self.global_step += 1

            print("Epoch " + str(epoch) + " completed")
            print()

        # Save the final model
        checkpoint_name=self.cfg.CKPT_DIR + \
            f"/final_weights_{self.cfg.EXP_NAME}.pth"
        self.save_checkpoint(self.global_step, checkpoint_name, is_best=True)
