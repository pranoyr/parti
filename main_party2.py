import parti as pm
from parti.utils import datasets

data_path = '/home/pranoy/datasets/coco2017'
transform = pm.stage2_transform(img_size=256, is_train=True, scale=0.8)
dataset = datasets.CoCo(root=data_path, transform=transform) 
# or your own dataset, the output format should be (image: torch.Tensor, caption: str)

# load pretrained weights I upload to huggingface, not finish yet
# or load your pretrained weights
model = pm.create_pipeline_for_train(version='paintmindv1', stage1_pretrained=True)


trainer = pm.PaintMindTrainer(
    model                       = model,
    dataset                     = dataset,
    num_epoch                   = 40,
    valid_size                  = 1,
    optim                       = 'adamw',
    lr                          = 1e-4,
    lr_min                      = 1e-5,
    warmup_steps                = 10000,
    weight_decay                = 0.05,
    warmup_lr_init              = 1e-6,
    decay_steps                 = 80000,
    batch_size                  = 8,
    num_workers                 = 0,
    pin_memory                  = True,
    grad_accum_steps            = 8,
    mixed_precision             = 'bf16',
    max_grad_norm               = 1.0,
    save_every                  = 5000,
    sample_every                = 5000,
    result_folder               = "your/result/folder",
    log_dir                     = "your/log/dir",
    )
trainer.train()