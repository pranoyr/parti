import torchvision
import numpy as np    
import deeplake    
from torchvision import transforms, models
from pycocotools.coco import COCO
import cv2
import os
from parti.utils.transform import stage1_transform
from PIL import Image
# import torch
# # from parti.parti import Parti

# print(1e-4)

import torch

# # create a 2d empty tensor
# gen_seq = torch.empty(1, dtype=torch.long)

# a = torch.tensor([2995])

# print(torch.cat((gen_seq, a), dim=0))


# import torch
# src = torch.randint(0, 256, (1, 1024))
# src_mask = torch.ones_like(src).bool()
# print(src_mask.shape)\
# def name2(c=None, d=None):
#     print(c)
#     print(d)



# def name(a=None,b=None, **kwargs):
#     print(a)
#     print(b)
#     # check if g is there in kwargs
#     if kwargs.get('g') is  None:
#         print("yes")
#     # name2(**kwargs)
   

# import torch

# tgt = torch.randint(0, 8192, (1, 1024))
# print(tgt.shape)

# name(a=1,b=1,c=2, d=2, g= 4)

# def exists(val):
#     return val is not None

# import torch
# context = torch.randn(1, 1, 1024)
# if context is None:
#     print("yes")


import cv2

class CoCo:
    def __init__(self, root, dataType='train2017', annType='captions', transform=None):
        self.root = root
        self.img_dir = '{}/{}'.format(root, dataType)
        annFile = '{}/annotations/{}_{}.json'.format(root, annType, dataType)
        self.coco = COCO(annFile)
        self.imgids = self.coco.getImgIds()
        # take on 1 sample
        self.imgids = self.imgids[:2]
        print(self.imgids)
        self.transform = transform
    
    def __getitem__(self, idx):
        imgid = self.imgids[idx]
        img_name = self.coco.loadImgs(imgid)[0]['file_name']
        annid = self.coco.getAnnIds(imgIds=imgid)
        img = Image.open(os.path.join(self.img_dir, img_name)).convert('RGB')
        ann = np.random.choice(self.coco.loadAnns(annid))['caption']

        img = stage1_transform(is_train=False, scale=0.8)(img)
        
        # if self.transform is not None:
        #     img = self.transform(img)
        data = {'img': img, 'text': ann}
        
        return data    
        
    def __len__(self):
        return len(self.imgids)
    


coco = CoCo(root='/home/pranoy/datasets/coco2017', dataType='train2017')
# parti = Parti().cuda()
loader = torch.utils.data.DataLoader(coco, batch_size=1, shuffle=True, num_workers=0)

while True :
    for i, (data) in enumerate(loader):
        img = data['img']
        print(data['text'])
        # covnert to cnumpy 
        img = img[0].permute(1, 2, 0).cpu().numpy()
        cv2.imshow("image", img)
        cv2.waitKey(0)
        print(data['text'])
    print ()
    # loss = parti(data)
    # print(loss)