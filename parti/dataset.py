import torchvision
import numpy as np    
import deeplake    
from torchvision import transforms, models
from pycocotools.coco import COCO
import os
from .utils.transform import stage1_transform
from PIL import Image
import torch
from parti.parti import Parti

# ['A man riding on the back of a motorcycle]


class CoCo:
    def __init__(self, root, dataType='train2017', annType='captions', transform=None):
        self.root = root
        self.img_dir = '{}/{}'.format(root, dataType)
        annFile = '{}/annotations/{}_{}.json'.format(root, annType, dataType)
        self.coco = COCO(annFile)
        self.imgids = self.coco.getImgIds()
        self.transform = transform
        #self.imgids = self.imgids[:1000]
    
    def __getitem__(self, idx):
        imgid = self.imgids[idx]
        img_name = self.coco.loadImgs(imgid)[0]['file_name']
        annid = self.coco.getAnnIds(imgIds=imgid)
        img = Image.open(os.path.join(self.img_dir, img_name)).convert('RGB')
        ann = np.random.choice(self.coco.loadAnns(annid))['caption']



        img = stage1_transform(is_train=False, scale=0.8)(img)

        
        # if self.transform is not None:
        #     img = self.transform(img)
        # data = {'img': img, 'text': ann}
        # print(ann)
        
        return img, ann    
        
    def __len__(self):
        return len(self.imgids)
    


# coco = CoCo(root='/home/pranoy/datasets/coco2017', dataType='train2017')
# parti = Parti().cuda()
# loader = torch.utils.data.DataLoader(coco, batch_size=2, shuffle=True, num_workers=0)

# for i, (img, ann) in enumerate(loader):
#     loss = parti(img, ann)
#     print(loss)