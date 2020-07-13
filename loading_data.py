import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import torch
import random
from dataset import listDataset


#TRAIN_SIZE = [768, 1024]
TRAIN_SIZE = [576, 768]

def get_min_size(batch):

    min_ht = TRAIN_SIZE[0]
    min_wd = TRAIN_SIZE[1]

    for i_sample in batch:
        
        _,ht,wd = i_sample.shape
        if ht<min_ht:
            min_ht = ht
        if wd<min_wd:
            min_wd = wd
    return min_ht//8*8,min_wd//8*8

def random_crop(img,den,dst_size):
    # dst_size: ht, wd

    _,ts_hd,ts_wd = img.shape

    x1 = random.randint(0, ts_wd - dst_size[1])
    y1 = random.randint(0, ts_hd - dst_size[0])
    x2 = x1 + dst_size[1]
    y2 = y1 + dst_size[0]

    label_x1 = x1//8
    label_y1 = y1//8
    label_x2 = x2//8
    label_y2 = y2//8

    return img[:,y1:y2,x1:x2], den[label_y1:label_y2,label_x1:label_x2]

def share_memory(batch):
    out = None
    if False:
        # If we're in a background process, concatenate directly into a
        # shared memory tensor to avoid an extra copy
        numel = sum([x.numel() for x in batch])
        storage = batch[0].storage()._new_shared(numel)
        out = batch[0].new(storage)
    return out

def SHHA_collate(batch):
    # @GJY 
    r"""Puts each data field into a tensor with outer dimension batch size"""

    transposed = list(zip(*batch)) # imgs and dens
    imgs, dens = [transposed[0],transposed[1]]


    error_msg = "batch must contain tensors; found {}"
    if isinstance(imgs[0], torch.Tensor) and isinstance(dens[0], torch.Tensor):
        
        min_ht, min_wd = get_min_size(imgs)

        # print min_ht, min_wd

        # pdb.set_trace()
        
        cropped_imgs = []
        cropped_dens = []
        for i_sample in range(len(batch)):
            _img, _den = random_crop(imgs[i_sample],dens[i_sample],[min_ht,min_wd])
            cropped_imgs.append(_img)
            cropped_dens.append(_den)


        cropped_imgs = torch.stack(cropped_imgs, 0, out=share_memory(cropped_imgs))
        cropped_dens = torch.stack(cropped_dens, 0, out=share_memory(cropped_dens))

        return [cropped_imgs,cropped_dens]

    raise TypeError((error_msg.format(type(batch[0]))))

def loading_train_data(train_list, batch_size=1):

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                            std=[0.229, 0.224, 0.225])
        ])

    train_loader = DataLoader(listDataset(
        train_list,
        transform=transform,
        train=True,
        num_workers=4,
        ),
        batch_size=batch_size,
        collate_fn=SHHA_collate,
        shuffle=True)

    return train_loader
