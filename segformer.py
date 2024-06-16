from transformers import SegformerImageProcessor, AutoModelForSemanticSegmentation
from PIL import Image
import requests
import matplotlib.pyplot as plt
import torch.nn as nn
from torchinfo import summary
import torch
import numpy as np
import torch.optim as optim
import torchvision.transforms as T
import torch.nn.functional as F
from collections import defaultdict
import cv2
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import pandas as pd
device = torch.device("cuda:4" if torch.cuda.is_available() else "cpu")
tf=T.ToTensor()
params={'image_size':512,
        'lr':2e-4,
        'beta1':0.5,
        'beta2':0.999,
        'batch_size':8,
        'epochs':100,}
image1=np.load('../../data/cv0_ori.npy')
image1=image1.astype(np.uint8)
image2=np.load('../../data/cv1_ori.npy')
image2=image2.astype(np.uint8)
image3=np.load('../../data/cv2_ori.npy')
image3=image3.astype(np.uint8)
image4=np.load('../../data/cv3_ori.npy')
image4=image4.astype(np.uint8)
image5=np.load('../../data/cv4_ori.npy')
image5=image5.astype(np.uint8)
mask1=np.load('../../data/cv0_mask.npy')
mask1=(mask1*255).astype(np.uint8)
mask2=np.load('../../data/cv1_mask.npy')
mask2=(mask2*255).astype(np.uint8)
mask3=np.load('../../data/cv2_mask.npy')
mask3=(mask3*255).astype(np.uint8)
mask4=np.load('../../data/cv3_mask.npy')
mask4=(mask4*255).astype(np.uint8)
mask5=np.load('../../data/cv4_mask.npy')
mask5=(mask5*255).astype(np.uint8)

np_data={'image1':image1,'image2':image2,'image3':image3,'image4':image4,'image5':image5,'mask1':mask1,'mask2':mask2,'mask3':mask3,'mask4':mask4,'mask5':mask5}

class CustomDataset(Dataset):
    def __init__(self, image_list, label_list):
        self.img_path = image_list
        self.label = label_list

    def __len__(self):
        return len(self.img_path)

    def __getitem__(self, idx):
        image_path = self.img_path[idx]
        image_path=tf(cv2.cvtColor(image_path, cv2.COLOR_GRAY2RGB))
        
        label_path = self.label[idx]
        label_path = tf(cv2.resize(label_path, (512, 512)))
       
        return image_path, label_path
    
def dice_loss(pred, target, num_classes=4):
    smooth = 1.
    dice_per_class = torch.zeros(num_classes).to(pred.device)

    for class_id in range(num_classes):
        pred_class = pred[:, class_id, ...]
        target_class = target[:, class_id, ...]

        intersection = torch.sum(pred_class * target_class)
        A_sum = torch.sum(pred_class * pred_class)
        B_sum = torch.sum(target_class * target_class)

        dice_per_class[class_id] = 1 - \
            (2. * intersection + smooth) / (A_sum + B_sum + smooth)

    return torch.mean(dice_per_class)


metrics = defaultdict(float)
for k in range(5):
    val_loss=1000
    df=pd.DataFrame(columns=['epoch', 'train_loss', 'val_loss', 'train_acc', 'val_acc'])
    train_list=[0,1,2,3,4]
    train_list.remove(k)
    train_image=np.concatenate([np_data['image'+str(i+1)] for i in train_list])
    train_mask=np.concatenate([np_data['mask'+str(i+1)] for i in train_list])
    val_image=np_data['image'+str(k+1)]
    val_mask=np_data['mask'+str(k+1)]
    train_dataset = CustomDataset(image1, mask1)

    val_dataset = CustomDataset(image1, mask1)
    train_dataloader = DataLoader(
    train_dataset, batch_size=params['batch_size'], shuffle=True, drop_last=True)
    validation_dataloader = DataLoader(
    val_dataset, batch_size=params['batch_size'], shuffle=True, drop_last=True)
    
    model = AutoModelForSemanticSegmentation.from_pretrained("mattmdjaga/segformer_b2_clothes",num_labels=4,ignore_mismatched_sizes=True).to(device)
    optimizer = optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()), lr=params['lr'], betas=(params['beta1'], params['beta2']))
    for epoch in range(100):
        train = tqdm(train_dataloader)
        count = 0
        running_loss = 0.0
        acc_loss = 0
        for x, y in train:
            model.train()
            y = y.to(device).float()
            count += 1
            x = x.to(device).float()
            optimizer.zero_grad()  # optimizer zero 로 초기화
            predict = model(x).logits.to(device)
            predict = nn.functional.interpolate(
                predict,
                size=(512,512),
                mode="bilinear",
                align_corners=False,
            )
            cost = dice_loss(predict, y)  # cost 구함
            acc = 1-cost.item()
            cost.backward()  # cost에 대한 backward 구함
            optimizer.step()
            running_loss += cost.item()
            acc_loss += acc
            train.set_description(
                f"epoch: {epoch+1}/{300} Step: {count+1} dice_loss : {running_loss/count:.4f} dice_score: {1-running_loss/count:.4f}")
            
        with torch.no_grad():
            val = tqdm(validation_dataloader)
            count = 0
            val_running_loss = 0.0
            acc_loss = 0
            for x, y in val:
                model.eval()
                y = y.to(device).float()
                count += 1
                x = x.to(device).float()
                predict = model(x).logits.to(device)
                predict = nn.functional.interpolate(
                    predict,
                    size=(512,512),
                    mode="bilinear",
                    align_corners=False,
                )
                cost = dice_loss(predict, y)  # cost 구함
                acc = 1-cost.item()
                val_running_loss += cost.item()
                acc_loss += acc
                y = y.cpu()
                count += 1
                x = x.cpu()
                val.set_description(
                    f"val_epoch: {epoch+1}/{300} Step: {count+1} dice_loss : {val_running_loss/count:.4f} dice_score: {1-val_running_loss/count:.4f}")
                if val_loss>val_running_loss/count:
                    val_loss=val_running_loss/count
                    torch.save(model.state_dict(), '../../model/segformer/seg_former_'+str(k+1)+'_check.pth')
        df.loc[len(df)]=[epoch+1,running_loss/len(train_dataloader),val_running_loss/len(validation_dataloader),1-running_loss/len(train_dataloader),1-val_running_loss/len(validation_dataloader)]
        df.to_csv('../../model/segformer/seg_former_'+str(k+1)+'.csv',index=False)
    torch.save(model.state_dict(), '../../model/segformer/seg_former_'+str(k+1)+'.pth')