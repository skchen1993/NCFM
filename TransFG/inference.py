from __future__ import absolute_import, division, print_function
import models.configs as configs
import logging
import argparse
import os
import random
import numpy as np
import time
from PIL import Image
from datetime import timedelta
import torch
import pandas as pd
from torch.utils.data import Dataset
from torchvision import transforms
from models.modeling import VisionTransformer, CONFIGS
import csv


CONFIGS = {
    'ViT-B_16': configs.get_b16_config(),
    'ViT-B_32': configs.get_b32_config(),
    'ViT-L_16': configs.get_l16_config(),
    'ViT-L_32': configs.get_l32_config(),
    'ViT-H_14': configs.get_h14_config(),
    'testing': configs.get_testing(),
}

"""
class FineGrainDataset(Dataset):
    def __init__(self, root_dir, annotation_file, transform=None, test=False):
        self.root_dir = root_dir  # 圖片本人路徑
        self.annotations = pd.read_csv(annotation_file)  # 上一步做的CSV
        self.transform = transform  # 定義要做的transform, 含有resize把圖片先resize成依樣
        self.test = test

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, index):
        img_id = self.annotations.iloc[index, 0]  # 取出image id => ex: 0003.jpg
        if self.test == False:
            img = Image.open(os.path.join(self.root_dir, img_id)).convert(
                "RGB")  # 取出image id 對應的圖片本人, 並且轉RGB(等等用transform來轉tensor)
            temp = self.annotations.iloc[index, 1]
            y_label = torch.tensor(self.annotations.iloc[index, 1]).long()
            img = self.transform(img)
            return (img, y_label - 1)
        else:
            # print("fetch testing image id: ", img_id)
            img = Image.open(os.path.join(self.root_dir, img_id)).convert("RGB")
            img = self.transform(img)
            return (img, img_id)
"""

class FineGrainDataset_final(Dataset):
    def __init__(self, root_dir, annotation_file , transform=None, test=False):
        #self.root_dir = root_dir  # "/home/skchen/ML_practice/final/fish"
        self.annotations = pd.read_csv(annotation_file)  
        self.transform = transform  # 定義要做的transform, 含有resize把圖片先resize成依樣
        self.test = test

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, index):
        img_id = self.annotations.iloc[index, 0]  # 絕對路徑
        if self.test == False:
            img = Image.open(os.path.join(self.root_dir, img_id)).convert(
                "RGB")  # 取出image id 對應的圖片本人, 並且轉RGB(等等用transform來轉tensor)
                        # 取出對應的 image label 並且轉成float tensor
            y_label = torch.tensor(self.annotations.iloc[index, 0]).long()
            img = self.transform(img)
            #return (img, y_label - 1)
            # final dataset label 從 0 編, 不會有問題
            return (img, y_label)
        else:
            # img_id 是絕對路徑了
            img = Image.open(img_id).convert("RGB")
            img = self.transform(img)
            basename = os.path.basename(img_id)
            split_list = img_id.split('/')
            if split_list[-2] == "test_stg2":
                name = os.path.join("test_stg2", basename)
            else:
                name = basename
            return img, name


def main():
    #training model parameter setting
    config = CONFIGS["ViT-B_16"]
    config.slide_step = 12
    config.split = 'overlap'
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    #inference parameter setting
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", default=4,
                        help="inference batch size", type=int)
    parser.add_argument("--pretrained_model_path", default="/home/skchen/ML_practice/final/VRDL_1_TrangFG/output/TransFG_lr0.03_tbs16_checkpoint.bin",
                        help="TransFG training model path", type=str)
    parser.add_argument("--output_name", default="submission.csv",
                        help="file name for different setup", type=str)
    args = parser.parse_args()



    #inference data transform
    test_transform = transforms.Compose([transforms.Resize((600, 600), Image.BILINEAR),
                                         transforms.CenterCrop((448, 448)),
                                         transforms.ToTensor(),
                                         transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

    #inference dataset and dataloader prepared
    test_img_path = None
    test_csv_name = "/home/skchen/ML_practice/final/test.csv"
    test_dataset = FineGrainDataset_final(None, test_csv_name, test_transform, test=True)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

    #Pretrained TransFG mdoel prepared
    pretrained_model_path = args.pretrained_model_path
    model = VisionTransformer(config, 448, zero_head=True, num_classes=8,
                              smoothing_value=0.0)
    if pretrained_model_path is not None:
        pretrained_model = torch.load(pretrained_model_path)['model']
        model.load_state_dict(pretrained_model)
    model.to(device)
    print("------model prepared-------")

    model.eval()

    count = 0
    prediction = []
    prediction_range = []
    imgid = []
    with torch.no_grad():
        for i, data in enumerate(test_loader):
            print("count:", count)
            count += 1
            # deal with negative number
            test_pred = torch.exp(model(data[0].cuda()))
            print("test_pred(Non-negative):", test_pred)
            test_label = np.argmax(test_pred.cpu().data.numpy(), axis=1)
            print("test_label: ", test_label)
            print(" data[1]: ", data[1])
            for y in test_label:
                prediction.append(y)
            for x in data[1]:
                imgid.append(x)
            for z in test_pred:
                prediction_range.append(z)
           

    print("length: ", len(imgid), " , ", len(prediction))
    result = args.output_name
    print("output: ", result )
    print("load model: ", args.pretrained_model_path)

    with open(result, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)

        writer.writerow(['image','ALB','BET','DOL','LAG','NoF','OTHER','SHARK','YFT'])
        for i in range(len(imgid)):
            a = prediction_range[i].tolist()
            content = [str(imgid[i])] + a
            writer.writerow(content)
        


if __name__ == "__main__":
    main()