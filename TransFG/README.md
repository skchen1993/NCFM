# TransFG

## Installation

* The following packages are required to run the scripts:
  - [Python >= 3.6]
  - [PyTorch = 1.5]
  - [Torchvision = 0.6.1]
  - [ml_collections]

* Install other needed packages
```
pip install -r requirements.txt
```

## Prepare data
```
cd data
sh prepare_dataset.sh
```
* Three csv files will be generated:
1. train.csv
2. val.csv
3. test.csv

## Download Google pre-trained ViT models

* [Get models in this link](https://console.cloud.google.com/storage/vit_models/): ViT-B_16, ViT-B_32...
```bash
wget https://storage.googleapis.com/vit_models/imagenet21k/{MODEL_NAME}.npz
```

## Train
```
CUDA_VISIBLE_DEVICES=2,3,4,6 python3 -m torch.distributed.launch --nproc_per_node=4 train.py --dataset final --split overlap --num_steps 10000  --name TransFG_lr0.015_tbs8 --train_csv_name {train.csv in ./data} --test_csv_name {val.csv in ./data} 
```


## Inference
* [Pretrained TransFG model](https://drive.google.com/file/d/1B03DSv1eGXNyAySEdcqcpoboakF-V7y9/view?usp=sharing)
* Put it under `output/`

```
CUDA_VISIBLE_DEVICES=9 python inference.py --pretrained_model_path {pretrained TransFG model} --output_name {output csv name for submission}
```


