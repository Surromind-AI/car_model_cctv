# TransFG: A Transformer Architecture for Fine-grained Recognition

[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/transfg-a-transformer-architecture-for-fine/fine-grained-image-classification-on-cub-200)](https://paperswithcode.com/sota/fine-grained-image-classification-on-cub-200?p=transfg-a-transformer-architecture-for-fine) [![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/transfg-a-transformer-architecture-for-fine/fine-grained-image-classification-on-nabirds)](https://paperswithcode.com/sota/fine-grained-image-classification-on-nabirds?p=transfg-a-transformer-architecture-for-fine) [![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/transfg-a-transformer-architecture-for-fine/fine-grained-image-classification-on-stanford-1)](https://paperswithcode.com/sota/fine-grained-image-classification-on-stanford-1?p=transfg-a-transformer-architecture-for-fine) [![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/transfg-a-transformer-architecture-for-fine/image-classification-on-inaturalist)](https://paperswithcode.com/sota/image-classification-on-inaturalist?p=transfg-a-transformer-architecture-for-fine)

Official PyTorch code for the paper:  [*TransFG: A Transformer Architecture for Fine-grained Recognition (AAAI2022)*](https://arxiv.org/abs/2103.07976)  

![](./TransFG.png)

## Dependencies:
+ Python 3.6.9
<br>
## LICENSE
#### MIT License
<br>

## 1. packages 설치 (우분투 기준)

#### 다음 명령어를 사용하여 필수 라이브러리 설치:

```bash
pip3 install -r requirements.txt
```  
#### apex 설치 
1. git clone https://github.com/NVIDIA/apex
2. cd apex
3. pip3 install -v --disable-pip-version-check --no-cache-dir --global-option="--cpp_ext" --global-option="--cuda_ext" ./
4. cd ..
<br>

#### 구글 프리트레인 ViT 모델 다운로드 

wget https://storage.googleapis.com/vit_models/imagenet21k/ViT-B_16.npz 
<br> 

## 2. 데이터 셋 준비
##### 2.1 make_data 폴더에 raw_data 준비(라벨폴더, 이미지폴더)
##### 2.2 make_data 폴더의 make_image_folder.py 파일을 실행하여 datasets/custom 폴더에 폴더(라벨명)/이미지 형태로 이미지 생성
##### 2.3 make_data 폴더의 text_to_df.py 파일을 실행하여 train, val, test 데이터 경로가 담긴 csv 파일 생성  

<br>

## 3. Train

 1 개의 GPU gpu로 학습할 경우:
```bash
CUDA_VISIBLE_DEVICES=0 python3 train.py --dataset custom --split overlap --num_steps 10000 --fp16 --name sample_run

```
<br>

## Citation

If you find our work helpful in your research, please cite it as:

```
@article{he2021transfg,
  title={TransFG: A Transformer Architecture for Fine-grained Recognition},
  author={He, Ju and Chen, Jie-Neng and Liu, Shuai and Kortylewski, Adam and Yang, Cheng and Bai, Yutong and Wang, Changhu and Yuille, Alan},
  journal={arXiv preprint arXiv:2103.07976},
  year={2021}
}
```
  
## Acknowledgement

Many thanks to [ViT-pytorch](https://github.com/jeonsworld/ViT-pytorch) for the PyTorch reimplementation of [An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale](https://arxiv.org/abs/2010.11929)

