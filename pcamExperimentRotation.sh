#python train.py --dataset=BACH --mode=pretrain --ss_task=rotation



#python train.py --dataset=BACH --mode=supervised --init_cond=random
#python train.py --dataset=BACH --mode=test --init_cond=random

python train.py --dataset=pcam --mode=supervised --init_cond=rotation
python train.py --dataset=pcam --mode=test --init_cond=rotation


#python train.py --dataset=BACH --mode=supervised --init_cond=imagenet
#python train.py --dataset=BACH --mode=test --init_cond=rotation



