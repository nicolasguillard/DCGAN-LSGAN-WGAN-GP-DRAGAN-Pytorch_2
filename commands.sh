DATA_PATH=${1:-"."}
DATA_PATH="--data_path=$DATA_PATH"

# Fashion-MNIST
#CUDA_VISIBLE_DEVICES=0 python train.py --dataset=fashion_mnist --epoch=25 --adversarial_loss_mode=gan $DATA_PATH
#CUDA_VISIBLE_DEVICES=0 python train.py --dataset=fashion_mnist --epoch=25 --adversarial_loss_mode=gan --gradient_penalty_mode=1-gp --gradient_penalty_sample_mode=dragan $DATA_PATH
#CUDA_VISIBLE_DEVICES=0 python train.py --dataset=fashion_mnist --epoch=25 --adversarial_loss_mode=lsgan $DATA_PATH
#CUDA_VISIBLE_DEVICES=0 python train.py --dataset=fashion_mnist --epoch=50 --adversarial_loss_mode=wgan --gradient_penalty_mode=1-gp --gradient_penalty_sample_mode=line --n_d=5 $DATA_PATH

# CelebA
CUDA_VISIBLE_DEVICES=0 python train.py --dataset=celeba --epoch=25 --adversarial_loss_mode=gan $DATA_PATH
CUDA_VISIBLE_DEVICES=0 python train.py --dataset=celeba --epoch=25 --adversarial_loss_mode=gan --gradient_penalty_mode=1-gp --gradient_penalty_sample_mode=dragan $DATA_PATH
CUDA_VISIBLE_DEVICES=0 python train.py --dataset=celeba --epoch=25 --adversarial_loss_mode=gan --gradient_penalty_mode=lp --gradient_penalty_sample_mode=dragan $DATA_PATH
CUDA_VISIBLE_DEVICES=0 python train.py --dataset=celeba --epoch=25 --adversarial_loss_mode=lsgan $DATA_PATH
CUDA_VISIBLE_DEVICES=0 python train.py --dataset=celeba --epoch=50 --adversarial_loss_mode=wgan --gradient_penalty_mode=1-gp --gradient_penalty_sample_mode=line --n_d=5 $DATA_PATH
CUDA_VISIBLE_DEVICES=0 python train.py --dataset=celeba --epoch=50 --adversarial_loss_mode=wgan --gradient_penalty_mode=lp --gradient_penalty_sample_mode=line --n_d=5 $DATA_PATH

# Anime
CUDA_VISIBLE_DEVICES=0 python train.py --dataset=anime --epoch=100 --adversarial_loss_mode=gan --gradient_penalty_mode=1-gp --gradient_penalty_sample_mode=dragan $DATA_PATH
CUDA_VISIBLE_DEVICES=0 python train.py --dataset=anime --epoch=100 --adversarial_loss_mode=gan --gradient_penalty_mode=lp --gradient_penalty_sample_mode=dragan $DATA_PATH
CUDA_VISIBLE_DEVICES=0 python train.py --dataset=anime --epoch=200 --adversarial_loss_mode=wgan --gradient_penalty_mode=1-gp --gradient_penalty_sample_mode=line --n_d=5 $DATA_PATH
CUDA_VISIBLE_DEVICES=0 python train.py --dataset=anime --epoch=200 --adversarial_loss_mode=wgan --gradient_penalty_mode=lp --gradient_penalty_sample_mode=line --n_d=5 $DATA_PATH