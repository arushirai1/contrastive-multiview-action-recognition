#!/bin/bash
#SBATCH --nodes=1                      # Get one node
#SBATCH --cpus-per-task=8              # Two cores per task
#SBATCH --ntasks=1                     # But only one task
#SBATCH --mem-per-gpu=32gb
#SBATCH --constraint=gpu32
#SBATCH --gres-flags=enforce-binding   # Insist on good CPU/GPU alignment
#SBATCH --gres=gpu:1                   # And two GPU
#SBATCH --time=24:00:00              # Run for 1 hours, at most
#SBATCH --job-name=contrastive-multiview-action              # Name the job so I can see it in squeue
#SBATCH --output=cross-subject-i3d-%J.out
#SBATCH --mail-type=BEGIN,END,FAIL     # Send me email for various states
#SBATCH --mail-user arai4@hawk.iit.edu        # Use this address

module load anaconda/anaconda3
module load cuda/cuda-11.0
source activate cv-arushi

echo "No Augmentation Contrastive Learning"
#python train-test.py --dataset ntuard --no-clips 3 --arch contrastive --cross-subject --resume ./results/NTUARD_CONTRASTIVEresnet3D181616_False_clips_3_lr_0.03/model_best.pth.tar --endpoint B --finetune --epochs 75 --batch-size 16 --lr 0.03 --seed 1 --num-class 60 --frames-path /datasets/NTU-ARD/frames-240x135 --num-workers 16
python contrastive_train.py --dataset ntuard --no-clips 3 --backbone resnet3D18 --epochs 100 --batch-size 64 --lr 0.03 --seed 1 --frames-path /datasets/NTU-ARD/frames-240x135 --num-workers 32
