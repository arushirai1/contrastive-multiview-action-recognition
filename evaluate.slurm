#!/bin/bash
#SBATCH --nodes=1                      # Get one node
#SBATCH --cpus-per-task=2              # Two cores per task
#SBATCH --ntasks=1                     # But only one task
#SBATCH --mem-per-cpu=16gb
#SBATCH --gres-flags=enforce-binding   # Insist on good CPU/GPU alignment
#SBATCH --gres=gpu:1                   # And two GPU
#SBATCH --time=15:00              # Run for 7 days, at most
#SBATCH --job-name=eval              # Name the job so I can see it in squeue
#SBATCH --output=cross-subject-i3d-%J.out
#SBATCH --mail-type=BEGIN,END,FAIL     # Send me email for various states
#SBATCH --mail-user arai4@hawk.iit.edu        # Use this address

module load anaconda/anaconda3
module load cuda/cuda-11.0
source activate cv-arushi

echo "Cross View - Pretrained"
echo "NTUARD_SUPERVISED_TRAININGresnet3D181616_True_clips_3_gru_False_CS_False"
python eval_script.py --dataset ntuard --arch resnet3D18 --batch-size 32 --num-class 60 --no-clips 3 --modeldir ./results/NTUARD_SUPERVISED_TRAININGresnet3D181616_True_clips_3_gru_False_CS_False --num-workers 16 --frames-path /datasets/NTU-ARD/frames-240x135

echo "NTUARD_SUPERVISED_TRAININGi3d1616_True_clips_3_gru_False_CS_False"
python eval_script.py --dataset ntuard --arch i3d --batch-size 32 --no-clips 3 --num-class 60 --modeldir ./results/NTUARD_SUPERVISED_TRAININGi3d1616_True_clips_3_gru_False_CS_False --num-workers 16 --frames-path /datasets/NTU-ARD/frames-240x135

echo "Cross Subject Results - Pretrained"
echo "NTUARD_SUPERVISED_TRAININGresnet3D181616_True_clips_3_gru_False_CS_True"
python eval_script.py --cross-subject --dataset ntuard --arch resnet3D18 --batch-size 32 --num-class 60 --no-clips 3 --modeldir ./results/NTUARD_SUPERVISED_TRAININGresnet3D181616_True_clips_3_gru_False_CS_True --num-workers 16 --frames-path /datasets/NTU-ARD/frames-240x135

echo "NTUARD_SUPERVISED_TRAININGi3d1616_True_clips_3_gru_False_CS_True"
python eval_script.py --cross-subject --dataset ntuard --arch i3d --batch-size 32 --no-clips 3 --num-class 60 --modeldir ./results/NTUARD_SUPERVISED_TRAININGi3d1616_True_clips_3_gru_False_CS_True --num-workers 16 --frames-path /datasets/NTU-ARD/frames-240x135

echo "Cross View "
echo "NTUARD_SUPERVISED_TRAININGresnet3D181616_False_clips_3_gru_False_CS_False"
python eval_script.py --dataset ntuard --arch resnet3D18 --no-clips 3 --batch-size 32 --num-class 60 --modeldir ./results/NTUARD_SUPERVISED_TRAININGresnet3D181616_False_clips_3_gru_False_CS_False --num-workers 16 --frames-path /datasets/NTU-ARD/frames-240x135

echo "NTUARD_SUPERVISED_TRAININGi3d1616_False_clips_3_gru_False_CS_False"
python eval_script.py --dataset ntuard --arch i3d --batch-size 32 --num-class 60 --no-clips 3 --modeldir ./results/NTUARD_SUPERVISED_TRAININGi3d1616_False_clips_3_gru_False_CS_False --num-workers 16 --frames-path /datasets/NTU-ARD/frames-240x135

echo "Cross Subject Results"
echo "NTUARD_SUPERVISED_TRAININGresnet3D181616_False_clips_3_gru_False_CS_True"
python eval_script.py --cross-subject --dataset ntuard --arch resnet3D18 --batch-size 32 --num-class 60 --no-clips 3 --modeldir ./results/NTUARD_SUPERVISED_TRAININGresnet3D181616_False_clips_3_gru_False_CS_True --num-workers 16 --frames-path /datasets/NTU-ARD/frames-240x135

echo "NTUARD_SUPERVISED_TRAININGi3d1616_False_clips_3_gru_False_CS_True"
python eval_script.py --cross-subject --dataset ntuard --arch i3d --batch-size 32 --no-clips 3 --num-class 60 --modeldir ./results/NTUARD_SUPERVISED_TRAININGi3d1616_False_clips_3_gru_False_CS_True --num-workers 16 --frames-path /datasets/NTU-ARD/frames-240x135
