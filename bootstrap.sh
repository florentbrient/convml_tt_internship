#!/bin/sh
#SBATCH --mail-user=raysamram@gmail.com
#SBATCH --mail-type=all
#SBATCH --job-name=GPU-Embedding-Learning
#SBATCH --output=/home/rramoul/convml_tt_gpu/Embedding.out
#SBATCH --error=/home/rramoul/convml_tt_gpu/Embedding.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --gres-flags=enforce-binding

python -m convml_tt.trainer data/ZOONIVERSE --max-epochs 50 --base-arch resnet18 --log-to-wandb --anti-aliased-backbone --batch-size 4 --lr 1e-04 --gpus 1