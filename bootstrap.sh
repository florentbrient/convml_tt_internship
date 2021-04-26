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
ulimit -s unlimited
module purge
module load python/3.8-anaconda2020-11
conda init bash
cd /home/rramoul/convml_tt_gpu/
conda activate  convml_tt
pip install .
srun python -m convml_tt.trainer data/LARGE --max-epochs 500 --base-arch resnet34 --log-to-wandb --preload-data --num-dataloader-workers 16
