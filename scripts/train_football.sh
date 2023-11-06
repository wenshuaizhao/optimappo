#!/bin/sh
#SBATCH --job-name=football
#SBATCH --account=project_462000215
#SBATCH --partition=standard-g
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=52
#SBATCH --mem=60G
#SBATCH --time=48:00:00
#SBATCH --gres=gpu:1
#SBATCH --output=../results/lumi_output/job_%A_%a.out
#SBATCH --error=../results/lumi_output/array_job_err_%A_%a.txt
SLURM_CPUS_PER_TASK=52


srun --cpus-per-task=$SLURM_CPUS_PER_TASK singularity run --cleanenv \
--rocm -B /scratch/project_462000215/mappo:/users/wenshuai/projects/mappo \
--env PYTHONPATH=/users/wenshuai/projects/mappo/prj_env/lib/python3.8/site-packages \
/scratch/project_462000215/docker/gfootball_latest.sif \
/bin/sh /users/wenshuai/projects/mappo/scripts/train_football_scripts/train_football_corner.sh

# /bin/sh cd /users/wenshuai/projects/mappo/scripts; \
# ./train_football_scripts/train_football_corner.sh
