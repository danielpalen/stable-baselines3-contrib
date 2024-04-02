#!/bin/bash
#SBATCH -J SB3-XQ
#SBATCH -a 1-5
#SBATCH -n 1
#SBATCH -c 2
#SBATCH --mem-per-cpu=7000
#SBATCH -t 72:00:00
#SBATCH -p gpu
#SBATCH --gres=gpu:1
#SBATCH -C 'rtx3090|a5000'
#SBATCH -o /home/palenicek/projects/stable-baselines3-contrib/logs/%A_%a.out.log
#SBATCH -e /home/palenicek/projects/stable-baselines3-contrib/logs/%A_%a.err.log
#SBATCH --comment daniel.palenicek@tu-darmstadt.com
## Make sure to create the logs directory /home/user/Documents/projects/prog/logs, BEFORE launching the jobs.

# Setup Env
SCRIPT_PATH=$(dirname $(scontrol show job $SLURM_JOBID | awk -F= '/Command=/{print $2}'))
echo $SCRIPT_PATH

source $SCRIPT_PATH/conda_hook
echo "Base Conda: $(which conda)"
eval "$($(which conda) shell.bash hook)"
conda activate sb3-contrib
echo "Conda Env:  $(which conda)"

export GTIMER_DISABLE='1'
echo "GTIMER_DISABLE: $GTIMER_DISABLE"

cd $SCRIPT_PATH
echo "Working Directory:  $(pwd)"

python /home/palenicek/projects/stable-baselines3-contrib/run.py \
    -env $ENV \
    -seed $SLURM_ARRAY_TASK_ID
