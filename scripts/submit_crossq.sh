#!/bin/bash

for env in 'Walker2d-v4' 'Hopper-v4' 'Ant-v4' 'HalfCheetah-v4' 'Humanoid-v4' 'HumanoidStandup-v4'; do
    ENV=$env sbatch slurm_experiment.sh;
done
