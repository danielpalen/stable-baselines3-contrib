#!/bin/bash

for env in 'Ant-v4' 'HalfCheetah-v4' 'Walker2d-v4' 'Hopper-v4' 'Humanoid-v4' 'HumanoidStandup-v4'; do
    ENV=$env ALGO='CrossQ' sbatch slurm_experiment.sh;
    # ENV=$env ALGO='SAC' sbatch slurm_experiment.sh;
done
