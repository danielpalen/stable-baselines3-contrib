#!/bin/bash

for env in 'Ant-v4' 'Humanoid-v4'; do
    ENV=$env ALGO='CrossQ' sbatch slurm_experiment.sh;
    # ENV=$env ALGO='SAC' sbatch slurm_experiment.sh;
done
