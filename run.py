import argparse
import os
import wandb

from sb3_contrib import CrossQ, SAC



parser = argparse.ArgumentParser() 
parser.add_argument("-env", type=str, default="Walker2d-v4", help="Environment string") 
parser.add_argument("-seed", type=int, required=False, default=1, help="Set Seed.")
args = parser.parse_args()

env = args.env
seed = args.seed

def is_slurm_job():
    """Checks whether the script is run within slurm"""
    return bool(len({k: v for k, v in os.environ.items() if 'SLURM' in k}))

with wandb.init(
    entity='ias', # TODO: remove for publication
    project='sb3-contrib-crossq',
    name=f"seed={seed}",
    # group=f"{env}_BN_fix",
    group=f"{env}_SAC_lr=1e-3",
    tags=[],
    sync_tensorboard=True,
    # config=args_dict,
    settings=wandb.Settings(start_method="fork") if is_slurm_job() else None,
    mode='online' if is_slurm_job() else 'disabled',
) as wandb_run:
    
    # model = CrossQ(
    model = SAC(
        "MlpPolicy", 
        env, 
        learning_starts=5_000,
        learning_rate=1e-3,
        seed=seed,
        tensorboard_log=f"logs/{env}",
        verbose=1
    )
    model.learn(total_timesteps=1_000_000, log_interval=10)
    # model.save("crossq_walker")