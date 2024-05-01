import argparse
import os
import wandb

import gymnasium

from stable_baselines3.common.callbacks import EvalCallback, CallbackList

from sb3_contrib import CrossQ, SAC



parser = argparse.ArgumentParser() 
parser.add_argument("-env", type=str, default="Walker2d-v4", help="Environment string") 
parser.add_argument("-algo", type=str, default="CrossQ", help="Environment string") 
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
    # group=f"{env}_{args.algo}_lr=3e-4_BRN_fixed",
    group=f"{env}_{args.algo}_lr=1e-3_BRN_recoded_long",
    tags=[],
    sync_tensorboard=True,
    # config=args_dict,
    settings=wandb.Settings(start_method="fork") if is_slurm_job() else None,
    mode='online' if is_slurm_job() else 'disabled',
) as wandb_run:
    
    # model = CrossQ(
    
    cls = {
        "CrossQ": CrossQ,
        "SAC": SAC,
    }[args.algo]

    model = cls(
        "MlpPolicy", 
        env, 
        learning_starts=5_000,
        learning_rate=1e-3,
        # learning_rate=3e-4,
        seed=seed,
        tensorboard_log=f"logs/{env}",
        verbose=1
    )
    callback_list = CallbackList([
        EvalCallback(
            eval_env=gymnasium.make(env),
            n_eval_episodes=1,
            eval_freq=max(5_000_000 // 300, 1),
            deterministic=True,
            callback_on_new_best=None,
            best_model_save_path=f"models/{env}",
            verbose=1,
        ),
    ])
    model.learn(total_timesteps=5_000_000, log_interval=4, callback=callback_list)
    # model.save("crossq_walker")