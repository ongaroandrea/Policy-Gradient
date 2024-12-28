import torch
import gym
import numpy as np
import argparse
import matplotlib.pyplot as plt
from agent import Agent, Policy
from cp_cont import CartPoleEnv
import pandas as pd
import sys
import time

import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

from stable_baselines3 import PPO, SAC
from stable_baselines3.common.evaluation import evaluate_policy

from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.results_plotter import load_results, ts2xy


def set_seed(seed):
    if seed > 0:
        np.random.seed(seed)


def create_model(args, env):
    # T4 TODO
    if args.algo == "ppo":
        model = PPO(
            args.policy,
            env,
            learning_rate=args.lr,
            verbose=1,
            tensorboard_log=args.tensorboard_log,
        )

    elif args.algo == "sac":
        model = SAC(
            args.policy,
            env,
            learning_rate=args.lr,
            verbose=1,
            tensorboard_log=args.tensorboard_log,
        )
    else:
        raise ValueError(f"RL Algo not supported: {args.algo}")
    return model


def load_model(args, env):
    # T4 TODO
    if args.algo == "ppo":
        model = PPO.load(args.test, env=env)
    elif args.algo == "sac":
        model = SAC.load(args.test, env=env)
    else:
        raise ValueError(f"RL Algo not supported: {args.algo}")
    return model


def moving_average(values, window):
    """
    Smooth values by doing a moving average
    :param values: (numpy array)
    :param window: (int)
    :return: (numpy array)
    """
    weights = np.repeat(1.0, window) / window
    return np.convolve(values, weights, "valid")


def plot_results(log_folder, title="Learning Curve", save_path=None):
    """
    plot the results

    :param log_folder: (str) the save location of the results to plot
    :param title: (str) the title of the task to plot
    """
    x, y = ts2xy(load_results(log_folder), "timesteps")
    y = moving_average(y, window=50)
    # Truncate x
    x = x[len(x) - len(y) :]

    plt.figure(title)
    plt.plot(x, y)
    # Increase font size of axis
    plt.tick_params(axis="both", which="major", labelsize=14)
    plt.xlabel("Number of Timesteps", fontsize=18)
    plt.ylabel("Rewards", fontsize=18)
    plt.title(title + " Smoothed", fontsize=18)
    if save_path is not None:
        plt.savefig(save_path)
    else:
        plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--test", "-t", type=str, default=None, help="Model to be tested"
    )
    parser.add_argument(
        "--env", type=str, default="ContinuousCartPole-v0", help="Environment to use"
    )
    parser.add_argument(
        "--total_timesteps",
        type=int,
        default=50000,
        help="The total number of samples to train on",
    )
    parser.add_argument("--render_test", action="store_true", help="Render test")
    parser.add_argument("--seed", default=0, type=int, help="Random seed")
    parser.add_argument("--algo", default="sac", type=str, help="RL Algo [ppo, sac]")
    parser.add_argument("--lr", default=0.0003, type=float, help="Learning rate")
    parser.add_argument(
        "--gradient_steps",
        default=-1,
        type=int,
        help="Number of gradient steps when policy is updated in sb3 using SAC. -1 means as many as --args.now",
    )
    parser.add_argument(
        "--test_episodes", default=300, type=int, help="# episodes for test evaluations"
    )

    # Custom arguments for PPO
    parser.add_argument(
        "--policy",
        default="MlpPolicy",
        type=str,
        help="Policy model to use (MlpPolicy, CnnPolicy, ...)",
    )
    parser.add_argument(
        "--tensorboard_log",
        default="./data/log/ppo_cartpole_tensorboard/",
        type=str,
        help="Directory to save tensorboard logs",
    )

    args = parser.parse_args()

    set_seed(args.seed)

    env = gym.make(args.env)

    log_dir = "./data/tmp/gym/"
    os.makedirs(log_dir, exist_ok=True)
    # Logs will be saved in log_dir/monitor.csv
    env = Monitor(env, log_dir)

    # If no model was passed, train a policy from scratch.
    # Otherwise load the model from the file and go directly to testing.
    if args.test is None:
        try:
            model = create_model(args, env)
            # Policy training (T4) TODO
            start = time.time()
            model.learn(total_timesteps=args.total_timesteps)
            end = time.time()
            print(f"Training time: {end - start} s")
            # Saving model (T4) TODO
            model.save(f"./data/model/{args.algo}_{args.env}")
            # Plotting result
            plot_results(
                log_dir,
                title=f"{args.algo}",
                save_path=f"./data/plot/{args.algo}_{args.env}.png",
            )

        # Handle Ctrl+C - save model and go to tests
        except KeyboardInterrupt:
            print("Interrupted!")
            model.save(f"{args.algo}_{args.env}")

    else:
        print("Testing...")
        model = load_model(args, env)
        # Policy evaluation (T4) TODO
        mean_reward, std_reward = evaluate_policy(
            model,
            model.get_env(),
            n_eval_episodes=args.test_episodes,
            render=args.render_test,
        )

        print(
            f"Test reward (avg +/- std): ({mean_reward} +/- {std_reward}) - Num episodes: {args.test_episodes}"
        )

    env.close()
