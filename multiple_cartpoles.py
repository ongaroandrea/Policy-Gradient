import torch
import gym
import numpy as np
import argparse
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from itertools import repeat
import sys
import multiprocessing as mp
from cartpole import train


# Parse script arguments
def parse_args(args=sys.argv[1:]):
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--env", type=str, default="ContinuousCartPole-v0", help="Environment to use"
    )
    parser.add_argument(
        "--num_runs",
        type=int,
        default=10,
        help="How many independent training runs to perform",
    )

    parser.add_argument(
        "--algorithm",
        type=str,
        default="normalized",
        help="Algorithm to use for training (basic, constant_baseline, normalized)",
    )
    
    return parser.parse_args(args)


def trainer(args):
    trainer_id, env, algo = args
    print("Trainer id", trainer_id, "started")
    training_history = train(env, False, False, trainer_id, algorithm=algo)
    print("Trainer id", trainer_id, "finished")
    return training_history


# The main function
def main(args):
    # Create a pool with cpu_count() workers
    pool = mp.Pool(processes=mp.cpu_count())

    # Run the train function num_runs times
    results = pool.map(
        trainer, zip(range(args.num_runs), repeat(args.env, args.num_runs), repeat(args.algorithm, args.num_runs))
    )

    # Put together the results from all workers in a single dataframe
    all_results = pd.concat(results)

    # Calculate the mean of the rewards of total runs
    print("Each run's mean reward:", all_results.groupby("train_run_id")["reward"].mean())

    all_results_s = (
        all_results.groupby("train_run_id")["reward"].mean().reset_index().mean()
    )

    print("Mean of the rewards of total runs:", all_results_s["reward"])

    # Save the dataframe to a file
    all_results.to_pickle("data/dataframe/rewards_" + args.algorithm + ".pkl")

    sns.set_theme(font_scale=2.3)
    figsize = (20, 12)
    plt.gcf().set_size_inches(*figsize)

    # Plot the mean learning curve, with the standard deviation
    sns.lineplot(x="episode", y="reward", data=all_results, errorbar="sd")

    # Plot (up to) the first 5 runs, to illustrate the variance
    n_show = min(args.num_runs, 5)
    smaller_df = all_results.loc[all_results.train_run_id < n_show]
    sns.lineplot(
        x="episode",
        y="reward",
        hue="train_run_id",
        data=smaller_df,
        dashes=[(2, 2)] * n_show,
        palette="Set2",
        style="train_run_id",
    )
    plt.title("Training performance")
    plt.savefig("data/plot/training_multiple_" + args.algorithm + "_10000.png")
    #plt.savefig("data/plot/training_multiple_" + args.algorithm + "_10000.pdf")
    # plt.show()


# Entry point of the script
if __name__ == "__main__":
    args = parse_args()
    main(args)
