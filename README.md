# Exercise 3 - RL Policy Gradient algorithms

For this exercise you must use gym v0.21.0 and stable-baselines3 v1.6.2.

You can install the required python packages with this command:

```sh
pip install -r ./required_python_libs.txt
```

# For mac m1 user

If after installation of the packages you get this error when launching the tensorboard:

AttribsuteError: module 'numpy' has no attribute 'object'. `np.object` was a deprecated alias for the builtin `object`. To avoid this error in existing code, use `object` by itself.

Use this command to fix with this

```sh
pip install numpy==1.23.4
pip install tensorflow==2.1.0
```

Test if everything works

```sh
tensorboard --logdir .
```

# Instructions

## Training

```sh
python cartpole.py --algorithm "name"
```

Possible algorithm:

- basic
- constant_baseline with b = 20
- normalized rewards

The script produces:

- a model available in data/model
- a plot available in data/plot

Other possible parameters:

- "--env"
- --train_episodes

# Testing

```sh
python cartpole.py --test ./data/model/model_ContinuousCartPole-v0_0_basic.mdl --render --episodes=10
```

```sh
python cartpole.py --test ./data/model/model_ContinuousCartPole-v0_0_constant_baseline.mdl --render --episodes=10
```

```sh
python cartpole.py --test ./data/model/model_ContinuousCartPole-v0_0_normalized.mdl --render --episodes=10
```

```sh
python cartpole.py --test ./data/model/model_ContinuousCartPole-v0_0_basic_10000.mdl --algo=basic
```

```sh
python cartpole.py --test ./data/model/model_ContinuousCartPole-v0_0_constant_baseline_10000.mdl --algo=constant_baseline
```

```sh
python cartpole.py --test ./data/model/model_ContinuousCartPole-v0_0_normalized_10000.mdl --algo=normalized
```

# Testing SAC & POC

```sh
python cartpole_sb3.py --test ./data/model/ppo_ContinuousCartPole-v0.zip --render_test --algo=ppo

python cartpole_sb3.py --test ./data/model/sac_ContinuousCartPole-v0.zip --render_test --algo=sac
```
