[![Review Assignment Due Date](https://classroom.github.com/assets/deadline-readme-button-22041afd0340ce965d47ae6ef1cefeee28c7c493a6346c4f15d667ab976d596c.svg)](https://classroom.github.com/a/8DzvMbmT)

# Exercise 3

Exercise 3 - RL Policy Gradient algorithms

For this exercise you must use gym v0.21.0 and stable-baselines3 v1.6.2.

You can install the required python packages with this command:

```sh
pip install -r ./required_python_libs.txt
```

# For mac m1 user
tensorboard --logdir .
zsh: command not found: tensorboard
pip install tensorboard=2.1.0
AttribsuteError: module 'numpy' has no attribute 'object'. `np.object` was a deprecated alias for the builtin `object`. To avoid this error in existing code, use `object` by itself.
pip install numpy==1.23.4
tensorboard --logdir .


# Instructions

## Training 

```sh
python cartpole.py --algorithm "name"
```

Possible algorithm:
- basic
- constant_baseline with b = 20
- normalized

The script produces:
- a model available in data/model
- a plot available in data/plot

Other possible parameters:

- "--env"
- --train_episodes

# Testing 
```sh
python cartpole.py --test ./data/model/model_ContinuousCartPole-v0_0_basic.mdl --render --episodes 10
```

```sh
python cartpole.py --test ./data/model/model_ContinuousCartPole-v0_0_constant_baseline.mdl --render --episodes 10
```

```sh
python cartpole.py --test ./data/model/model_ContinuousCartPole-v0_0_normalized.mdl --render --episodes 10
```
