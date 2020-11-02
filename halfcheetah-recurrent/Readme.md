# Training Recurrent Policies for HalfCheetah

## Code Credits

The PPO implementation with recurrent policies is taken from the repo available from Google-research. The original repo along with the reference to paper is available [here](https://github.com/google-research/batch-ppo). The changes include modification of the network architecture to make both value and policy network recurrent, and plugin of the custom HalfCheetah environment allowing variable timing properties during training.


## Requirements
Simple way to setup the environment
```
$ conda create --name rlenv python=3.6
$ conda activate rlenv
$ conda install -c conda-forge tensorflow=1.14
$ conda install -c conda-forge gym
$ pip install pybullet
$ conda install -c conda-forge ruamel.yaml
$ conda install -c conda-forge matplotlib
```

## Training Policies

Follow the instructions in the *RL_Model_Training* to train the recurrent models for HalfCheetah.
