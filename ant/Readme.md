## Summary:
This folder contains the code to train models for Ant task.
Code credits: The ppo training code is taken from open AI baselines with modifications done to allow
variable timing characteristics during training. The Ant environment and robot is taken from the Pybullet
code with modifications for the variable timing characteristics.

**The code in the current folder trains the fully connected policies for Time-in-State and Domain Randomization settings**

## Requirements
a) Installation and setup the OpenAI baselines.

b) OpenAI Gym and Pybullet

c) Simple way to setup the environment
```
$ conda create --name rlenv python=3.6
$ conda activate rlenv
$ conda install -c conda-forge tensorflow=1.14

Install OpenAI baselines: Follow official instructions

$ conda install -c conda-forge gym
$ pip install pybullet
$ conda install -c conda-forge ruamel.yaml
$ conda install -c conda-forge matplotlib
```

## Training the Time-in-State (TS) and Domain Randomization (DR) policy

a) TS policy training:

```
python Main_ts.py
```
The policies and progress is saved in the folder *ant_ts_policies* in the current directory.
Change the "checkpoint_path = 'custom_path'" on line 9 in Main_ts.py to save the checkpoints and training progress to any custom location.


b) DR policy training:

```
python Main_dr.py
```
The policies and progress is saved in the folder *ant_dr_policies* in the current directory.
Change the "checkpoint_path = 'custom_path'" on line 9 in Main_dr.py to save the checkpoints and training progress to any custom location.



## Monitoring the progress of Training
In the folder Checkpoint_path, the 'progress.csv' saves the information of training that is used to plot the learning curve.
The checkpoint with the highest 'mean_evaluation_reward' from the progress.csv is selected.

## Benchmarking the policies
The TS and DR policies can be benchmarked using the files benchmark_ts.py and benchmark_dr.py respectively.
The files requires the checkpoint to be added in the starting of the files by specifying path1, path2, path3 for
the three saved checkpoints.
Run the files using:

```
python benchmark_ts.py
python benchmark_dr.py
```

The data is saved in the folders: 'data_ts' and 'data_dr'

## Visualize the results
This step assumes, the benchmarking is done by following the Benchmarking instructions.

```
python visualize.py
```

## Trained Checkpoints
A sample trained checkpoint is available for both TS and DR in the folder *ant_dr_policies* and *ant_ts_policies*.
