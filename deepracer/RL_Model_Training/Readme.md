# Training policies for deepracer

The instructions assume the deepracer simulator is running locally on the default ports.
Install and setup the OpenAI baselines and OpenAI Gym.


## 1. Train the policies with Time-in-State(TS) for deepracer
Add to the Main_ts.py file the path to save model checkpoints on line 14: log_path = 'Path_to_save_checkpoints'
```
python Main_ts.py
```


## 2. Train the policies using Domain Randomization(DR) for deepracer
Add to the Main_dr.py file the path to save model checkpoints on line 14: log_path = 'Path_to_save_checkpoints'
```
python Main_dr.py
```

## 3. Monitoring the progress of Training
In the folder Checkpoint_path, the 'progress.csv' saves the information of training that is used to plot the learning curve.
The checkpoint with the highest 'mean_evaluation_reward' from the progress.csv is selected.


## 4. Benchmarking the policies
The TS and DR policies can be benchmarked using the files benchmark_ts.py and benchmark_dr.py respectively.
The files requires the checkpoint to be added in the starting of the files by specifying path1, path2, path3 for
the three saved checkpoints.
Run the files using:

```
python benchmark_ts.py
python benchmark_dr.py
```


The data is saved in the folders: 'data_ts' and 'data_dr'


## 6. Visualize the results by running the following file
```
python visualize.py
```
This file assumes, the benchmarking is done by following the step 4.
