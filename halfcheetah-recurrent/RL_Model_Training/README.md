## 1. Training Policies

## Training Time-in-state (TS) recurrent policy
```
$ python -m agents.scripts.train --logdir=/path/to/logdir --config=cheetah_ts
```


## Training Domain Randomization (DR) recurrent policy
```
$ python -m agents.scripts.train --logdir=/path/to/logdir --config=cheetah_dr
```

## Training vanilla recurrent policy
```
$ python -m agents.scripts.train --logdir=/path/to/logdir --config=cheetah_va
```


## 2. Monitor progress of training
```
$ tensorboard --logdir=/path/to/logdir
```

## 3. Benchmarking the policies
The TS and DR policies can be benchmarked using the files benchmark_ts.py and benchmark_dr.py respectively. The files requires the checkpoint to be added in the starting of the files. The pretrained checkpoints are added for reference. Run the files using:

```
python benchmark_ts.py
python benchmark_dr.py
```

The data is saved in the folders: 'data_ts' and 'data_dr'

## 4. Visualize the results
This step assumes, the benchmarking is done by following the Benchmarking instructions.

```
python visualize.py
```

## Trained Checkpoints
A sample trained checkpoint is available for both TS and DR in the folder *trained_checkpoints*.

