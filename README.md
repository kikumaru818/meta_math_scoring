Store the NAEP folder in `../data/` folder.

## Final Submission Dataset
To load the submission datasets for some task, use,
```
from utils.load_data import submission_load_dataset
data = submission_load_dataset(task, create_hash=False,train=0.8, valid=0.2)
```
The `l1` aka score1 and `l2` aka score2 fields of the test samples are set to 1 arbitrarily. 

## Local Dataset,
To load some task, use,
```
from utils.load_data import load_dataset
data = load_dataset(task, create_hash=False,train=0.6, valid=0.2, fold=1/2/3/4/5)
```
Task list can be found in `data/tasks.json`. This code will check the split hash with my hash stored in `data\task_hash.json` file. 
