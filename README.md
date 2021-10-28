Store the NAEP folder in `../data/` folder.

To load some task, use,
```
from utils.load_data import load_dataset
data = load_dataset(task, create_hash=False,train=0.6, valid=0.2)
```
Task list can be found in `data/tasks.json`. This code will check the split hash with my hash stored in `data\task_hash.json` file. 
