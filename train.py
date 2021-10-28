from utils.load_data import load_dataset
from utils.utils import open_json
from transformers import AutoTokenizer, AutoModel
#tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
#model = AutoModel.from_pretrained("bert-base-uncased")
tasks = open_json('data/tasks.json')
for task in tasks:
    #create_hash=True will create a new hash, So use create_hash=False if not trying new split. It will check hashes in data/task_hashes.json file.
    data = load_dataset(task, create_hash=False,train=0.6, val=0.2, test=0.2)