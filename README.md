# Automatic Short Math Answer Grading via In-context Meta-learning


## Example for executing train.py for generalizing to new questions

```console
foo@bar:~$ train.py 
    --in_context_tuning     #applying in context learning
    --meta_learning         #training all data as one model
    --finetune              #finetune mode for few-shot learning
    --name=meta_example     #name for saving
    --batch_size=4
    --iters=10
    --seed=-1
    --new_examples=-1
    --data_folder=[***.csv] #should be csv format
    --fold=4                #choose folder for cross-validation
    --alias=_meta           #alias append on "name"
    --all                   #record all result
    --lm=saved_models/[dir] #Loading pretrained model should include saved_model path
                            #[dir] is the folder name for your model 
```


```python
There are some more alternative options, 
located at train.py  def add_learner_params()
```

## For citation: 
```
```