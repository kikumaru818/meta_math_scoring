import itertools
import collections
import glob
import os
import datetime
import subprocess
import string
import sys
from utils.utils import safe_makedirs,open_json
tasks = open_json('data/tasks.json')

create_dirs = ['logs/', 'slurm/', 'configs/']
for c in create_dirs:
    safe_makedirs(c)

def get_memory(combo):
    return 20000

def get_cpu(combo):
    return 3

class SafeDict(dict):
    def __missing__(self, key):
        return '{' + key + '}'

def get_run_id():
    filename = "logs/expts.txt"
    if os.path.isfile(filename) is False:
        with open(filename, 'w') as f:
            f.write("")
        return 0
    else:
        with open(filename, 'r') as f:
            expts = f.readlines()
        run_id = len(expts)
    return run_id

    
def is_long(combo):
    return 'long'

save = False
hyperparameters = [
    [('task',), tasks],#[  "Grade 4/2017_DBA_DR04_1715RE1T10_05"]],
    [('lm',), ['bert-base-uncased']],#'bert-base-uncased','roberta-base','bert-large-uncased','roberta-large','gpt2'
    [('losses',), ['cce;qwp', 'cce', 'qwp']],
    [('lr',), [1e-5]],#2e-4
    [('iters',), [100]],
    [('seed',), [999]],
    [('batch_size',), [32]],
]

def get_gpu(combo):
    if 'large' in combo['lm']:
        return "m40"
    if 'cce' in combo['losses']:
        return "1080ti"
    return "titanx"

    
def is_valid(combo):
    return True


other_dependencies = {'gpu': get_gpu, 'memory': get_memory, 'n_cpu':get_cpu, 'valid':is_valid, 'long':is_long}

run_id = int(get_run_id())

key_hyperparameters = [x[0] for x in hyperparameters]
value_hyperparameters = [x[1] for x in hyperparameters]
combinations = list(itertools.product(*value_hyperparameters))

scripts = []
gpu_counts =collections.defaultdict(int)

for combo in combinations:
    # Write the scheduler scripts
    template_name = "template.sh"
    with open(template_name, 'r') as f:
        schedule_script = f.read()

    combo = {k[0]: v for (k, v) in zip(key_hyperparameters, combo)}

    for k, v in other_dependencies.items():
         combo[k] = v(combo)
    if not combo['valid']:
        #print(combo)
        continue
    combo['run_id'] = run_id
    gpu_counts[combo['gpu']] +=1

    for k, v in combo.items():
        if "{%s}" % k in schedule_script:
            schedule_script = schedule_script.replace("{%s}" % k, str(v))
    


    schedule_script += "\n"

    # Write schedule script
    script_name = 'configs/lm_%d.sh' % run_id
    with open(script_name, 'w') as f:
        f.write(schedule_script)
    scripts.append(script_name)

    # Making files executable
    subprocess.check_output('chmod +x %s' % script_name, shell=True)

    # Update experiment logs
    output = "Script Name = " + script_name +", Time Now= "+ datetime.datetime.now().strftime("%Y-%m-%d %H:%M") + "\n" 
    with open("logs/expts.txt", "a") as f:
        f.write(output)
    # For the next job
    run_id += 1

print(gpu_counts)
# schedule jobs
for script in scripts:#--exclude=node078
    command = "sbatch  %s" % script
    #print(command)
    print(subprocess.check_output(command, shell=True))