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
    if combo['task']=='all':
        return 50000
    return 20000

def get_cpu(combo):
    if combo['task']=='all':
        return 4
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

UNITY_CONSTRAINTS = '#SBATCH --constraint="ials_gigabyte_gpu_2020"'
UNITY_BASE = "/gypsum/scratch1/arighosh/naep"
GYPSUM_BASE = "/mnt/nfs/scratch1/arighosh/naep"
UNITY_PATHS = 'module load python/3.9.1\nmodule load cuda/10.2.89\ncd {}\nsource ~/.venv/catext/bin/activate'.format(UNITY_BASE)
GYPSUM_PATHS = 'module load python3/current\ncd {}\nsource ../venv/simclr/bin/activate'.format(GYPSUM_BASE)
    
def is_long(combo):
    return 'long'

save = False
fixed_params = '   '.join(['--neptune', '--cuda', '--include_question'])
hyperparameters = [
    [('task',), tasks]#[  "Grade 4/2017_DBA_DR04_1715RE1T10_05"]],#, 'facebook/bart-large','microsoft/deberta-v2-xlarge', 'facebook/bart-large'
    ,[('lm',), ['bert-base-uncased']]#'bert-base-uncased','roberta-base','bert-large-uncased','roberta-large','gpt2'
    ,[('losses',), [ 'cce' ]]
    ,[('problem',), [ 'base' ]]
    ,[('generate',), ['none']]
    ,[('lr',), [2e-5]]#2e-4
    ,[('iters',), [100]]
    ,[('fold',), [1,2,3,4,5]]
    ,[('seed',), [999]]
    ,[('batch_size',), [8]]
    ,[('fixed_params',), [fixed_params]]
    ,[('cluster',), ['unity']]
]

def get_base_path(combo):
    return UNITY_BASE if combo['cluster'] =='unity' else GYPSUM_BASE

def get_gpu(combo):
    return 'gpu'
    if 'xlarge' in combo['lm']:
        return "m40"
    if 'cce' in combo['losses']:
        return "1080ti"
    return "1080ti"

    
def is_valid(combo):
    return True

def get_constraints(combo):
    if combo['cluster']=='unity':
        return UNITY_CONSTRAINTS
    return ""
def get_paths(combo):
    if combo['cluster']=='unity':
        return UNITY_PATHS
    return GYPSUM_PATHS
    
other_dependencies = {'gpu': get_gpu, 'memory': get_memory, 'n_cpu':get_cpu, 'valid':is_valid, 'long':is_long, 'constraints':get_constraints, 'paths':get_paths, 'base_path' :get_base_path}

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
excludes =  "--exclude=node128,node097,node094,node095" if combo['cluster']=='gypsum' else "--exclude=node46,node53"
for script in scripts:#--exclude=node078
    command = "sbatch {} {}".format(excludes,script)
    # print(command)
    print(subprocess.check_output(command, shell=True))