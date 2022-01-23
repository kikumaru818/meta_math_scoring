#!/bin/bash
#SBATCH --gres gpu:1
#SBATCH --cpus-per-task={n_cpu}
#SBATCH --mem={memory}
#SBATCH --mail-type=END
#SBATCH --mail-user=aritraghosh.iem@gmail.com
#SBATCH --partition={gpu}
#SBATCH -o {base_path}/slurm/%j.out
{constraints}
{paths}
python train.py\
    --task "{task}"\
    --lm {lm}\
    --losses "{losses}"\
    --generate {generate}\
    --lr {lr}\
    --fold {fold}\
    --batch_size {batch_size}\
    --seed {seed}\
    --problem {problem}\
    --iters {iters}\
    {fixed_params}\
    --name "${SLURM_JOB_ID}" --nodes "${SLURM_JOB_NODELIST}" --slurm_partition "${SLURM_JOB_PARTITION}"