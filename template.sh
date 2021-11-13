#!/bin/bash
#SBATCH --gres gpu:1
#SBATCH --cpus-per-task={n_cpu}
#SBATCH --mem={memory}
#SBATCH --mail-type=END
#SBATCH --mail-user=aritraghosh.iem@gmail.com
#SBATCH --partition={gpu}-{long}
#SBATCH -o /mnt/nfs/scratch1/arighosh/naep/slurm/%j.out
module load python3/current
cd /mnt/nfs/scratch1/arighosh/naep
source ../venv/simclr/bin/activate
python train.py\
    --task "{task}"\
    --lm {lm}\
    --losses "{losses}"\
    --generate {generate}\
    --lr {lr}\
    --batch_size {batch_size}\
    --seed {seed}\
    --iters {iters}\
    {fixed_params}\
    --name "${SLURM_JOB_ID}" --nodes "${SLURM_JOB_NODELIST}" --slurm_partition "${SLURM_JOB_PARTITION}"