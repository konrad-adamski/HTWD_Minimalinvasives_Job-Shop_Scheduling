#!/bin/bash
#SBATCH --job-name=run_single_experiment
#SBATCH --time=6-23:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --output=slurm_%j.out

PROJECT_DIR=/data/horse/ws/koad444h-scheduling-workspace/HTWD_Minimalinvasives_Job-Shop_Scheduling

module load GCCcore/13.2.0
module load Python/3.11.5

source "$PROJECT_DIR/env/bin/activate"
export PYTHONPATH="$PROJECT_DIR:$PYTHONPATH"

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

export MAX_CPU_NUMB="${MAX_CPU_NUMB:-${SLURM_CPUS_PER_TASK:-8}}"

echo "CPUs=$MAX_CPU_NUMB, RAM(total/node)=${SLURM_MEM_PER_NODE:-unset}, Time=6d23h"

cd "$PROJECT_DIR"

# --- Python starten ---
python run_single_experiment.py "$@"
