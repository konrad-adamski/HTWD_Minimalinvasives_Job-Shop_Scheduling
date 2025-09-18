#!/bin/bash
#SBATCH --job-name=run_experiments_cp_sat
#SBATCH --time=6-23:59:00
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

# --- Pflichtargumente prüfen ---
if [[ "$@" != *"--time_limit"* ]] || [[ "$@" != *"--bound_no_improvement_time"* ]] || [[ "$@" != *"--bound_warmup_time"* ]]; then
    echo "ERROR: You must provide --time_limit, --bound_no_improvement_time and --bound_warmup_time!"
    echo "Example: sbatch run_cp_experiments.sh --util 0.75 --sigma 0.1 --time_limit 600 --bound_no_improvement_time 120 --bound_warmup_time 60"
    exit 1
fi

# Zusätzlich: --sigma erforderlich
if [[ "$@" != *"--sigma"* ]]; then
    echo "ERROR: You must provide --sigma!"
    echo "Example: sbatch run_cp_experiments.sh --util 0.75 --sigma 0.1 --time_limit 600 --bound_no_improvement_time 120 --bound_warmup_time 60"
    exit 1
fi

# Zusätzlich: --util erforderlich
if [[ "$@" != *"--util"* ]]; then
    echo "ERROR: You must provide --util!"
    echo 'Allowed: a single value from the config or "all".'
    echo "Example: sbatch run_cp_experiments.sh --util 0.75 --sigma 0.1 --time_limit 600 --bound_no_improvement_time 120 --bound_warmup_time 60"
    exit 1
fi

# --- Python starten ---
python run_cp_experiments.py "$@"
