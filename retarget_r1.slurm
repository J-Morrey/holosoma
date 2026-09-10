#!/bin/bash
#SBATCH --job-name=r1-retarget
#SBATCH --nodes=1 --ntasks=1 --cpus-per-task=32
#SBATCH --nodelist=zac
#SBATCH --time=14-00:00:00
#SBATCH --output=logs/retarget-%j.out

source ~/miniconda3/etc/profile.d/conda.sh && conda activate hsretarget
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1

DATA=/scratch/$USER/seed
cd $(python -c "import holosoma_retargeting,os;print(os.path.dirname(holosoma_retargeting.__file__))")

python examples/parallel_robot_retarget.py \
    --data-dir $DATA/npz \
    --task-type robot_only --robot r1 --data-format soma \
    --save-dir $DATA/retarget \
    --task-config.object-name ground --task-config.ground-range -15 15 \
    --retargeter.foot-sticking-tolerance 0.02 \
    --max-workers 28
