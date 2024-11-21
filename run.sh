#!/bin/bash
#SBATCH -J MD             # task name
#SBATCH -N 1                      # node for task
#SBATCH --ntasks=1
#SBATCH --gres gpu:rtx_3090:1          # gpu nums for task
# SBATCH --gres gpu:rtx_A6000:1  
#SBATCH --time=3-0
#SBATCH --output=/home/zengwenqi/projects/slurm_output/output_%j.out 
#SBATCH --error=/home/zengwenqi/projects/slurm_output/err_%j.err

# srun -N 1 --gres gpu:rtx_A6000:1 --cpus-per-task 2 --time-min 14400 --qos preemptive --pty bash
source /home/zengwenqi/local/anaconda3/bin/activate tf_env
# cd /home/zengwenqi/projects/Trace_Seg_7class
# python Pointer_Net/train2.py --lr=0.0002 --hidden_dim=64 --dropout=0.1 --num_layers=2 --batch_size=64
python -u train/train_trans.py --batch_type=window --interval=1 --trans_block=2 --gpu_id=0