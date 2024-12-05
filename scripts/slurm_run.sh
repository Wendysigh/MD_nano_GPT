#!/bin/bash
#SBATCH -J MD             # task name
#SBATCH -N 1                      # node for task
#SBATCH --ntasks=1
#SBATCH --gres gpu:rtx_3090:1          # gpu nums for task
# SBATCH --gres gpu:rtx_A6000:1  
#SBATCH --time=3-0
#SBATCH --output=/home/zengwenqi/projects/MD_code2/slurm_logs/output_%j.out 
#SBATCH --error=/home/zengwenqi/projects/MD_code2/slurm_logs/err_%j.err

# srun -N 1 --gres gpu:rtx_A6000:1 --cpus-per-task 2 --time-min 14400 --qos preemptive --pty bash
source /home/zengwenqi/local/anaconda3/bin/activate tf_env
cd /home/zengwenqi/projects/MD_code2/
python -u train_trans.py --batch_type=window --interval=5 --trans_block=2 --gpu_id=0 &&
python -u generation.py --gpu_id=0 --ckpt_choice=epoch40 --ckpt_task=Label0.0_window50_interval5_lr0.0005_emb_dim128_l100_block2_scheduled