# python -u train/train_trans.py --gpu_id=5 --batch_type=sparse --interval=1 --trans_block=2&

# srun -N 1 --gres gpu:rtx_A6000:1 --cpus-per-task 2 --time-min 14400 --qos preemptive --pty bash
# srun -N 1 --gres gpu:rtx_3090:1 --cpus-per-task 2 --time-min 14400 --qos preemptive --pty bash
source /home/zengwenqi/local/anaconda3/bin/activate tf_env


python -u train_trans.py --batch_type=window --interval=5 --trans_block=2 --gpu_id=1 --data_type='Fip35_macro5'
python -u train_trans.py --batch_type=window --interval=1 --trans_block=2 --gpu_id=1 --data_type='Fip35_macro5'
python -u train_trans.py --batch_type=window --interval=1 --trans_block=2 --gpu_id=1 --data_type='Fip35_micro100'
python -u train_trans.py --batch_type=window --interval=5 --trans_block=2 --gpu_id=1 --data_type='Fip35_micro100'
python -u generation.py --gpu_id=1 --data_type='Fip35_macro5' --ckpt_choice=epoch40 --ckpt_task=Label0.0_window50_interval5_lr0.0005_emb_dim128_l100_block2_scheduled
python -u generation.py --gpu_id=1 --data_type='Fip35_macro5' --ckpt_choice=epoch40 --ckpt_task=Label0.0_window50_interval5_lr0.0005_emb_dim128_l100_block2_scheduled
python -u generation.py --gpu_id=1 --data_type='Fip35_macro5' --ckpt_choice=epoch40 --ckpt_task=Label0.0_window50_interval5_lr0.0005_emb_dim128_l100_block2_scheduled
python -u generation.py --gpu_id=1 --data_type='Fip35_macro5' --ckpt_choice=epoch40 --ckpt_task=Label0.0_window50_interval5_lr0.0005_emb_dim128_l100_block2_scheduled


tmux new -s session_name
tmux ls
tmux attach -t session_name
tmux kill-session -t session_name

