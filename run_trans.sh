# python -u train/train_trans.py --gpu_id=5 --batch_type=sparse --interval=1 --trans_block=2&

# srun -N 1 --gres gpu:rtx_A6000:1 --cpus-per-task 2 --time-min 14400 --qos preemptive --pty bash
# srun -N 1 --gres gpu:rtx_3090:1 --cpus-per-task 2 --time-min 14400 --qos preemptive --pty bash
source /home/zengwenqi/local/anaconda3/bin/activate tf_env

# python Pointer_Net/train2.py --lr=0.0002 --hidden_dim=64 --dropout=0.1 --num_layers=2 --batch_size=64
python -u train/train_trans.py --batch_type=window --interval=1 --trans_block=2 --gpu_id=0
python -u generation.py --gpu_id=2
# python script/run.py -c config/transductive/fb15k237_astarnet.yaml --gpus [0]
# cd /home/zengwenqi/projects/Trace_Seg_7class
# python -u train_CRF.py
# python /home/zengwenqi/projects/Trace_Seg_7class/evaluate_CRF.py

tmux new -s session_name
tmux ls
tmux attach -t session_name
tmux kill-session -t session_name

