# python -u /home/wzengad/projects/MD_code/LSTM/train_once_lstm.py --gpu_id=5 --data_type='phi' --interval=1&
# python -u /home/wzengad/projects/MD_code/LSTM/train_once_lstm.py --gpu_id=5 --data_type='psi' --interval=1&
# wait
python -u /home/wzengad/projects/MD_code/LSTM/generation_lstm.py --ckpt_task=Label0.0_sparse50_interval1_lr0.001_l100_units512_emb128_no_pos --data_type='phi' --gpu_id=4 --ckpt_choice=epoch10 &
python -u /home/wzengad/projects/MD_code/LSTM/generation_lstm.py --ckpt_task=Label0.0_sparse50_interval1_lr0.001_l100_units512_emb128_no_pos --data_type='phi' --gpu_id=4 --ckpt_choice=epoch50 &
python -u /home/wzengad/projects/MD_code/LSTM/generation_lstm.py --ckpt_task=Label0.0_sparse50_interval1_lr0.001_l100_units512_emb128_no_pos --data_type='phi' --gpu_id=4 --ckpt_choice=epoch100 &
python -u /home/wzengad/projects/MD_code/LSTM/generation_lstm.py --ckpt_task=Label0.0_sparse50_interval1_lr0.001_l100_units512_emb128_no_pos --data_type='phi' --gpu_id=4 --ckpt_choice=epoch150 &
python -u /home/wzengad/projects/MD_code/LSTM/generation_lstm.py --ckpt_task=Label0.0_sparse50_interval1_lr0.001_l100_units512_emb128_no_pos --data_type='psi' --gpu_id=5 --ckpt_choice=epoch10 &
python -u /home/wzengad/projects/MD_code/LSTM/generation_lstm.py --ckpt_task=Label0.0_sparse50_interval1_lr0.001_l100_units512_emb128_no_pos --data_type='psi' --gpu_id=5 --ckpt_choice=epoch50 &
python -u /home/wzengad/projects/MD_code/LSTM/generation_lstm.py --ckpt_task=Label0.0_sparse50_interval1_lr0.001_l100_units512_emb128_no_pos --data_type='psi' --gpu_id=5 --ckpt_choice=epoch100 &
python -u /home/wzengad/projects/MD_code/LSTM/generation_lstm.py --ckpt_task=Label0.0_sparse50_interval1_lr0.001_l100_units512_emb128_no_pos --data_type='psi' --gpu_id=5 --ckpt_choice=epoch150 &
