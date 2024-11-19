# python -u /home/wzengad/projects/MD_code/LSTM/train_trans.py --gpu_id=5 --batch_type=sparse --interval=1 --trans_block=2&
# python -u /home/wzengad/projects/MD_code/LSTM/train_trans.py --gpu_id=5 --batch_type=sparse --data_type='psi' --interval=1 --trans_block=2&
# wait


# # python -u /home/wzengad/projects/MD_code/post_ana/Fig_kinetic_distance.py --ckpt_task='Label0.0_sparse50_interval1_lr0.0001_emb_dim128_l100_block2_scheduled' &
# python -u /home/wzengad/projects/MD_code/LSTM/generation_4states.py --ckpt_task=Label0.0_window50_interval1_lr0.0001_emb_dim128_l100_block2_scheduled --ckpt_choice=epoch10 --gpu_id=6 &
# python -u /home/wzengad/projects/MD_code/LSTM/generation_4states.py --ckpt_task=Label0.0_window50_interval1_lr0.0001_emb_dim128_l100_block3_scheduled --ckpt_choice=epoch10 --gpu_id=7 &
# python -u /home/wzengad/projects/MD_code/LSTM/generation_4states.py --ckpt_task=Label0.0_window50_interval1_lr0.0001_emb_dim256_l100_block2_scheduled --ckpt_choice=epoch10 --gpu_id=8 &
# python -u /home/wzengad/projects/MD_code/LSTM/generation_4states.py --ckpt_task=Label0.0_window50_interval1_lr0.0001_emb_dim256_l100_block3_scheduled --ckpt_choice=epoch10 --gpu_id=8 &

# python -u /home/wzengad/projects/MD_code/post_ana/Fig_kinetic_distance.py --ckpt_task=Label0.0_window50_interval1_lr0.0001_emb_dim128_l100_block2_scheduled&
python -u /home/wzengad/projects/MD_code/post_ana/Fig_kinetic_distance.py --ckpt_task=Label0.0_window50_interval1_lr0.0001_emb_dim128_l100_block3_scheduled&
wait
python -u /home/wzengad/projects/MD_code/post_ana/Fig_kinetic_distance.py --ckpt_task=Label0.0_window50_interval1_lr0.0001_emb_dim256_l100_block2_scheduled&
wait
python -u /home/wzengad/projects/MD_code/post_ana/Fig_kinetic_distance.py --ckpt_task=Label0.0_window50_interval1_lr0.0001_emb_dim256_l100_block3_scheduled&
wait
python -u /home/wzengad/projects/MD_code/post_ana/Fig_kinetic_distance.py --ckpt_task=Label0.0_sparse50_interval1_lr0.0001_emb_dim128_l100_block2_scheduled&
wait 
python -u /home/wzengad/projects/MD_code/post_ana/Fig_kinetic_distance.py --ckpt_task=Label0.0_sparse50_interval1_lr0.0001_emb_dim128_l100_block3_scheduled&
wait
python -u /home/wzengad/projects/MD_code/post_ana/Fig_kinetic_distance.py --ckpt_task=Label0.0_sparse50_interval1_lr0.0001_emb_dim256_l100_block2_scheduled&
wait
python -u /home/wzengad/projects/MD_code/post_ana/Fig_kinetic_distance.py --ckpt_task=Label0.0_sparse50_interval1_lr0.0001_emb_dim256_l100_block3_scheduled&