######### train model #########
python main.py --mode train --config config.yaml  --max_episodes 100 --model_save_path ./results/models/0617_2

# ppo both
python main.py --mode train --config ./configs/opt_both_config_ppo.yaml --max_episodes 400 --model_save_path ./results/models/0821_1

python main.py --mode train --config ./configs/opt_size_config_ppo.yaml --max_episodes 400 --model_save_path ./results/models/0906_3

######### test model #########
# test spacing optimization
python main.py --mode test --config ./configs/opt_spacing_config.yaml --resume_path ./results/models/0728_3/dqn_final.pth

# test size optimization
python main.py --mode test --config ./configs/opt_size_config.yaml --resume_path ./results/models/0729_3/best_model.pth

# ppo spacing
python main.py --mode test --config ./configs/opt_spacing_config_ppo.yaml --resume_path ./results/models/0809_1/ppo_final.pth

# ppo_both
python main.py --mode test --config ./configs/opt_both_config_ppo.yaml --resume_path ./results/models/0809_3/ppo_final.pth
