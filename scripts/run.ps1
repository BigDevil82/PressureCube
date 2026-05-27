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

#########  optimize with trained PPO model #########
python -m src.reinforcement.sb3.optimize `
  --model_path results\sb3\0422_2206\final_model.zip `
  --L 8.0 --W 3.6 --H 3.6 --pressure 30000 `
  --eval_mode surrogate `
  --surrogate_path results\surrogate_data\200k\model_advanced.pkl `
  --max_steps 30 --validate_fea

# engineering layout
python -m src.reinforcement.sb3.optimize `
  --model_path results\sb3\0527_1633\final_model.zip `
  --layout_strategy engineering `
  --L 8.0 --W 3.6 --H 3.6 `
  --pressure 30000 `
  --eval_mode surrogate `
  --surrogate_path src\data\engineering_model.pkl `
  --max_steps 30 `
  --seed 42 `
  --validate_fea


#########  optimize with GA #########
.\.venv\Scripts\python.exe scripts\run_ga.py `
  --layout-strategy engineering `
  --eval_mode fea `
  --L 8 --W 3.6 --H 3.6 --pressure 30000 `
  --pop_size 32 `
  --generations 10 `
  --n_jobs 4 `
  --grouped `
  --grouping-samples 12 `
  --grouping-n-jobs 4 `
  --grouping-mesh-size 0.5 `
  --group-frame 2 `
  --group-main 8 `
  --group-sec 7 `
  --group-brace 14