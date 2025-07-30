# train model
python main.py --mode train --config config.yaml  --max_episodes 100 --model_save_path ./results/models/0617_2

# test model
# test spacing optimization
python main.py --mode test --config ./configs/opt_spacing_config.yaml --resume_path ./results/models/0728_3/dqn_final.pth

# test size optimization
python main.py --mode test --config ./configs/opt_size_config.yaml --resume_path ./results/models/0729_3/best_model.pth