# train model
python main.py --mode train --config config.yaml  --max_episodes 100 --model_save_path ./results/models/0617_2

# test model
python main.py --mode test --config config.yaml --model_save_path ./results/models/0617_2