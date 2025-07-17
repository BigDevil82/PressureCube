import argparse
import os
import random  # Added for replay buffer sampling
import shutil  # Added for directory operations
import sys
from collections import deque  # Added for replay buffer
from time import time
from typing import Union

import numpy as np
import torch
from torch_geometric.data import Batch  # Added for batching experiences

from src.reinforcement.dqn_agent import DQNAgent
from src.reinforcement.rl_agent import DDPGAgent  # Added
from src.reinforcement.structural_env import StructuralEnv  # Added
from src.structure.prototypes import get_prototype_configs
from src.structure.structural_model import BeamType
from src.utils.constants import *
from src.utils.model_utils import StructureVisualizer, create_structural_model, load_config
from src.utils.util import plot_progress


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description="高海拔增压建筑智能设计")
    parser.add_argument("--config", type=str, default="config.yaml", help="配置文件路径")
    parser.add_argument(
        "--mode", type=str, choices=["train", "test", "visualize"], default="train", help="运行模式"
    )
    parser.add_argument(
        "--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="设备: cuda/cpu"
    )
    parser.add_argument("--seed", type=int, default=42, help="随机种子")
    parser.add_argument("--resume_path", type=str, default=None, help="恢复训练路径")
    parser.add_argument(
        "--max_episodes", type=int, default=500, help="最大训练回合数"
    )  # Reduced for quicker test runs
    parser.add_argument(
        "--model_save_path",
        type=str,
        default="./results/models",
        help="Path to save/load trained model",
    )
    return parser.parse_args()


def evaluate_on_prototypes(agent: Union[DDPGAgent, DQNAgent], config: dict, device: torch.device, seed: int):
    """
    用当前agent评估所有原型结构，返回平均reward和详细表现。
    """

    prototypes = get_prototype_configs(
        config_path=config["config_path"] if "config_path" in config else "config.yaml"
    )
    results = []
    total_reward = 0
    agent.set_eval_mode()
    for proto_config in prototypes:
        env = StructuralEnv(initial_params_config=proto_config, seed=seed)
        state_pyg, info = env.reset()
        episode_reward = 0
        done = False
        state_pyg = state_pyg.to(device)
        while not done:
            with torch.no_grad():
                # DDPGAgent和DQNAgent都支持training参数
                action = agent.select_action(state_pyg, 0.0)
            next_state_pyg, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            episode_reward += reward
            state_pyg = next_state_pyg.to(device)
        total_reward += episode_reward
        analysis_params = config["analysis_params"]
        is_feasible = (
            info["max_displacement_mm"] <= analysis_params["max_disp"]
            and info["max_stress"] <= analysis_params["max_stress"]
        )
        results.append(
            {
                "prototype": proto_config["name"],
                "reward": episode_reward,
                "final_displacement": info["max_displacement_mm"],
                "final_stress": info["max_stress"],
                "final_volume": info["model_volume_m3"],
                "is_feasible": is_feasible,
            }
        )
    avg_reward = total_reward / len(prototypes)
    return avg_reward, results


def train(args, config):
    """训练强化学习智能体"""
    print(f"开始训练 DDPG 智能体，设备: {args.device}")
    device = torch.device(args.device)

    env_config = config["env_config"]
    hebg_config = config["hebg_config"]
    rl_params = config["rl_params"]

    env = StructuralEnv(initial_params_config=args.config, seed=args.seed)

    # Determine HEBG in_channels from a sample observation
    sample_state, _ = env.reset()

    hebg_config["in_channels"] = sample_state.x.size(1)

    if rl_params["agent"] == "ddpg":
        agent = DDPGAgent(
            hebg_config=hebg_config,
            action_dim=env_config["action_dim"],
            actor_lr=rl_params["actor_lr"],
            critic_lr=rl_params["critic_lr"],
            gamma=rl_params["gamma"],
            sigma=rl_params["sigma"],
            tau=rl_params["tau"],
            hidden_dim_fc=rl_params["ddpg_fc_hidden_dim"],
            device=device,
        )
    else:
        agent = DQNAgent(
            state_dim=sample_state.x.size(1),
            action_dim=env_config["action_dim"],
            device=device,
            gamma=rl_params["gamma"],
            lr=rl_params["lr"],
            target_update=rl_params["update_interval"],
        )

    if args.resume_path:
        agent.load_model(args.resume_path)
        print(f"Model resumed from {args.resume_path}")

    replay_buffer = deque(maxlen=rl_params["buffer_size"])

    # For plotting and saving metrics
    metrics_history = {
        "rewards": [],
        "max_displacements": [],
        "max_stresses": [],
        "model_volumes": [],
        "steps": [],
        "episode_numbers": [],
    }
    plot_save_dir = os.path.join(args.model_save_path, "plots")
    os.makedirs(plot_save_dir, exist_ok=True)
    model_base_name = rl_params["agent"]
    global_step = 0

    best_eval_avg_reward = -float("inf")
    for episode in range(args.max_episodes):
        state_pyg, info_reset = env.reset(options={"random_structure": {"random_keys": ["length"]}})
        current_frame_length = info_reset["frame_length"]
        step = 0
        episode_reward = 0
        done = False

        state_pyg = state_pyg.to(device)

        while not done:
            step += 1
            global_step += 1
            action = agent.select_action(state_pyg, 0.01)  # state_pyg is a single Data object
            # DDPG actor might output [1, action_dim]
            # action_np = action_tensor.cpu().detach().numpy().squeeze(0)

            next_state_pyg, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            episode_reward += reward

            next_state_pyg = next_state_pyg.to(device)

            # Store experience (state, action, reward, next_state, done)
            replay_buffer.append(
                (
                    state_pyg,
                    torch.tensor(action, dtype=torch.long).to(device),
                    torch.tensor([reward], dtype=torch.float32).to(device),
                    next_state_pyg,
                    torch.tensor([done], dtype=torch.float32).to(device),
                )
            )

            if len(replay_buffer) >= rl_params["min_replay_buffer_fill"]:
                experiences = random.sample(replay_buffer, rl_params["batch_size"])

                states_list = [e[0] for e in experiences]
                actions_tensor = torch.stack(
                    [e[1] for e in experiences]
                )  # Stack directly if actions are already [action_dim]
                rewards_tensor = torch.stack([e[2] for e in experiences])
                next_states_list = [e[3] for e in experiences]
                dones_tensor = torch.stack([e[4] for e in experiences])

                # Batch PyG Data objects
                state_batch_pyg = Batch.from_data_list(states_list).to(device)
                next_state_batch_pyg = Batch.from_data_list(next_states_list).to(device)

                batch_dict = {
                    "state": state_batch_pyg,
                    "action": actions_tensor,
                    "reward": rewards_tensor,
                    "next_state": next_state_batch_pyg,
                    "done": dones_tensor,
                }
                agent.update(batch_dict)

            state_pyg = next_state_pyg

        # Store metrics for this episode
        metrics_history["rewards"].append(episode_reward)
        metrics_history["max_displacements"].append(info["max_displacement_mm"])
        metrics_history["max_stresses"].append(info["max_stress"])
        metrics_history["model_volumes"].append(info["model_volume_m3"] / current_frame_length)
        metrics_history["steps"].append(step)  # Store number of steps taken
        metrics_history["episode_numbers"].append(episode + 1)

        # Plot progress after each episode
        if episode % rl_params.get("plot_interval", 10) == 0:
            plot_progress(metrics_history, plot_save_dir, filename="training_progress.png")

        print(
            f"Episode {episode+1}/{args.max_episodes}, Episode Return: {episode_reward:.2f}, Steps: {step}, Info: {info.get('error', 'OK')}"
        )
        env.render()
        print()

        # 每10个episode评估一次当前模型在prototypes上的表现
        if (episode + 1) % rl_params["eval_interval"] == 0:
            eval_avg_reward, eval_results = evaluate_on_prototypes(agent, config, device, args.seed)
            print(f"[Prototype Eval] Episode {episode+1}: avg_reward={eval_avg_reward:.2f}")
            for res in eval_results:
                print(
                    f"  原型: {res['prototype']} | reward: {res['reward']:.2f} | displacement: {res['final_displacement']:.2f} | stress: {res['final_stress']:.2f} | feasible: {'是' if res['is_feasible'] else '否'}"
                )
            # 若当前模型优于历史最佳，则保存
            if eval_avg_reward > best_eval_avg_reward:
                best_eval_avg_reward = eval_avg_reward
                best_model_save_path = os.path.join(args.model_save_path, "best_model.pth")
                agent.save_model(best_model_save_path)
                print(f"[Prototype Eval] 新最佳模型已保存到: {best_model_save_path}")

        # Save model periodically
        if (episode + 1) % rl_params.get("save_interval", 100) == 0:
            periodic_save_path = os.path.join(
                args.model_save_path, f"{model_base_name}_episode_{episode+1}.pth"
            )
            agent.save_model(periodic_save_path)
            print(f"Model saved to {periodic_save_path} at episode {episode+1}")

            # Also save metrics data periodically
            metrics_save_path = os.path.join(plot_save_dir, "metrics.npz")
            np.savez(
                metrics_save_path,
                rewards=np.array(metrics_history["rewards"]),
                max_displacements=np.array(metrics_history["max_displacements"]),
                max_stresses=np.array(metrics_history["max_stresses"]),
                model_volumes=np.array(metrics_history["model_volumes"]),
                steps=np.array(metrics_history["steps"]),
                episode_numbers=np.array(metrics_history["episode_numbers"]),
            )
            print(f"Training metrics saved to {metrics_save_path}")

    # Save the final model
    final_model_save_path = os.path.join(args.model_save_path, f"{model_base_name}_final.pth")
    agent.save_model(final_model_save_path)
    print(f"Final model saved to {final_model_save_path}")

    # Save final metrics data
    final_metrics_save_path = os.path.join(plot_save_dir, "metrics.npz")
    np.savez(
        final_metrics_save_path,
        rewards=np.array(metrics_history["rewards"]),
        max_displacements=np.array(metrics_history["max_displacements"]),
        max_stresses=np.array(metrics_history["max_stresses"]),
        model_volumes=np.array(metrics_history["model_volumes"]),
        steps=np.array(metrics_history["steps"]),
        episode_numbers=np.array(metrics_history["episode_numbers"]),
    )
    print(f"Final training metrics saved to {final_metrics_save_path}")

    print("训练完成.")
    # Plot final combined metrics
    if metrics_history["rewards"]:
        plot_progress(metrics_history, plot_save_dir, "training_progress.png")
        print(f"Final training plot saved in {plot_save_dir}")


def test(args, config):
    """测试训练好的智能体"""
    print(f"开始测试 DDPG 智能体，设备: {args.device}")
    device = torch.device(args.device)

    env_config = config["env_config"]
    rl_params = config["rl_params"]

    if args.resume_path:
        final_model_load_path = args.resume_path
    else:
        model_base_name = "ddpg_agent"
        final_model_load_path = os.path.join(args.model_save_path, f"{model_base_name}_final.pth")

    if not os.path.exists(final_model_load_path):
        print(f"Error: Trained final model not found at {final_model_load_path}. Cannot run test.")
        return

    checkpoint = torch.load(final_model_load_path, map_location=device)
    hebg_config = checkpoint.get("hebg_config")

    env = StructuralEnv(initial_params_config=args.config, seed=args.seed)
    action_dim = env_config["action_dim"]

    agent = DDPGAgent(
        hebg_config=hebg_config,
        action_dim=action_dim,
        hidden_dim_fc=rl_params.get("ddpg_fc_hidden_dim", 256),
        actor_lr=rl_params.get("actor_lr", 1e-4),  # Added for completeness, though not strictly for inference
        critic_lr=rl_params.get("critic_lr", 1e-3),  # Added for completeness
        gamma=rl_params.get("gamma", 0.99),  # Added for completeness
        tau=rl_params.get("tau", 0.005),  # Added for completeness
        sigma=rl_params.get("sigma", 0.1),  # Added for completeness
        device=device,  # Added for completeness
    )
    agent.load_model(final_model_load_path)
    agent.actor.eval()  # Set to evaluation mode

    num_test_episodes = rl_params.get("num_test_episodes", 1)
    all_episode_rewards = []
    test_plot_dir = os.path.join(args.model_save_path, "test_plots")
    os.makedirs(test_plot_dir, exist_ok=True)

    for episode in range(num_test_episodes):
        state_pyg, info = env.reset()
        current_frame_length = info["frame_length"]
        print(f"\n--- Starting Test Episode {episode+1}/{num_test_episodes} ---")
        print("Initial state:")
        env.render()
        print("-" * 100)
        episode_reward = 0

        # Metrics for this test episode
        test_metrics_history = {
            "rewards": [],
            "max_displacements": [],
            "max_stresses": [],
            "model_volumes": [],
        }

        state_pyg = state_pyg.to(device)
        done = False
        step = 0
        while not done:
            step += 1
            with torch.no_grad():  # Ensure no gradients are computed
                action_tensor = agent.select_action(state_pyg, training=False)
            action_np = action_tensor.cpu().numpy().squeeze(0)

            next_state_pyg, reward, terminated, truncated, info = env.step(action_np)
            done = terminated or truncated
            episode_reward += reward

            env.render()  # Optionally print parameters
            print()

            # Collect metrics
            test_metrics_history["rewards"].append(reward)
            test_metrics_history["max_displacements"].append(info["max_displacement_mm"])
            test_metrics_history["max_stresses"].append(info["max_stress"])
            test_metrics_history["model_volumes"].append(info["model_volume_m3"] / current_frame_length)

            state_pyg = next_state_pyg.to(device)

        all_episode_rewards.append(episode_reward)
        print(f"--- Test Episode {episode+1} Finished ---")
        print(f"Total Reward: {episode_reward:.2f}, Steps: {step}")
        if info.get("error"):
            print(f"  Episode ended with error: {info['error']}")

        # Plot metrics for the completed episode
        plot_progress(
            test_metrics_history,
            test_plot_dir,
            filename=f"test_episode_{episode+1}_progress.png",
            x_axis_label="Step",
            main_title_prefix=f"Test Episode {episode+1}",
        )
        print(f"Test plot for episode {episode+1} saved in {test_plot_dir}")

    print("\nTesting completed.")
    avg_reward = np.mean(all_episode_rewards) if all_episode_rewards else 0
    print(f"Average reward over {num_test_episodes} episodes: {avg_reward:.2f}")


def main():
    args = parse_args()
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    random.seed(args.seed)  # For replay buffer sampling consistency

    config = load_config(args.config)

    if args.mode == "train":
        train(args, config)
    elif args.mode == "test":
        test(args, config)
    elif args.mode == "visualize":
        model = create_structural_model(config)
        print(f"节点数量: {len(model.points)}")
        print(f"梁构件数量: {len(model.beams)}")
        print(f"结构总质量: {model.calculate_mass():.2f} kg")
        beam_type_counts = {beam_type: 0 for beam_type in BeamType}
        beam_type_lengths = {beam_type: 0.0 for beam_type in BeamType}
        for beam in model.beams.values():
            beam_type_counts[beam.beam_type] += 1
            beam_type_lengths[beam.beam_type] += beam.length(model.points)
        for beam_type in BeamType:
            print(
                f"{beam_type.name} 梁数量: {beam_type_counts[beam_type]}, 总长度: {beam_type_lengths[beam_type]:.2f} m"
            )
        visualizer = StructureVisualizer(model)
        visualizer.show_all_views()


if __name__ == "__main__":
    main()  # Reinstate main function call
    # from src.graph.hebg import test_HEBG # Keep this commented or remove
    # test_HEBG()
