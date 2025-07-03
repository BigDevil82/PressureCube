import argparse
import os
import random  # Added for replay buffer sampling
import shutil  # Added for directory operations
import sys
from collections import deque  # Added for replay buffer
from time import time

import matplotlib.pyplot as plt  # Added for plotting
import numpy as np
import torch
from torch_geometric.data import Batch  # Added for batching experiences

from src.analysis.opensees_analysis import OpenSeesAnalyzer
from src.reinforcement.rl_agent import DDPGAgent  # Added
from src.reinforcement.structural_env import StructuralEnv  # Added
from src.structure.structural_model import BeamType
from src.utils.constants import *
from src.utils.model_utils import StructureVisualizer, create_structural_model, load_config


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


def plot_training_progress(metrics_history, save_dir):
    """Plots and saves the training progress for various metrics."""
    if not os.path.exists(save_dir):
        os.makedirs(save_dir, exist_ok=True)

    # Create x-axis values (episode numbers)
    episodes = list(range(1, len(metrics_history["rewards"]) + 1))

    fig, axs = plt.subplots(2, 2, figsize=(15, 15), sharex=True)

    # Plot Episode Rewards
    axs[0, 0].plot(
        episodes, metrics_history["rewards"], label="Episode Reward", color="blue", marker="o", linestyle="-"
    )
    axs[0, 0].set_ylabel("Episode Return")
    axs[0, 0].set_title("Cumulative Rewards per Episode")
    axs[0, 0].legend()
    axs[0, 0].grid(True)

    # Plot Max Displacement
    axs[0, 1].plot(
        episodes,
        metrics_history["max_displacements"],
        label="Max Displacement",
        color="red",
        marker="o",
        linestyle="-",
    )
    axs[0, 1].set_ylabel("Max Displacement (mm)")
    axs[0, 1].set_title("Maximum Displacement per Episode")
    axs[0, 1].legend()
    axs[0, 1].grid(True)
    # plot displacement constraint line
    axs[0, 1].axhline(y=14.4, color="red", linestyle="--", label="Displacement Constraint")

    # Plot Max Stress
    axs[1, 0].plot(
        episodes,
        metrics_history["max_stresses"],
        label="Max Stress",
        color="green",
        marker="o",
        linestyle="-",
    )
    axs[1, 0].set_ylabel("Max Stress (MPa)")
    axs[1, 0].set_xlabel("Episode")
    axs[1, 0].set_title("Maximum Stress per Episode")
    axs[1, 0].legend()
    axs[1, 0].grid(True)
    # plot stress constraint line
    axs[1, 0].axhline(y=355, color="red", linestyle="--", label="Stress Constraint")

    # Plot Model Volume
    axs[1, 1].plot(
        episodes,
        metrics_history["model_volumes"],
        label="Model Volume",
        color="purple",
        marker="o",
        linestyle="-",
    )
    axs[1, 1].set_ylabel("Model Volume (m³)")
    axs[1, 1].set_xlabel("Episode")
    axs[1, 1].set_title("Structure Volume per Episode")
    axs[1, 1].legend()
    axs[1, 1].grid(True)

    plt.tight_layout()
    plot_filename = os.path.join(save_dir, f"training_progress.png")
    plt.savefig(plot_filename)
    plt.close(fig)  # Close the figure to free memory


def train(args, config):
    """训练强化学习智能体"""
    print(f"开始训练 DDPG 智能体，设备: {args.device}")
    device = torch.device(args.device)

    env_action_config = config["env_action_config"]
    hebg_config = config["hebg_config"]
    rl_params = config["rl_params"]

    env = StructuralEnv(
        initial_params_config_path=args.config, env_action_config=env_action_config, seed=args.seed
    )

    # Determine HEBG in_channels from a sample observation
    sample_state, _ = env.reset()
    if sample_state is None or not hasattr(sample_state, "x") or sample_state.x is None:
        print("Error: Could not get a valid initial state from environment to determine HEBG in_channels.")
        return
    hebg_config["in_channels"] = sample_state.x.size(1)

    print(f"HEBG Config determined: {hebg_config}")

    agent = DDPGAgent(
        hebg_config=hebg_config,
        action_dim=env_action_config["action_dim"],
        actor_lr=rl_params["actor_lr"],
        critic_lr=rl_params["critic_lr"],
        gamma=rl_params["gamma"],
        sigma=rl_params["sigma"],
        tau=rl_params["tau"],
        hidden_dim_fc=rl_params["ddpg_fc_hidden_dim"],
        device=device,
    )

    if args.resume_path:
        agent.load_model(args.resume_path)
        print(f"Model resumed from {args.resume_path}")

    replay_buffer = deque(maxlen=rl_params["buffer_size"])
    episode_rewards = []

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
    model_base_name = "ddpg_agent"
    global_step = 0

    for episode in range(args.max_episodes):
        state_pyg, info_reset = env.reset()
        episode_reward = 0

        if state_pyg is None or not hasattr(state_pyg, "x") or state_pyg.x is None:
            print(f"Episode {episode+1}: Failed to get valid initial state. Skipping episode.")
            episode_rewards.append(0)  # Or handle as error
            continue
        state_pyg = state_pyg.to(device)

        for step in range(rl_params["max_steps_per_episode"]):
            global_step += 1
            action_tensor = agent.select_action(state_pyg)  # state_pyg is a single Data object
            action_np = (
                action_tensor.cpu().detach().numpy().squeeze(0)
            )  # DDPG actor might output [1, action_dim]

            next_state_pyg, reward, terminated, truncated, info = env.step(action_np)
            done = terminated or truncated
            episode_reward += reward

            if next_state_pyg is None or not hasattr(next_state_pyg, "x") or next_state_pyg.x is None:
                print(
                    f"Warning: Episode {episode+1}, Step {step+1}: Received invalid next_state. Treating as done."
                )
                done = True
                if next_state_pyg is None:
                    next_state_pyg = state_pyg.clone()  # Simplistic handling

            next_state_pyg = next_state_pyg.to(device)

            # Store experience (state, action, reward, next_state, done)
            replay_buffer.append(
                (
                    state_pyg,
                    torch.tensor(action_np, dtype=torch.float32).to(device),
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
            if done:
                break

            # Store metrics for this episode
            metrics_history["rewards"].append(episode_reward)
            metrics_history["max_displacements"].append(info.get("max_displacement_mm", float("nan")))
            metrics_history["max_stresses"].append(info.get("max_stress", float("nan")))
            metrics_history["model_volumes"].append(info.get("model_volume_m3", float("nan")))
            metrics_history["steps"].append(step + 1)  # Store number of steps taken
            metrics_history["episode_numbers"].append(episode + 1)

            # Plot progress after each episode
            if global_step % rl_params.get("plot_interval", 10) == 0:
                plot_training_progress(metrics_history, plot_save_dir)

        print(
            f"Episode {episode+1}/{args.max_episodes}, Episode Return: {episode_reward:.2f}, Steps: {step+1}, Info: {info.get('error', 'OK')}"
        )
        env.render()
        print()

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
        plot_training_progress(metrics_history, plot_save_dir)
        print(f"Final training plot saved in {plot_save_dir}")


def test(args, config):
    """测试训练好的智能体"""
    print(f"开始测试 DDPG 智能体，设备: {args.device}")
    device = torch.device(args.device)

    env_action_config = config["env_action_config"]
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

    env = StructuralEnv(
        initial_params_config_path=args.config, env_action_config=env_action_config, seed=args.seed
    )
    action_dim = env_action_config["action_dim"]

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
    # agent.critic.load_state_dict(checkpoint['critic_state_dict']) # Critic not strictly needed for testing actions
    agent.actor.eval()  # Set to evaluation mode

    num_test_episodes = rl_params.get("num_test_episodes", 1)
    all_episode_rewards = []

    for episode in range(num_test_episodes):
        state_pyg, _ = env.reset()
        print("initial state: ")
        env.render()
        print("-" * 100 + "\n")
        episode_reward = 0
        if state_pyg is None or not hasattr(state_pyg, "x") or state_pyg.x is None:
            print(f"Test Episode {episode+1}: Failed to get valid initial state. Skipping.")
            continue
        state_pyg = state_pyg.to(device)

        for step in range(rl_params["max_steps_per_episode"]):
            with torch.no_grad():  # Ensure no gradients are computed
                action_tensor = agent.select_action(state_pyg, training=False)
            action_np = action_tensor.cpu().numpy().squeeze(0)

            next_state_pyg, reward, terminated, truncated, info = env.step(action_np)
            done = terminated or truncated
            episode_reward += reward

            print(
                f"  Test Step {step+1}: Action: {np.round(action_np, 2)}, Reward: {reward:.2f}, MaxDisp: {info.get('max_displacement_mm', -1):.2f} mm"
            )
            env.render()  # Optionally print parameters

            if next_state_pyg is None or not hasattr(next_state_pyg, "x") or next_state_pyg.x is None:
                done = True  # End episode if next state is invalid
            else:
                state_pyg = next_state_pyg.to(device)

            if done:
                break
        all_episode_rewards.append(episode_reward)
        print(
            f"Test Episode {episode+1}/{num_test_episodes}, Total Reward: {episode_reward:.2f}, Steps: {step+1}"
        )
        if info.get("error"):
            print(f"  Episode ended with error: {info['error']}")

    print("\nTesting completed.")
    print(f"Average reward over {num_test_episodes} episodes: {np.mean(all_episode_rewards):.2f}")


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
