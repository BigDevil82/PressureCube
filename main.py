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

# Default configuration for agent actions and environment interaction
# This should ideally be part of your main config.yaml or a separate settings file
DEFAULT_ENV_ACTION_CONFIG = {
    "parameter_action_mapping": {
        "main_beam_spacing": {"action_idx": 0, "type": "continuous", "bounds": [1.0, 3.0], "module": 0.1},
        "secondary_beam_spacing": {
            "action_idx": 1,
            "type": "continuous",
            "bounds": [0.5, 2.0],
            "module": 0.1,
        },
        "brace_spacing": {"action_idx": 2, "type": "continuous", "bounds": [0.5, 2.0], "module": 0.1},
        "frame_section_index": {
            "action_idx": 3,
            "type": "discrete",
            "num_options": 3,
        },  # Example: 3 choices for frame sections
        "main_beam_section_index": {
            "action_idx": 4,
            "type": "discrete",
            "num_options": 5,
        },  # Example: 5 choices for main beam sections
        "secondary_beam_section_index": {"action_idx": 5, "type": "discrete", "num_options": 5},
        "brace_section_index": {"action_idx": 6, "type": "discrete", "num_options": 4},
    },
    "max_steps_per_episode": 100,  # Max steps per episode during training
    "replay_buffer_size": 10000,
    "batch_size": 32,
    "min_replay_buffer_fill": 200,  # Start training after buffer has this many samples
}

# Placeholder for HEBG configuration - ideally from config.yaml or auto-detected
DEFAULT_HEBG_CONFIG = {
    # 'in_channels' will be determined from the environment observation
    "hidden_channels": 64,  # Example
    "out_channels": 128,  # Example, this is the state dim for DDPG FC layers
    "num_beam_types": 4,
}


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


def plot_training_progress(episode_num, metrics_history, save_dir):
    """Plots and saves the training progress for various metrics."""
    if not os.path.exists(save_dir):
        os.makedirs(save_dir, exist_ok=True)

    epochs = list(range(1, episode_num + 1))

    fig, axs = plt.subplots(2, 2, figsize=(15, 15), sharex=True)

    # Plot Episode Rewards
    axs[0, 0].plot(
        epochs, metrics_history["rewards"], label="Episode Reward", color="blue", marker="o", linestyle="-"
    )
    axs[0, 0].set_ylabel("Episode Return")
    axs[0, 0].set_title("Training Progress")
    axs[0, 0].legend()
    axs[0, 0].grid(True)

    # Plot Max Displacement
    axs[0, 1].plot(
        epochs,
        metrics_history["max_displacements"],
        label="Max Displacement (mm)",
        color="red",
        marker="o",
        linestyle="-",
    )
    axs[0, 1].set_ylabel("Max Displacement (mm)")
    # axs[1].set_yscale('log') # If values vary wildly
    axs[0, 1].legend()
    axs[0, 1].grid(True)

    # Plot Max Stress
    axs[1, 0].plot(
        epochs,
        metrics_history["max_stresses"],
        label="Max Stress (MPa)",
        color="green",
        marker="o",
        linestyle="-",
    )
    axs[1, 0].set_ylabel("Max Stress (MPa)")
    axs[1, 0].legend()
    axs[1, 0].grid(True)

    # Plot Model Volume
    axs[1, 1].plot(
        epochs,
        metrics_history["model_volumes"],
        label="Model Volume (m^3)",
        color="purple",
        marker="o",
        linestyle="-",
    )
    axs[1, 1].set_ylabel("Model Volume (m^3)")
    axs[1, 1].set_xlabel("Episode")
    axs[1, 1].legend()
    axs[1, 1].grid(True)

    plt.tight_layout()
    plot_filename = os.path.join(save_dir, f"training_progress.png")
    plt.savefig(plot_filename)
    plt.close(fig)  # Close the figure to free memory
    # print(f"Training progress plot saved to {plot_filename}")


def analyze_structure(model):
    """分析结构性能"""
    analyzer = OpenSeesAnalyzer()
    analyzer.setup_opensees_model(model)
    analyzer.apply_gravity_loads(model)
    static_success = analyzer.run_static_analysis()

    results = {"static_success": static_success}
    constraints = {}
    if static_success:
        displacements = analyzer.get_node_displacements()
        max_disp = 0
        if displacements:
            for disp_val in displacements.values():
                max_disp = max(max_disp, np.max(np.abs(disp_val[:3])))
        results["max_displacement_mm"] = max_disp
        # Simplified constraint check based on max displacement
        constraints["max_displacement_check"] = max_disp < 20.0  # Example: 20mm limit
        satisfied = constraints["max_displacement_check"]
    else:
        results["max_displacement_mm"] = float("inf")
        constraints["max_displacement_check"] = False
        satisfied = False
    return satisfied, constraints, results


def train(args, config):
    """训练强化学习智能体"""
    print(f"开始训练 DDPG 智能体，设备: {args.device}")
    device = torch.device(args.device)

    # Load env_action_config and hebg_config (using defaults for now)
    # In a real setup, these might come from the main config file or be more elaborate
    env_action_config = config.get("env_action_config", DEFAULT_ENV_ACTION_CONFIG)
    hebg_config = config.get("hebg_config", DEFAULT_HEBG_CONFIG)
    rl_params = config.get("rl_params", {})

    env = StructuralEnv(initial_params_config_path=args.config, env_action_config=env_action_config)

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

    replay_buffer = deque(maxlen=rl_params["buffer_size"])
    episode_rewards = []

    # For plotting
    metrics_history = {"rewards": [], "max_displacements": [], "max_stresses": [], "model_volumes": []}
    plot_save_dir = os.path.join(args.model_save_path, "plots")
    # Ensure the base model save directory exists
    if not os.path.exists(args.model_save_path):
        os.makedirs(args.model_save_path, exist_ok=True)
    # Base name for the model, e.g., "ddpg_agent"
    model_base_name = "ddpg_agent"

    for episode in range(args.max_episodes):
        state_pyg, info_reset = env.reset()
        episode_reward = 0

        if state_pyg is None or not hasattr(state_pyg, "x") or state_pyg.x is None:
            print(f"Episode {episode+1}: Failed to get valid initial state. Skipping episode.")
            episode_rewards.append(0)  # Or handle as error
            continue
        state_pyg = state_pyg.to(device)

        for step in range(rl_params["max_steps_per_episode"]):
            start_time = time()
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
                # Create a placeholder for next_state if needed by buffer, or skip storing this transition
                done = True
                # Potentially use current state_pyg as next_state_pyg to avoid None in buffer,
                # but ensure done flag prevents Q-value bootstrapping from it.
                if next_state_pyg is None:
                    next_state_pyg = state_pyg.clone()  # Simplistic handling

            next_state_pyg = next_state_pyg.to(device)

            # Store experience (state, action, reward, next_state, done)
            # Action stored should be the one applied (action_np), ensure it's a tensor for batching
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
                # Critical: Ensure Batch.from_data_list is used for states and next_states
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

            # end_time = time()
            # # print(f"Episode {episode+1}, Step {step+1}, Time: {end_time - start_time:.2f} seconds")
            # duration = end_time - start_time

            episode_rewards.append(episode_reward)
            # Store metrics for plotting
            metrics_history["rewards"].append(episode_reward)
            metrics_history["max_displacements"].append(info.get("max_displacement_mm", float("nan")))
            metrics_history["max_stresses"].append(info.get("max_stress", float("nan")))
            metrics_history["model_volumes"].append(info.get("model_volume_m3", float("nan")))
            # Plot progress after each episode
            if (episode * rl_params["max_steps_per_episode"] + step + 1) % rl_params.get(
                "plot_interval", 1
            ) == 0:
                plot_training_progress(
                    episode * rl_params["max_steps_per_episode"] + step + 1, metrics_history, plot_save_dir
                )

        print(
            f"Episode {episode+1}/{args.max_episodes}, Reward: {episode_reward:.2f}, Steps: {step+1}, Info: {info.get('error', 'OK')}"
        )
        if info.get("max_displacement_mm") is not None:
            print(f"  Max displacement: {info['max_displacement_mm']:.3f} mm")

        # Save model periodically (example)
        if (episode + 1) % rl_params.get("save_interval", 100) == 0:
            periodic_save_path = os.path.join(
                args.model_save_path, f"{model_base_name}_episode_{episode+1}.pth"
            )
            agent.save_model(periodic_save_path)
            print(f"Model saved to {periodic_save_path} at episode {episode+1}")

    # Save the final model
    final_model_save_path = os.path.join(args.model_save_path, f"{model_base_name}_final.pth")
    agent.save_model(final_model_save_path)
    print(f"Final model saved to {final_model_save_path}")

    print("训练完成.")
    # Plot final combined metrics
    if metrics_history["rewards"]:
        plot_training_progress(args.max_episodes, metrics_history, plot_save_dir)
        print(f"Final training plot saved in {plot_save_dir}")


def test(args, config):
    """测试训练好的智能体"""
    print(f"开始测试 DDPG 智能体，设备: {args.device}")
    device = torch.device(args.device)

    env_action_config = config.get("env_action_config", DEFAULT_ENV_ACTION_CONFIG)
    rl_params = config.get("rl_params", {})
    # Load HEBG config that was used during training
    # This requires the saved model to include hebg_config

    model_dir = os.path.dirname(args.model_save_path)
    model_base_name = os.path.splitext(os.path.basename(args.model_save_path))[0]
    final_model_load_path = os.path.join(model_dir, f"{model_base_name}_final.pth")

    if not os.path.exists(final_model_load_path):
        print(f"Error: Trained final model not found at {final_model_load_path}. Cannot run test.")
        # Fallback: try to load a model with the exact name if _final doesn't exist (old behavior)
        if os.path.exists(args.model_save_path):
            print(f"Attempting to load {args.model_save_path} instead.")
            final_model_load_path = args.model_save_path
        else:
            print(f"Also, {args.model_save_path} not found. Exiting test.")
            return

    checkpoint = torch.load(final_model_load_path, map_location=device)
    hebg_config = checkpoint.get("hebg_config")
    if hebg_config is None:
        print("Error: hebg_config not found in checkpoint. Using default, which might be incorrect.")
        hebg_config = config.get("hebg_config", DEFAULT_HEBG_CONFIG)
        # Try to get in_channels if env can be set up
        try:
            test_env_for_channels = StructuralEnv(
                initial_params_config_path=args.config, env_action_config=env_action_config
            )
            sample_state, _ = test_env_for_channels.reset()
            if sample_state and hasattr(sample_state, "x") and sample_state.x is not None:
                hebg_config["in_channels"] = sample_state.x.size(1)
            else:  # Fallback if cannot determine
                print(
                    "Warning: Could not determine in_channels for HEBG during test. Model might not load correctly."
                )
                # hebg_config['in_channels'] = some_default_if_absolutely_necessary
        except Exception as e:
            print(f"Error setting up temp env for HEBG config: {e}")

    env = StructuralEnv(initial_params_config_path=args.config, env_action_config=env_action_config)
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
        episode_reward = 0
        if state_pyg is None or not hasattr(state_pyg, "x") or state_pyg.x is None:
            print(f"Test Episode {episode+1}: Failed to get valid initial state. Skipping.")
            continue
        state_pyg = state_pyg.to(device)

        for step in range(rl_params["max_steps_per_episode"]):
            with torch.no_grad():  # Ensure no gradients are computed
                action_tensor = agent.select_action(state_pyg)
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
    # Merge or override with default env_action_config and hebg_config if not in main config
    if "env_action_config" not in config:
        config["env_action_config"] = DEFAULT_ENV_ACTION_CONFIG
    if "hebg_config" not in config:
        config["hebg_config"] = DEFAULT_HEBG_CONFIG
    # Potentially load other rl_params from config if they exist
    if "rl_params" not in config:
        config["rl_params"] = {}

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
        try:
            satisfied, constraints, results = analyze_structure(model)
            print(f"结构约束满足: {satisfied}")
            for constraint_name, is_satisfied in constraints.items():
                print(f"  {constraint_name}: {'满足' if is_satisfied else '不满足'}")
        except Exception as e:
            print(f"结构分析失败: {e}")
            # import traceback; traceback.print_exc()


if __name__ == "__main__":
    main()  # Reinstate main function call
    # from src.graph.hebg import test_HEBG # Keep this commented or remove
    # test_HEBG()
