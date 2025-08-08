import argparse
import os
import random  # Added for replay buffer sampling
import shutil  # Added for directory operations
import sys
from collections import deque  # Added for replay buffer
from datetime import datetime
from time import time
from typing import Union

import numpy as np
import torch
from torch_geometric.data import Batch, Data  # Added for batching experiences

from src.reinforcement.dqn_agent import DQNAgent
from src.reinforcement.dqn_size_agent import DQNSizeAgent
from src.reinforcement.dqn_spacing_agent import DQNSpacingAgent
from src.reinforcement.epsilon_scheduler import EpsilonScheduler
from src.reinforcement.ppo_agent import Memory, PPOAgent
from src.reinforcement.rl_agent import DDPGAgent  # Added
from src.reinforcement.structural_env import StructuralEnv  # Added
from src.structure.prototypes import get_prototype_configs
from src.structure.structural_model import BeamType
from src.utils.constants import *
from src.utils.logger import Logger
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


def get_agent(config: dict, device: torch.device, sample_state: Data):
    if config["rl_params"]["agent"] == "dqn":
        agent_cls = DQNSizeAgent if config["env_config"]["opt_type"] == "size" else DQNSpacingAgent
        agent = agent_cls(
            state_dim=sample_state.x.size(1),
            glb_feat_dim=sample_state.global_features.size(1),
            device=device,
            gamma=config["rl_params"]["gamma"],
            lr=config["rl_params"]["lr"],
            target_update=config["rl_params"]["update_interval"],
        )
    elif config["rl_params"]["agent"] == "ddpg":
        hebg_config = config["hebg_config"]
        hebg_config["in_channels"] = sample_state.x.size(1)
        agent = DDPGAgent(
            hebg_config=hebg_config,
            action_dim=config["env_config"]["action_dim"],
            actor_lr=config["rl_params"]["actor_lr"],
            critic_lr=config["rl_params"]["critic_lr"],
            gamma=config["rl_params"]["gamma"],
            sigma=config["rl_params"]["sigma"],
            tau=config["rl_params"]["tau"],
            hidden_dim_fc=config["rl_params"]["ddpg_fc_hidden_dim"],
            device=device,
        )
    elif config["rl_params"]["agent"] == "ppo":
        agent = PPOAgent(
            state_dim=sample_state.x.size(1),
            glb_feat_dim=sample_state.global_features.size(1),
            device=device,
            lr=config["rl_params"]["lr"],
            gamma=config["rl_params"]["gamma"],
            k_epochs=config["rl_params"]["k_epochs"],
            eps_clip=config["rl_params"]["eps_clip"],
            gae_lambda=config["rl_params"]["gae_lambda"],
        )
    else:
        raise ValueError(f"Invalid agent type: {config['rl_params']['agent']}")
    return agent


def evaluate_on_prototypes(
    agent: Union[DQNSpacingAgent, DQNSizeAgent, PPOAgent],
    config: dict,
    device: torch.device,
    logger: Logger,
):
    """
    用当前agent评估所有原型结构，返回平均reward和详细表现。
    """

    prototypes = get_prototype_configs(config)
    results = []
    total_reward = 0
    agent.set_mode(eval=True)
    memory = Memory()
    for proto_config in prototypes:
        print("=" * 50)
        print(f"Evaluating prototype: {proto_config['name']}")
        print("=" * 50)
        env = StructuralEnv(initial_params_config=proto_config, logger=logger)
        state_pyg, info = env.reset()
        episode_reward = 0
        done = False
        state_pyg = state_pyg.to(device)
        while not done:
            with torch.no_grad():
                # DDPGAgent和DQNAgent都支持training参数
                if isinstance(agent, PPOAgent):
                    action = agent.select_action(state_pyg, memory)
                else:
                    action = agent.select_action(state_pyg, 0.0)
            next_state_pyg, reward, terminated, truncated, info = env.step(action)
            env.render()
            print()
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
        print(f"Prototype: {proto_config['name']} evaluated.\n\n")
    avg_reward = total_reward / len(prototypes)
    agent.set_mode(eval=False)
    return avg_reward, results


def train(args, config):
    """训练强化学习智能体"""
    print(f"开始训练 DDPG 智能体，设备: {args.device}")
    device = torch.device(args.device)
    ts = datetime.now().strftime("%m%d_%H%M")
    logger = Logger(f"rewards_{ts}.log")

    rl_params = config["rl_params"]

    env = StructuralEnv(initial_params_config=args.config, logger=logger)
    epsilon_scheduler = EpsilonScheduler()

    # Determine HEBG in_channels from a sample observation
    sample_state, _ = env.reset()

    agent = get_agent(config, device, sample_state)
    is_ppo = isinstance(agent, PPOAgent)
    if is_ppo:
        memory = Memory()
    else:
        replay_buffer = deque(maxlen=rl_params["buffer_size"])

    if args.resume_path:
        agent.load_model(args.resume_path)
        print(f"Model resumed from {args.resume_path}")

    # For plotting and saving metrics
    metrics_history = {
        "rewards": [],
        "max_displacements": [],
        "max_stresses": [],
        "model_volumes": [],
        "volume_efficiencies": [],
        "steps": [],
        "episode_numbers": [],
    }
    plot_save_dir = os.path.join(args.model_save_path, "plots")
    os.makedirs(plot_save_dir, exist_ok=True)
    model_base_name = rl_params["agent"]
    global_step = 0

    best_eval_avg_reward = -float("inf")
    for episode in range(args.max_episodes):
        state_pyg, info_reset = env.reset(options={"random_structure": {"random_keys": ["pressure"]}})
        step = 0
        episode_reward = 0
        done = False

        state_pyg = state_pyg.to(device)

        while not done:
            step += 1
            global_step += 1
            if is_ppo:
                action = agent.select_action(state_pyg, memory)
            else:
                action = agent.select_action(state_pyg, epsilon_scheduler.get_epsilon())

            next_state_pyg, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            episode_reward += reward

            if is_ppo:
                memory.rewards.append(reward)
                memory.dones.append(done)
                next_state_pyg = next_state_pyg.to(device)

            else:
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

        if is_ppo:
            _, last_value = agent.policy_old.forward(next_state_pyg)
            agent.update(memory, last_value, done)
            memory.clear_memory()

        # Store metrics for this episode
        metrics_history["rewards"].append(episode_reward)
        metrics_history["max_displacements"].append(info["max_displacement_mm"])
        metrics_history["max_stresses"].append(info["max_stress"])
        metrics_history["model_volumes"].append(info["normalized_volume"])
        metrics_history["volume_efficiencies"].append(info["volume_efficiency"])
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

        # 评估当前模型在prototypes上的表现
        if (episode + 1) % rl_params["eval_interval"] == 0:
            eval_avg_reward, eval_results = evaluate_on_prototypes(agent, config, device, logger)
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
        if (episode + 1) % rl_params["save_interval"] == 0:
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
                volume_efficiencies=np.array(metrics_history["volume_efficiencies"]),
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
        volume_efficiencies=np.array(metrics_history["volume_efficiencies"]),
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
    ts = datetime.now().strftime("%m%d_%H%M")
    logger = Logger(f"rewards_{ts}.log")

    model_base_name = config["rl_params"]["agent"]

    if args.resume_path:
        final_model_load_path = args.resume_path
    elif args.model_save_path:
        final_model_load_path = os.path.join(args.model_save_path, f"{model_base_name}_final.pth")
    else:
        raise ValueError("No model path provided")

    if not os.path.exists(final_model_load_path):
        print(f"Error: Trained final model not found at {final_model_load_path}. Cannot run test.")
        return

    env = StructuralEnv(initial_params_config=args.config, logger=logger)
    sample_state, _ = env.reset()

    agent = get_agent(config, device, sample_state)
    agent.load_model(final_model_load_path)

    evaluate_on_prototypes(agent, config, device, logger)


def main():
    args = parse_args()

    # set seed globally
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
