import argparse
import os
import random  # Added for replay buffer sampling
import shutil  # Added for directory operations
import sys
from collections import deque  # Added for replay buffer
from datetime import datetime
from time import time
from typing import List, Union

import numpy as np
import torch
from torch_geometric.data import Batch, Data  # Added for batching experiences

from src.reinforcement.dqn_agent import DQNAgent
from src.reinforcement.dqn_size_agent import DQNSizeAgent
from src.reinforcement.dqn_spacing_agent import DQNSpacingAgent
from src.reinforcement.epsilon_scheduler import EpsilonScheduler

# from src.reinforcement.ppo_agent import Memory, PPOAgent
from src.reinforcement.ppo_vector_agent import Memory as VectorMemory
from src.reinforcement.ppo_vector_agent import VectorPPOAgent
from src.reinforcement.rl_agent import DDPGAgent  # Added
from src.reinforcement.structural_env import StructuralEnv  # Added
from src.structure.prototypes import get_prototype_configs
from src.structure.structural_model import BeamType
from src.utils.constants import *
from src.utils.convergence_checker import ConvergenceChecker
from src.utils.logger import Logger
from src.utils.model_utils import StructureVisualizer, create_structural_model, load_config
from src.utils.util import plot_progress


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description="高海拔增压建筑智能设计")
    parser.add_argument("--config", type=str, default="config.yaml", help="配置文件路径")
    parser.add_argument(
        "--mode",
        type=str,
        choices=["train", "test", "visualize", "convergence"],
        default="train",
        help="运行模式",
    )
    parser.add_argument(
        "--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="设备: cuda/cpu"
    )
    parser.add_argument("--seed", type=int, default=42, help="随机种子")
    parser.add_argument("--resume_path", type=str, default=None, help="恢复训练路径")
    parser.add_argument("--max_episodes", type=int, default=500, help="最大训练回合数")
    parser.add_argument(
        "--model_save_path",
        type=str,
        default="./results/models",
        help="Path to save/load trained model",
    )
    parser.add_argument("--prototype", type=bool, default=False, help="是否评估原型结构")
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
        agent = VectorPPOAgent(
            state_dim=len(sample_state),
            action_dim=config["env_config"]["action_dim"],
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


def eval_on_config(
    agent: Union[DQNSpacingAgent, DQNSizeAgent, VectorPPOAgent],
    config: dict,
    device: torch.device,
    logger: Logger,
):
    t0 = time()
    agent.set_mode(eval=True)
    env = StructuralEnv(initial_params_config=config, logger=logger)
    state_pyg, info = env.reset()
    episode_reward = 0
    done = False
    state_pyg = state_pyg.to(device)
    while not done:
        with torch.no_grad():
            # DDPGAgent和DQNAgent都支持training参数
            if isinstance(agent, VectorPPOAgent):
                # 在评估时也使用动作掩码
                action_mask = None
                if config["env_config"]["opt_type"] == "size":
                    action_mask = env.get_action_mask_for_size()
                elif config["env_config"]["opt_type"] == "spacing":
                    action_mask = env.get_action_mask_for_spacing()
                elif config["env_config"]["opt_type"] == "both":
                    action_mask = env.get_action_mask_for_both()
                action = agent.select_action(state_pyg, None, action_mask)
            else:
                action = agent.select_action(state_pyg, 0.0)
        next_state_pyg, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        episode_reward += reward
        state_pyg = next_state_pyg.to(device)
        if done:
            env.render()
            print()
    analysis_params = config["analysis_params"]
    info["is_feasible"] = (
        info["max_displacement_mm"] <= analysis_params["max_disp"]
        and info["max_stress"] <= analysis_params["max_stress"]
    )
    info["episode_reward"] = episode_reward
    agent.set_mode(eval=False)
    t1 = time()
    print(f"Evaluation time: {t1 - t0:.2f} seconds")
    return info


def evaluate_on_prototypes(
    agent: Union[DQNSpacingAgent, DQNSizeAgent, VectorPPOAgent],
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

    for proto_config in prototypes:
        print("=" * 50)
        print(f"Evaluating prototype: {proto_config['name']}")
        print("=" * 50)
        res = eval_on_config(agent, proto_config, device, logger)
        episode_reward = res["episode_reward"]
        total_reward += episode_reward
        results.append(
            {
                "prototype": proto_config["name"],
                "reward": res["episode_reward"],
                "final_displacement": res["max_displacement_mm"],
                "final_stress": res["max_stress"],
                "final_volume": res["model_volume_m3"],
                "is_feasible": res["is_feasible"],
            }
        )
        print(f"Prototype: {proto_config['name']} evaluated.\n\n")
    avg_reward = total_reward / len(prototypes)
    return avg_reward, results


def evaluate_model_optimality(
    agent: Union[DQNSpacingAgent, DQNSizeAgent, VectorPPOAgent],
    config: dict,
    device: torch.device,
    logger: Logger,
    pressure_values: List[float] = None,
):
    """
    评估训练好的模型的最优性，计算与理论最优解的gap

    Args:
        agent: 训练好的智能体
        config: 配置字典
        device: 计算设备
        logger: 日志记录器
        pressure_values: 要测试的压力值列表
    """
    if pressure_values is None:
        pressure_values = [28.0, 30.0, 32.0, 35.0, 38.0]

    checker = ConvergenceChecker("config.yaml", logger)
    agent.set_mode(eval=True)

    # 首先进行穷举搜索找到理论最优解 (如果还没有的话)
    logger.info("正在进行穷举搜索以找到理论最优解...")
    optimal_reports = {}
    for pressure in pressure_values:
        logger.info(f"搜索压力 {pressure} kPa 的最优解...")
        report = checker.exhaustive_search(pressure=pressure, max_workers=4, save_results=True)
        optimal_reports[pressure] = report

    # 使用智能体求解并计算optimality gap
    logger.info("\n" + "=" * 60)
    logger.info("评估智能体性能...")
    logger.info("=" * 60)

    gaps = []
    for pressure in pressure_values:
        logger.info(f"\n测试压力: {pressure} kPa")

        # 准备环境
        test_config = config.copy()
        test_config["structural_params"]["pressure"] = pressure
        env = StructuralEnv(test_config, logger)

        # 智能体求解
        state_pyg, _ = env.reset()
        state_pyg = state_pyg.to(device)

        done = False
        step = 0
        max_steps = 50  # 防止无限循环
        memory = VectorMemory()

        while not done and step < max_steps:
            step += 1

            if isinstance(agent, VectorPPOAgent):
                action_mask = None
                if config["env_config"]["opt_type"] == "size":
                    action_mask = env.get_action_mask_for_size()
                elif config["env_config"]["opt_type"] == "spacing":
                    action_mask = env.get_action_mask_for_spacing()
                elif config["env_config"]["opt_type"] == "both":
                    action_mask = env.get_action_mask_for_both()
                action = agent.select_action(state_pyg, memory, action_mask)
            else:
                action = agent.select_action(state_pyg, 0.0)

            next_state_pyg, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            state_pyg = next_state_pyg.to(device)

        # 获取智能体解
        agent_solution = {
            "frame_section_index": env.current_params["frame_section_index"],
            "main_beam_section_index": env.current_params["main_beam_section_index"],
            "secondary_beam_section_index": env.current_params["secondary_beam_section_index"],
            "brace_section_index": env.current_params["brace_section_index"],
        }

        # 计算optimality gap
        optimal_volume = optimal_reports[pressure].optimal_volume
        gap = checker.calculate_optimality_gap(agent_solution, optimal_volume)
        gaps.append(gap)

        # 检查约束满足情况
        max_disp = info["max_displacement_mm"]
        max_stress = info["max_stress"]
        is_feasible = max_disp <= env.max_disp and max_stress <= env.max_stress

        logger.info(f"  智能体解: {agent_solution}")
        logger.info(f"  智能体体积: {info['model_volume_m3']:.4f} m³")
        logger.info(f"  最优体积: {optimal_volume:.4f} m³")
        logger.info(f"  Optimality Gap: {gap:.2f}%")
        logger.info(f"  约束满足: {'是' if is_feasible else '否'}")
        logger.info(f"  最大位移: {max_disp:.2f} mm (限制: {env.max_disp} mm)")
        logger.info(f"  最大应力: {max_stress:.2f} MPa (限制: {env.max_stress} MPa)")

    # 总结
    valid_gaps = [gap for gap in gaps if gap != float("inf")]
    if valid_gaps:
        avg_gap = np.mean(valid_gaps)
        max_gap = np.max(valid_gaps)
        min_gap = np.min(valid_gaps)

        logger.info(f"\n{'='*60}")
        logger.info(f"智能体性能总结:")
        logger.info(f"  平均Optimality Gap: {avg_gap:.2f}%")
        logger.info(f"  最大Gap: {max_gap:.2f}%")
        logger.info(f"  最小Gap: {min_gap:.2f}%")
        logger.info(f"  可行解比例: {len(valid_gaps)}/{len(gaps)} ({len(valid_gaps)/len(gaps)*100:.1f}%)")
        logger.info(f"={'='*60}")
    else:
        logger.info("\n警告: 智能体没有找到任何可行解!")

    agent.set_mode(eval=False)
    return gaps


def train(args, config):
    """训练强化学习智能体"""
    print(f"开始训练 DDPG 智能体，设备: {args.device}")
    device = torch.device(args.device)
    ts = datetime.now().strftime("%m%d_%H%M")
    logger = Logger(f"{args.model_save_path}/rewards_{ts}.log")

    rl_params = config["rl_params"]
    stress_limit = config["analysis_params"]["max_stress"]

    env = StructuralEnv(initial_params_config=args.config, logger=logger)
    epsilon_scheduler = EpsilonScheduler()

    # Determine HEBG in_channels from a sample observation
    sample_state, _ = env.reset()

    agent = get_agent(config, device, sample_state)
    is_ppo = isinstance(agent, VectorPPOAgent)
    if is_ppo:
        memory = VectorMemory()
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
        # 增加随机起点扰动以提升探索性
        reset_options = {
            "random_structure": {"random_keys": ["pressure"]},
            "random_start_perturbation": True,
            "perturbation_prob": episode / args.max_episodes,
        }
        state_pyg, info_reset = env.reset(options=reset_options)
        print("=" * 50)
        env.render()
        step = 0
        episode_reward = 0
        done = False

        state_pyg = state_pyg.to(device)

        while not done:
            step += 1
            global_step += 1
            if is_ppo:
                # 获取动作掩码(仅对size优化类型)
                action_mask = None
                if config["env_config"]["opt_type"] == "size":
                    action_mask = env.get_action_mask_for_size()
                elif config["env_config"]["opt_type"] == "spacing":
                    action_mask = env.get_action_mask_for_spacing()
                elif config["env_config"]["opt_type"] == "both":
                    action_mask = env.get_action_mask_for_both()
                action = agent.select_action(state_pyg, memory, action_mask)
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
            plot_progress(
                metrics_history, plot_save_dir, filename="training_progress.png", hline=stress_limit
            )

        print(
            f"Episode {episode+1}/{args.max_episodes}, Episode Return: {episode_reward:.2f}, Steps: {step}, Info: {info.get('error', 'OK')}"
        )
        env.render()
        print("=" * 50)
        print()

        # 评估当前模型在prototypes上的表现
        if (episode + 1) % rl_params["eval_interval"] == 0:
            eval_avg_reward, eval_results = evaluate_on_prototypes(agent, config, device, logger)
            print(f"[Prototype Eval] Episode {episode+1}: avg_reward={eval_avg_reward:.2f}")
            for res in eval_results:
                print(
                    f"  原型: {res['prototype']} | Vol: {res['final_volume']:.2f} | displacement: {res['final_displacement']:.2f} | stress: {res['final_stress']:.2f} | feasible: {'是' if res['is_feasible'] else '否'}"
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
        plot_progress(metrics_history, plot_save_dir, "training_progress.png", hline=stress_limit)
        print(f"Final training plot saved in {plot_save_dir}")


def test(args, config):
    """测试训练好的智能体"""
    print(f"开始测试智能体，设备: {args.device}")
    device = torch.device(args.device)
    ts = datetime.now().strftime("%m%d_%H%M")
    logger = Logger(f"test_{ts}.log")

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

    if args.prototype:
        # 基本原型评估
        print("\n" + "=" * 60)
        print("原型结构评估...")
        print("=" * 60)
        evaluate_on_prototypes(agent, config, device, logger)
    else:
        fname = os.path.basename(args.config).split(".")[0]
        print(f"在 {fname} 上评估智能体...")
        eval_on_config(agent, config, device, logger)

    # 最优性评估 (仅对size优化)
    if config["env_config"]["opt_type"] == "size":
        print("\n" + "=" * 60)
        print("最优性分析...")
        print("=" * 60)
        evaluate_model_optimality(agent, config, device, logger)


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
    elif args.mode == "convergence":
        print("开始穷举搜索以找到理论最优解...")
        from src.utils.convergence_checker import quick_convergence_test

        results = quick_convergence_test(args.config)
        print("\n穷举搜索完成！详细结果请查看日志文件。")
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
