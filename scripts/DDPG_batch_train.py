import argparse
import csv
import json
import os
import sys
import time

import matplotlib.pyplot as plt
import numpy as np
import torch

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
SRC_DIR = os.path.join(REPO_ROOT, "src")
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

from DDPG_agent import DDPGAgent
from UAV_env_revised import UAVEnv


TRAIN_MODE_MAP = {
    "isac": 0,
    "vlm": 1,
    "noisac": 2,
    "our": 3,
}

PROFILE_LIBRARY = {
    "nominal": {"ground_length": 200.0, "ground_width": 200.0, "height": 200.0, "position_mode": "preset"},
    "large": {"ground_length": 1000.0, "ground_width": 1000.0, "height": 250.0, "position_mode": "random"},
}


def build_parser():
    parser = argparse.ArgumentParser(description="Batch-train revised DDPG UAV policies.")
    parser.add_argument("--profiles", type=str, default="nominal,large")
    parser.add_argument("--num-ground-stations-list", type=str, default="6,8,12")
    parser.add_argument("--scene-list", type=str, default="urban,dense,highrise")
    parser.add_argument("--train-modes", type=str, default="isac,vlm")
    parser.add_argument("--episodes", type=int, default=200)
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--seed", type=int, default=958030)
    parser.add_argument("--output-tag", type=str, default="batch_train")
    parser.add_argument("--checkpoint-interval", type=int, default=0)
    parser.add_argument("--warmup-steps", type=int, default=5000)
    parser.add_argument("--exploration-std-start", type=float, default=0.30)
    parser.add_argument("--exploration-std-end", type=float, default=0.05)
    parser.add_argument("--noise-clip", type=float, default=0.50)
    parser.add_argument("--init-actor-path", type=str, default="")
    parser.add_argument("--init-critic-path", type=str, default="")
    parser.add_argument("--init-gate-path", type=str, default="")
    parser.add_argument("--position-mode-override", type=str, default="")
    return parser


def parse_int_list(raw):
    return [int(item.strip()) for item in raw.split(",") if item.strip()]


def parse_str_list(raw):
    return [item.strip() for item in raw.split(",") if item.strip()]


def ensure_dir(path):
    os.makedirs(path, exist_ok=True)
    return path


def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_initial_weights(agent, actor_path="", critic_path=""):
    loaded = {"init_actor_path": "", "init_critic_path": ""}
    device = next(agent.actor.parameters()).device
    if actor_path:
        if not os.path.exists(actor_path):
            raise FileNotFoundError(f"Initial actor path does not exist: {actor_path}")
        actor_state = torch.load(actor_path, map_location=device)
        agent.actor.load_state_dict(actor_state)
        agent.actor_target.load_state_dict(actor_state)
        loaded["init_actor_path"] = actor_path
    if critic_path:
        if not os.path.exists(critic_path):
            raise FileNotFoundError(f"Initial critic path does not exist: {critic_path}")
        critic_state = torch.load(critic_path, map_location=device)
        agent.critic.load_state_dict(critic_state)
        agent.critic_target.load_state_dict(critic_state)
        loaded["init_critic_path"] = critic_path
    return loaded


def load_initial_gate(env, gate_path=""):
    if not gate_path:
        return ""
    if not os.path.exists(gate_path):
        raise FileNotFoundError(f"Initial gate path does not exist: {gate_path}")
    state = torch.load(gate_path, map_location=env.device)
    env.load_gate_state_dict(state)
    return gate_path


def smooth_series(values, size=20, stride=6):
    if not values:
        return np.array([])
    window = max(int(size), 1)
    kernel = np.ones(window, dtype=float) / window
    flattened = np.asarray(values, dtype=float).reshape(-1)
    smoothed = np.convolve(flattened, kernel, mode="same")
    return smoothed[:: max(int(stride), 1)]


def sample_training_action(agent, state, action_dim, total_steps, max_steps, args):
    if total_steps < args.warmup_steps:
        return np.random.uniform(-1.0, 1.0, size=action_dim).astype(np.float32)

    action = agent.get_action(state)
    progress = min(max(total_steps - args.warmup_steps, 0) / max(max_steps - args.warmup_steps, 1), 1.0)
    noise_std = args.exploration_std_start + progress * (args.exploration_std_end - args.exploration_std_start)
    noise = np.random.normal(0.0, noise_std, size=action_dim)
    noise = np.clip(noise, -args.noise_clip, args.noise_clip)
    return np.clip(action + noise, -1.0, 1.0).astype(np.float32)


def save_csv(path, fieldnames, rows):
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def scenario_tag(config):
    return (
        f"{config['profile']}_{config['scene']}_{config['train_mode']}_"
        f"gs{config['num_ground_stations']}_{int(config['ground_length'])}x{int(config['ground_width'])}_"
        f"h{int(config['height'])}"
    )


def build_configs(args):
    profiles = parse_str_list(args.profiles)
    gt_list = parse_int_list(args.num_ground_stations_list)
    scenes = parse_str_list(args.scene_list)
    train_modes = parse_str_list(args.train_modes)

    configs = []
    for profile in profiles:
        if profile not in PROFILE_LIBRARY:
            raise ValueError(f"Unknown profile '{profile}'. Supported: {sorted(PROFILE_LIBRARY)}")
        base = PROFILE_LIBRARY[profile]
        position_mode = args.position_mode_override.strip() or base["position_mode"]
        for scene in scenes:
            for num_ground_stations in gt_list:
                for train_mode in train_modes:
                    if train_mode not in TRAIN_MODE_MAP:
                        raise ValueError(f"Unknown train mode '{train_mode}'. Supported: {sorted(TRAIN_MODE_MAP)}")
                    config = {
                        "profile": profile,
                        "scene": scene,
                        "num_ground_stations": num_ground_stations,
                        "ground_length": base["ground_length"],
                        "ground_width": base["ground_width"],
                        "height": base["height"],
                        "position_mode": position_mode,
                        "episodes": args.episodes,
                        "steps": args.steps,
                        "train_mode": train_mode,
                        "mode_id": TRAIN_MODE_MAP[train_mode],
                    }
                    config["scenario_tag"] = scenario_tag(config)
                    configs.append(config)
    return configs


def plot_training_curves(config_dir, scenario_tag_value, reward_buffer, actor_loss_buffer, critic_loss_buffer, current_q_buffer, target_q_buffer):
    plt.figure(figsize=(10, 5))
    plt.plot(reward_buffer)
    plt.xlabel("Episode")
    plt.ylabel("Reward")
    plt.title(f"Reward - {scenario_tag_value}")
    plt.grid()
    plt.savefig(os.path.join(config_dir, f"{scenario_tag_value}_reward.png"), dpi=300, bbox_inches="tight")
    plt.close()

    if actor_loss_buffer and critic_loss_buffer:
        plt.figure(figsize=(10, 5))
        plt.plot(actor_loss_buffer, label="Actor Loss")
        plt.plot(critic_loss_buffer, label="Critic Loss")
        plt.title(f"Loss - {scenario_tag_value}")
        plt.xlabel("Train Step")
        plt.ylabel("Loss")
        plt.legend()
        plt.savefig(os.path.join(config_dir, f"{scenario_tag_value}_loss.png"), dpi=300, bbox_inches="tight")
        plt.close()

    if current_q_buffer and target_q_buffer:
        current_q = smooth_series(current_q_buffer, size=20, stride=6)
        target_q = smooth_series(target_q_buffer, size=20, stride=6)
        plt.figure(figsize=(10, 5))
        plt.plot(current_q, label="Current_Q")
        plt.plot(target_q, label="Target_Q")
        plt.xlabel("Train Step")
        plt.ylabel("Q Value")
        plt.legend()
        plt.savefig(os.path.join(config_dir, f"{scenario_tag_value}_q.png"), dpi=300, bbox_inches="tight")
        plt.close(fig)


def train_single_config(config, args, root_output_dir):
    print(f"\n=== Training {config['scenario_tag']} ===")
    set_seed(args.seed)
    env = UAVEnv(
        config["num_ground_stations"],
        config["ground_length"],
        config["ground_width"],
        config["height"],
        scene=config["scene"],
        position_mode=config["position_mode"],
        random_seed=args.seed,
    )
    state_dim = UAVEnv.state_dim_for(config["num_ground_stations"])
    action_dim = config["num_ground_stations"] + 3
    agent = DDPGAgent(state_dim, action_dim)
    init_info = load_initial_weights(agent, args.init_actor_path, args.init_critic_path)
    init_gate_path = load_initial_gate(env, args.init_gate_path)
    if init_info["init_actor_path"]:
        print("Initialized actor from:", init_info["init_actor_path"])
    if init_info["init_critic_path"]:
        print("Initialized critic from:", init_info["init_critic_path"])
    if init_gate_path:
        print("Initialized gate from:", init_gate_path)

    reward_buffer = np.empty(shape=config["episodes"])
    actor_loss_buffer = []
    critic_loss_buffer = []
    current_q_buffer = []
    target_q_buffer = []
    episode_rows = []

    model_dir = ensure_dir(os.path.join(REPO_ROOT, "models"))
    config_dir = ensure_dir(os.path.join(root_output_dir, "per_config", config["scenario_tag"]))
    timestamp = time.strftime("%Y%m%d%H%M%S")
    total_steps = 0
    max_steps = config["episodes"] * config["steps"]
    best_reward = -float("inf")
    best_episode = 0
    best_actor_path = os.path.join(model_dir, f"ddpg_actor_{config['scenario_tag']}_best_{timestamp}.pth")
    best_critic_path = os.path.join(model_dir, f"ddpg_critic_{config['scenario_tag']}_best_{timestamp}.pth")
    best_gate_path = os.path.join(model_dir, f"ddpg_gate_{config['scenario_tag']}_best_{timestamp}.pth")

    for episode_i in range(config["episodes"]):
        state = env.reset(mode=config["mode_id"])
        episode_reward = 0.0
        last_debug = {}

        for _ in range(config["steps"]):
            action = sample_training_action(agent, state, action_dim, total_steps, max_steps, args)
            next_state, action_step, reward, done, data, v_h, v_v, debug = env.step(action, mode=config["mode_id"])
            agent.replay_buffer.add_memo(state, action_step, reward, next_state, done)
            state = next_state
            episode_reward += reward
            last_debug = debug
            total_steps += 1

            update_result = agent.update()
            if update_result != 0:
                actor_loss, critic_loss, current_q, target_q = update_result
                actor_loss_buffer.append(float(np.asarray(actor_loss).reshape(-1)[0]))
                critic_loss_buffer.append(float(np.asarray(critic_loss).reshape(-1)[0]))
                current_q_buffer.append(float(np.asarray(current_q).reshape(-1)[0]))
                target_q_buffer.append(float(np.asarray(target_q).reshape(-1)[0]))

            if env.e_battery_uav <= 0 or done:
                break

        reward_buffer[episode_i] = episode_reward
        episode_rows.append(
            {
                "scenario_tag": config["scenario_tag"],
                "profile": config["profile"],
                "scene": config["scene"],
                "train_mode": config["train_mode"],
                "num_ground_stations": config["num_ground_stations"],
                "episode": episode_i + 1,
                "reward": float(episode_reward),
                "battery_remaining": float(env.e_battery_uav),
                "energy_total_last": float(last_debug.get("energy_total", 0.0)),
                "energy_fly_last": float(last_debug.get("energy_fly", 0.0)),
                "energy_comp_last": float(last_debug.get("energy_comp", 0.0)),
            }
        )
        print(f"Episode: {episode_i + 1}, Reward: {round(episode_reward, 4)}")

        if episode_reward > best_reward:
            best_reward = float(episode_reward)
            best_episode = episode_i + 1
            torch.save(agent.actor.state_dict(), best_actor_path)
            torch.save(agent.critic.state_dict(), best_critic_path)
            torch.save(env.get_gate_state_dict(), best_gate_path)

        if args.checkpoint_interval > 0 and (episode_i + 1) % args.checkpoint_interval == 0:
            ckpt_actor = os.path.join(model_dir, f"ddpg_actor_{config['scenario_tag']}_ep{episode_i + 1}_{timestamp}.pth")
            ckpt_critic = os.path.join(model_dir, f"ddpg_critic_{config['scenario_tag']}_ep{episode_i + 1}_{timestamp}.pth")
            ckpt_gate = os.path.join(model_dir, f"ddpg_gate_{config['scenario_tag']}_ep{episode_i + 1}_{timestamp}.pth")
            torch.save(agent.actor.state_dict(), ckpt_actor)
            torch.save(agent.critic.state_dict(), ckpt_critic)
            torch.save(env.get_gate_state_dict(), ckpt_gate)

    final_actor_path = os.path.join(model_dir, f"ddpg_actor_{config['scenario_tag']}_{timestamp}.pth")
    final_critic_path = os.path.join(model_dir, f"ddpg_critic_{config['scenario_tag']}_{timestamp}.pth")
    final_gate_path = os.path.join(model_dir, f"ddpg_gate_{config['scenario_tag']}_{timestamp}.pth")
    torch.save(agent.actor.state_dict(), final_actor_path)
    torch.save(agent.critic.state_dict(), final_critic_path)
    torch.save(env.get_gate_state_dict(), final_gate_path)

    plot_training_curves(
        config_dir,
        config["scenario_tag"],
        reward_buffer,
        actor_loss_buffer,
        critic_loss_buffer,
        current_q_buffer,
        target_q_buffer,
    )

    summary_row = {
        "scenario_tag": config["scenario_tag"],
        "profile": config["profile"],
        "scene": config["scene"],
        "train_mode": config["train_mode"],
        "mode_id": config["mode_id"],
        "num_ground_stations": config["num_ground_stations"],
        "ground_length": config["ground_length"],
        "ground_width": config["ground_width"],
        "height": config["height"],
        "episodes": config["episodes"],
        "steps": config["steps"],
        "state_dim": state_dim,
        "action_dim": action_dim,
        "reward_mean": float(np.mean(reward_buffer)),
        "reward_std": float(np.std(reward_buffer)),
        "reward_final": float(reward_buffer[-1]),
        "reward_best": best_reward,
        "best_episode": best_episode,
        "actor_loss_final": float(actor_loss_buffer[-1]) if actor_loss_buffer else None,
        "critic_loss_final": float(critic_loss_buffer[-1]) if critic_loss_buffer else None,
        "current_q_final": float(current_q_buffer[-1]) if current_q_buffer else None,
        "target_q_final": float(target_q_buffer[-1]) if target_q_buffer else None,
        "best_actor_path": best_actor_path,
        "best_critic_path": best_critic_path,
        "best_gate_path": best_gate_path,
        "final_actor_path": final_actor_path,
        "final_critic_path": final_critic_path,
        "final_gate_path": final_gate_path,
        "init_actor_path": init_info["init_actor_path"],
        "init_critic_path": init_info["init_critic_path"],
        "init_gate_path": init_gate_path,
    }

    save_csv(os.path.join(config_dir, "episode_rewards.csv"), list(episode_rows[0].keys()), episode_rows)
    save_csv(os.path.join(config_dir, "training_summary.csv"), list(summary_row.keys()), [summary_row])
    return episode_rows, summary_row


def plot_summary(root_output_dir, summary_rows):
    metrics = ["reward_mean", "reward_final"]
    profiles = sorted({row["profile"] for row in summary_rows})
    scenes = sorted({row["scene"] for row in summary_rows})
    gt_values = sorted({row["num_ground_stations"] for row in summary_rows})
    train_modes = sorted({row["train_mode"] for row in summary_rows})

    summary_plot_dir = ensure_dir(os.path.join(root_output_dir, "summary_plots"))
    for profile in profiles:
        for metric in metrics:
            fig, axes = plt.subplots(1, len(scenes), figsize=(5 * len(scenes), 4), squeeze=False)
            for ax, scene in zip(axes[0], scenes):
                subset = [row for row in summary_rows if row["profile"] == profile and row["scene"] == scene]
                if not subset:
                    ax.set_visible(False)
                    continue
                for train_mode in train_modes:
                    y = []
                    for gt in gt_values:
                        match = next((row for row in subset if row["train_mode"] == train_mode and row["num_ground_stations"] == gt), None)
                        y.append(match[metric] if match else np.nan)
                    ax.plot(gt_values, y, marker="o", label=train_mode)
                ax.set_title(f"{profile} | {scene}")
                ax.set_xlabel("Number of GTs")
                ax.set_ylabel(metric)
                ax.grid(alpha=0.3)
                ax.set_xticks(gt_values)
            handles, labels = axes[0][0].get_legend_handles_labels()
            if handles:
                fig.legend(handles, labels, loc="upper center", ncol=max(1, len(train_modes)))
            fig.tight_layout(rect=(0, 0, 1, 0.92))
            plt.savefig(os.path.join(summary_plot_dir, f"{profile}_{metric}.png"), dpi=300, bbox_inches="tight")
            plt.close(fig)


def main():
    args = build_parser().parse_args()
    configs = build_configs(args)
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    root_output_dir = ensure_dir(os.path.join(REPO_ROOT, "batch_train_results", f"{args.output_tag}_{timestamp}"))
    ensure_dir(os.path.join(root_output_dir, "per_config"))

    with open(os.path.join(root_output_dir, "config_manifest.json"), "w", encoding="utf-8") as f:
        json.dump({"configs": configs, "device": str(torch.device('cuda' if torch.cuda.is_available() else 'cpu'))}, f, indent=2)

    all_episode_rows = []
    all_summary_rows = []
    model_registry = {}
    for config in configs:
        episode_rows, summary_row = train_single_config(config, args, root_output_dir)
        all_episode_rows.extend(episode_rows)
        all_summary_rows.append(summary_row)
        registry_key = "|".join(
            [
                summary_row["profile"],
                summary_row["scene"],
                str(summary_row["num_ground_stations"]),
                str(int(summary_row["ground_length"])),
                str(int(summary_row["ground_width"])),
                str(int(summary_row["height"])),
                summary_row["train_mode"],
            ]
        )
        model_registry[registry_key] = {
            "scenario_tag": summary_row["scenario_tag"],
            "profile": summary_row["profile"],
            "scene": summary_row["scene"],
            "num_ground_stations": summary_row["num_ground_stations"],
            "ground_length": summary_row["ground_length"],
            "ground_width": summary_row["ground_width"],
            "height": summary_row["height"],
            "train_mode": summary_row["train_mode"],
            "final_actor_path": summary_row["final_actor_path"],
            "final_critic_path": summary_row["final_critic_path"],
            "final_gate_path": summary_row["final_gate_path"],
            "best_actor_path": summary_row["best_actor_path"],
            "best_critic_path": summary_row["best_critic_path"],
            "best_gate_path": summary_row["best_gate_path"],
            "reward_best": summary_row["reward_best"],
            "best_episode": summary_row["best_episode"],
            "reward_mean": summary_row["reward_mean"],
            "reward_final": summary_row["reward_final"],
            "init_actor_path": summary_row["init_actor_path"],
            "init_critic_path": summary_row["init_critic_path"],
            "init_gate_path": summary_row["init_gate_path"],
        }

    if all_episode_rows:
        save_csv(os.path.join(root_output_dir, "all_episode_rewards.csv"), list(all_episode_rows[0].keys()), all_episode_rows)
    if all_summary_rows:
        save_csv(os.path.join(root_output_dir, "all_training_summary.csv"), list(all_summary_rows[0].keys()), all_summary_rows)
        plot_summary(root_output_dir, all_summary_rows)
    with open(os.path.join(root_output_dir, "model_registry.json"), "w", encoding="utf-8") as f:
        json.dump(model_registry, f, indent=2)

    print("\nBatch training completed.")
    print("Output directory:", root_output_dir)


if __name__ == "__main__":
    main()
