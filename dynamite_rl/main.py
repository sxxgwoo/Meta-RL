import numpy as np
import torch
import torch.nn.functional as F
import os
from datetime import datetime

from config import Config
from envs import GridWorldAlternate
from models import DynaMITE_VAE, ActorCritic
from agent import DynaMITEAgent

from torch.utils.tensorboard import SummaryWriter
'''
nohup python -u main.py > train_$(date +"%Y%m%d_%H%M%S").log 2>&1 &
tensorboard --logdir logs --port 6006

'''

def make_env():
    """
    GridWorldAlternate 환경 생성.
    Config.max_episode_length, Config.bernoulli_p 사용.
    """
    return GridWorldAlternate(
        max_steps=Config.max_episode_length,
        switch_prob=Config.bernoulli_p
    )


def evaluate(env, agent, config, num_episodes=5):
    """
    평가 함수 (Gradient Update 없이 에이전트 성능 측정)
    - num_episodes개 에피소드의 평균/표준편차 리턴
    """
    agent.vae.eval()
    agent.actor_critic.eval()

    total_rewards = []

    for _ in range(num_episodes):
        obs, _ = env.reset()
        done = False

        # 초기 입력
        h = None
        prev_action = torch.zeros(env.action_space.n).to(config.device)
        prev_reward = torch.zeros(1).to(config.device)
        obs_tensor = torch.as_tensor(obs, dtype=torch.float32, device=config.device).unsqueeze(0)

        ep_reward = 0.0

        while not done:
            with torch.no_grad():
                # 1. Belief inference
                mu, logvar, _, h = agent.vae.encode(
                    obs_tensor.unsqueeze(1),                # (1, 1, obs_dim)
                    prev_action.view(1, 1, -1),            # (1, 1, A)
                    prev_reward.view(1, 1, -1),            # (1, 1, 1)
                    hidden=h
                )
                m = agent.vae.reparameterize(mu, logvar).squeeze(1)  # (1, latent_dim)

                # 2. Policy action
                action_idx, _, _, _ = agent.actor_critic.get_action_and_value(obs_tensor, m)
                action = action_idx.item()

            next_obs, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            # 다음 스텝 준비
            action_one_hot = torch.zeros(env.action_space.n, device=config.device)
            action_one_hot[action] = 1.0

            obs_tensor = torch.as_tensor(next_obs, dtype=torch.float32, device=config.device).unsqueeze(0)
            prev_action = action_one_hot
            prev_reward = torch.as_tensor([reward], dtype=torch.float32, device=config.device)

            ep_reward += reward

        total_rewards.append(ep_reward)

    # 다시 학습 모드로 전환
    agent.vae.train()
    agent.actor_critic.train()

    return float(np.mean(total_rewards)), float(np.std(total_rewards))


def collect_rollouts(env, agent, num_episodes, config):
    """
    데이터 수집 함수 (학습용)
    - 여러 에피소드를 rollout하고 pad_sequence로 (B, T_max, ·) 형태로 반환
    """
    rollout_buffer = {
        "states": [],
        "actions": [],
        "rewards": [],
        "prev_actions": [],
        "masks": [],
        "log_probs": [],
        "terminations": [],  # session_changed (d_t) 라벨
    }

    total_rewards = []

    for _ in range(num_episodes):
        obs, _ = env.reset()
        done = False

        ep_states = []
        ep_actions = []
        ep_rewards = []
        ep_prev_actions = []
        ep_masks = []
        ep_log_probs = []
        ep_terminations = []

        h = None
        prev_action = torch.zeros(env.action_space.n, device=config.device)
        prev_reward = torch.zeros(1, device=config.device)
        obs_tensor = torch.as_tensor(obs, dtype=torch.float32, device=config.device).unsqueeze(0)

        ep_reward = 0.0

        while not done:
            with torch.no_grad():
                mu, logvar, _, h = agent.vae.encode(
                    obs_tensor.unsqueeze(1),                # (1, 1, obs_dim)
                    prev_action.view(1, 1, -1),            # (1, 1, A)
                    prev_reward.view(1, 1, -1),            # (1, 1, 1)
                    hidden=h
                )
                m = agent.vae.reparameterize(mu, logvar).squeeze(1)  # (1, latent_dim)

                action_idx, log_prob, _, _ = agent.actor_critic.get_action_and_value(
                    obs_tensor, m
                )
                action = action_idx.item()

            next_obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            session_changed = info.get("session_changed", False)

            action_one_hot = torch.zeros(env.action_space.n, device=config.device)
            action_one_hot[action] = 1.0

            # 에피소드 시퀀스에 push
            ep_states.append(obs_tensor)                                   # (1, obs_dim)
            ep_actions.append(action_one_hot.unsqueeze(0))                 # (1, A)
            ep_rewards.append(torch.tensor([[reward]], device=config.device, dtype=torch.float32))  # (1,1)
            ep_prev_actions.append(prev_action.unsqueeze(0))               # (1, A)
            ep_masks.append(torch.tensor([[1.0 if not done else 0.0]],
                                         device=config.device, dtype=torch.float32))                # (1,1)
            ep_log_probs.append(log_prob)                                  # (1,)
            ep_terminations.append(
                torch.tensor([[1.0 if session_changed else 0.0]],
                             device=config.device, dtype=torch.float32)    # (1,1)
            )

            # 다음 step 준비
            obs_tensor = torch.as_tensor(next_obs, dtype=torch.float32, device=config.device).unsqueeze(0)
            prev_action = action_one_hot
            prev_reward = torch.as_tensor([reward], dtype=torch.float32, device=config.device)

            ep_reward += reward

        total_rewards.append(ep_reward)

        # (T_i, ·)로 concat해서 rollout_buffer에 저장
        rollout_buffer["states"].append(torch.cat(ep_states, dim=0))
        rollout_buffer["actions"].append(torch.cat(ep_actions, dim=0))
        rollout_buffer["rewards"].append(torch.cat(ep_rewards, dim=0))
        rollout_buffer["prev_actions"].append(torch.cat(ep_prev_actions, dim=0))
        rollout_buffer["masks"].append(torch.cat(ep_masks, dim=0))
        rollout_buffer["log_probs"].append(torch.cat(ep_log_probs, dim=0))
        rollout_buffer["terminations"].append(torch.cat(ep_terminations, dim=0))

    # (B, T_max, ·)로 pad
    from torch.nn.utils.rnn import pad_sequence
    batch_data = {}
    for k, v in rollout_buffer.items():
        batch_data[k] = pad_sequence(v, batch_first=True).to(config.device)

    return batch_data, float(np.mean(total_rewards))


def main():
    cfg = Config()

    # --- 체크포인트 / 로그 경로 세팅 ---
    run_id = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    save_dir = os.path.join("checkpoints", f"run_{run_id}")
    os.makedirs(save_dir, exist_ok=True)

    base_log_dir = os.path.join("logs", "dynamite_gridworld")
    log_dir_run = os.path.join(base_log_dir, f"run_{run_id}")
    os.makedirs(log_dir_run, exist_ok=True)
    writer = SummaryWriter(log_dir=log_dir_run)

    print(f"Initializing DynaMITE-RL for {cfg.env_name}...")
    print(f"Checkpoints will be saved to: {save_dir}")
    print(f"TensorBoard logs will be saved to: {log_dir_run}")

    # --- 환경 및 모델 초기화 ---
    env = make_env()

    obs_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n

    vae = DynaMITE_VAE(
        obs_dim=obs_dim,
        action_dim=action_dim,
        latent_dim=cfg.latent_dim,
        embed_dim=cfg.embedding_dim,
        hidden_size=cfg.vae_hidden_size,
    )

    actor_critic = ActorCritic(
        obs_dim=obs_dim,
        action_dim=action_dim,
        latent_dim=cfg.latent_dim,
        hidden_size=cfg.actor_critic_hidden,
    )

    agent = DynaMITEAgent(vae, actor_critic, cfg)

    best_eval_reward = -float("inf")

    print("Starting Training...")

    for i in range(1, cfg.num_iterations + 1):
        # 1. Rollout 수집
        batch_data, train_avg_reward = collect_rollouts(
            env, agent, num_episodes=cfg.num_processes, config=cfg
        )

        # 2. VAE + PPO 업데이트
        metrics = agent.update(batch_data)

        # --- TensorBoard: 학습 로그 ---
        writer.add_scalar("Train/avg_episode_reward", train_avg_reward, i)
        if "loss_vae" in metrics:
            writer.add_scalar("Loss/vae", metrics["loss_vae"], i)
        if "loss_rl" in metrics:
            writer.add_scalar("Loss/rl", metrics["loss_rl"], i)

        # 3. 주기적 평가 및 모델 저장
        if i % 10 == 0:
            eval_reward, eval_std = evaluate(env, agent, cfg, num_episodes=10)

            print(
                f"Iter {i:4d} | "
                f"Train R: {train_avg_reward:.2f} | "
                f"Eval R: {eval_reward:.2f} (+/- {eval_std:.2f}) | "
                f"VAE Loss: {metrics['loss_vae']:.4f} | "
                f"RL Loss: {metrics['loss_rl']:.4f}"
            )

            # --- TensorBoard: 평가 로그 ---
            writer.add_scalar("Eval/avg_episode_reward", eval_reward, i)
            writer.add_scalar("Eval/episode_reward_std", eval_std, i)
            writer.add_scalar(
                "Eval/best_avg_episode_reward",
                max(best_eval_reward, eval_reward),
                i,
            )

            # Best 모델 저장
            if eval_reward > best_eval_reward:
                best_eval_reward = eval_reward
                save_path = os.path.join(save_dir, "best_model.pt")
                torch.save(
                    {
                        "iter": i,
                        "vae_state_dict": agent.vae.state_dict(),
                        "actor_critic_state_dict": agent.actor_critic.state_dict(),
                        "best_reward": best_eval_reward,
                    },
                    save_path,
                )
                print(
                    f"    >>> New Best Model Saved to {save_dir}! "
                    f"Reward: {best_eval_reward:.2f}"
                )

    print(f"Training Complete. Best Evaluation Reward: {best_eval_reward:.2f}")

    writer.close()


if __name__ == "__main__":
    main()
