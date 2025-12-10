import torch
import os
import time
import numpy as np

from config import Config
from envs import GridWorldAlternate
from models import DynaMITE_VAE, ActorCritic
from agent import DynaMITEAgent

# ==========================================
# [설정] 불러올 체크포인트 경로
CHECKPOINT_PATH = "checkpoints/run_2025-12-11_00-28-16/best_model.pt"
NUM_EPISODES_TO_WATCH = 1
SLEEP_TIME = 0.3  # 각 스텝마다 대기할 시간 (초)
# ==========================================

def print_grid(env, info, reward=None):
    """
    터미널에 현재 Grid 상태를 그리는 함수
    A: Agent
    T: True Goal (보상 주는 곳)
    F: False Goal (보상 없는 곳)
    .: Empty
    """
    grid_size = env.grid_size
    # 빈 맵 생성
    visual_map = [["." for _ in range(grid_size)] for _ in range(grid_size)]
    
    # Goal 표시
    # env.goals[0], env.goals[1]이 있음.
    # active_goal_idx가 현재 진짜 목표
    true_goal_idx = env.active_goal_idx
    false_goal_idx = 1 - true_goal_idx
    
    g_true = env.goals[true_goal_idx]
    g_false = env.goals[false_goal_idx]
    
    # 먼저 False Goal, 그 다음 True Goal (Agent가 겹칠 경우 대비)
    visual_map[g_false[0]][g_false[1]] = "F"
    visual_map[g_true[0]][g_true[1]] = "T"
    
    # Agent 표시
    agent_pos = env.agent_pos
    
    # Agent가 Goal 위에 있으면 겹쳐서 표시
    current_char = visual_map[agent_pos[0]][agent_pos[1]]
    if current_char == "T":
        display_char = "★" # True Goal 도착
    elif current_char == "F":
        display_char = "X" # False Goal 도착
    else:
        display_char = "A" # 그냥 이동 중
        
    visual_map[agent_pos[0]][agent_pos[1]] = display_char

    # 출력
    print("\n" + "="*20)
    print(f"Step: {env.steps} | Active Goal: {true_goal_idx} (T)")
    if reward is not None:
        print(f"Last Reward: {reward}")
    
    for row in visual_map:
        print(" ".join(row))
    print("="*20)


def main():
    cfg = Config()
    
    # 1. 환경 및 모델 준비
    env = GridWorldAlternate(max_steps=cfg.max_episode_length, switch_prob=cfg.bernoulli_p)
    obs_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n

    vae = DynaMITE_VAE(obs_dim, action_dim, cfg.latent_dim, cfg.embedding_dim, cfg.vae_hidden_size)
    actor_critic = ActorCritic(obs_dim, action_dim, cfg.latent_dim, cfg.actor_critic_hidden)
    agent = DynaMITEAgent(vae, actor_critic, cfg)

    # 2. 체크포인트 로드
    if not os.path.exists(CHECKPOINT_PATH):
        print(f"Error: 파일을 찾을 수 없습니다 -> {CHECKPOINT_PATH}")
        return

    print(f"Loading checkpoint from {CHECKPOINT_PATH}...")
    # [수정 후] weights_only=False 옵션 추가
    checkpoint = torch.load(CHECKPOINT_PATH, map_location=cfg.device, weights_only=False)
    # checkpoint = torch.load(CHECKPOINT_PATH, map_location=cfg.device)
    agent.vae.load_state_dict(checkpoint["vae_state_dict"])
    agent.actor_critic.load_state_dict(checkpoint["actor_critic_state_dict"])
    
    # 평가 모드 전환
    agent.vae.eval()
    agent.actor_critic.eval()

    print("Visualizing Agent Behavior...")
    time.sleep(1)

    # 3. 시각화 루프
    for ep in range(NUM_EPISODES_TO_WATCH):
        print(f"\n>>> Episode {ep+1} Start <<<")
        obs, info = env.reset()
        done = False
        
        # Recurrent Hidden State 초기화
        h = None
        prev_action = torch.zeros(env.action_space.n).to(cfg.device)
        prev_reward = torch.zeros(1).to(cfg.device)
        obs_tensor = torch.as_tensor(obs, dtype=torch.float32, device=cfg.device).unsqueeze(0)
        
        print_grid(env, info) # 초기 상태 출력
        time.sleep(SLEEP_TIME)

        total_reward = 0
        
        while not done:
            with torch.no_grad():
                # VAE Inference
                mu, logvar, _, h = agent.vae.encode(
                    obs_tensor.unsqueeze(1),
                    prev_action.view(1, 1, -1),
                    prev_reward.view(1, 1, -1),
                    hidden=h
                )
                m = agent.vae.reparameterize(mu, logvar).squeeze(1)

                # Policy Action
                action_idx, _, _, _ = agent.actor_critic.get_action_and_value(obs_tensor, m)
                action = action_idx.item()

            # Env Step
            next_obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

            total_reward += reward

            # 화면 그리기
            # 터미널 화면을 지우고 싶으면 아래 주석 해제 (os.system('cls' if os.name == 'nt' else 'clear'))
            print_grid(env, info, reward)
            time.sleep(SLEEP_TIME)

            # 다음 스텝 준비
            action_one_hot = torch.zeros(env.action_space.n, device=cfg.device)
            action_one_hot[action] = 1.0
            
            obs_tensor = torch.as_tensor(next_obs, dtype=torch.float32, device=cfg.device).unsqueeze(0)
            prev_action = action_one_hot
            prev_reward = torch.as_tensor([reward], dtype=torch.float32, device=cfg.device)

        print(f">>> Episode {ep+1} End. Total Reward: {total_reward:.2f}")
        time.sleep(1)

if __name__ == "__main__":
    main()