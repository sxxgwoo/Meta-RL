import torch

class Config:
    # --- General ---
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    seed = 42
    
    # --- Environment (GridWorld) ---
    env_name = "GridWorldAlternate"
    max_episode_length = 60
    
    # --- DynaMITE-RL Specific ---
    bernoulli_p = 0.07 
    
    latent_dim = 5
    embedding_dim = 8
    
    # --- PPO & Optimization ---
    num_iterations = 1000   
    num_processes = 16
    
    ppo_epochs = 10        # 데이터 재사용 횟수
    mini_batch_size = 256  # 미니배치 크기
    
    lr_policy = 3e-4
    lr_vae = 3e-4
    gamma = 0.99
    gae_lambda = 0.95
    max_grad_norm = 0.5
    ppo_clip_eps = 0.2
    entropy_coef = 0.01    # (튜닝 전: 0.01)
    value_loss_coef = 0.5
    
    # --- Loss Weights ---
    beta_consistency = 0.5 # (튜닝 전: 0.5)
    lambda_vae = 0.01
    lambda_term = 1.0           # NEW: termination BCE loss weight
    
    # Hidden sizes
    vae_hidden_size = 64   # (튜닝 전: 64)
    actor_critic_hidden = 128
    
    vae_epochs = 5