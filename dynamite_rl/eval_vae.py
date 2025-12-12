import argparse
import numpy as np
import torch
import torch.nn.functional as F

from config import Config
from envs import GridWorldAlternate   # 너 프로젝트 구조에 맞게 수정
from models import DynaMITE_VAE
'''
python eval_vae.py \
    --vae-checkpoint checkpoints/dynamite_gridworld_vae_reward_only.pth \
    --num-episodes 128 \
    --threshold 0.5
'''

# ---------------------------------------------------
# 0. 환경 생성 함수 (논문 Gridworld 설정과 동일하게)
# ---------------------------------------------------
def make_env():
    return GridWorldAlternate(
        max_steps=Config.max_episode_length,
        switch_prob=Config.bernoulli_p,
    )


def one_hot_action(action_idx, action_dim):
    """Discrete action -> one-hot 벡터"""
    vec = np.zeros(action_dim, dtype=np.float32)
    vec[action_idx] = 1.0
    return vec


# ---------------------------------------------------
# 1. VAE 평가용 롤아웃 수집 (랜덤 정책)
# ---------------------------------------------------
def collect_rollouts_for_vae(env, num_episodes, device):
    """
    VAE 인퍼런스 입력만 수집하는 간단한 rollouts:
      - state_t
      - prev_action_{t-1} (one-hot)
      - reward_t
      - termination_t (session_changed: 0/1)
      - mask_t (유효 스텝 = 1)
    """
    obs_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n
    T = Config.max_episode_length

    all_states = []
    all_prev_actions = []
    all_rewards = []
    all_terms = []
    all_masks = []

    for ep in range(num_episodes):
        obs, info = env.reset()
        # (T, ·) 버퍼
        states = np.zeros((T, obs_dim), dtype=np.float32)
        prev_actions = np.zeros((T, action_dim), dtype=np.float32)
        rewards = np.zeros((T, 1), dtype=np.float32)
        terms = np.zeros((T, 1), dtype=np.float32)
        masks = np.zeros((T, 1), dtype=np.float32)

        prev_a_oh = np.zeros(action_dim, dtype=np.float32)  # t=0 이전 action은 0 벡터

        for t in range(T):
            # 기록: 현재 step의 입력은 (s_t, a_{t-1}, r_t)
            states[t] = obs
            prev_actions[t] = prev_a_oh

            # 랜덤 액션 (VAE만 볼 거라 정책은 아무거나 괜찮음)
            a = env.action_space.sample()
            a_oh = one_hot_action(a, action_dim)

            next_obs, r, terminated, truncated, info = env.step(a)

            rewards[t, 0] = r
            masks[t, 0] = 1.0

            # 환경이 알려주는 "세션 변경" 플래그 (이름은 프로젝트에 맞춰 수정)
            # 예: info["session_changed"] 또는 info["context_switched"] 등
            session_changed = info.get("session_changed", False)
            terms[t, 0] = float(session_changed)

            prev_a_oh = a_oh
            obs = next_obs

            if terminated or truncated:
                # 나머지 스텝은 mask=0 상태 그대로 두고 break
                break

        all_states.append(states)
        all_prev_actions.append(prev_actions)
        all_rewards.append(rewards)
        all_terms.append(terms)
        all_masks.append(masks)

    # (B, T, ·) 로 묶어서 텐서로 변환
    states_t = torch.from_numpy(np.stack(all_states, axis=0)).to(device)
    prev_actions_t = torch.from_numpy(np.stack(all_prev_actions, axis=0)).to(device)
    rewards_t = torch.from_numpy(np.stack(all_rewards, axis=0)).to(device)
    terms_t = torch.from_numpy(np.stack(all_terms, axis=0)).to(device)
    masks_t = torch.from_numpy(np.stack(all_masks, axis=0)).to(device)

    rollouts = {
        "states": states_t,          # (B, T, obs_dim)
        "prev_actions": prev_actions_t,  # (B, T, action_dim)
        "rewards": rewards_t,        # (B, T, 1)
        "terminations": terms_t,     # (B, T, 1)
        "masks": masks_t,            # (B, T, 1)
    }
    return rollouts


# ---------------------------------------------------
# 2. VAE의 termination 예측 성능 평가
# ---------------------------------------------------
def evaluate_termination_prediction(vae, rollouts, device, threshold=0.5):
    vae.eval()

    states = rollouts["states"]
    prev_actions = rollouts["prev_actions"]
    rewards = rollouts["rewards"]
    term_gt = rollouts["terminations"]      # (B, T, 1)
    masks = rollouts["masks"]              # (B, T, 1)

    with torch.no_grad():
        mu, logvar, term_logits, _ = vae.encode(states, prev_actions, rewards)
        term_logit_flat = term_logits.squeeze(-1)  # (B, T)
        term_prob = torch.sigmoid(term_logit_flat)  # (B, T)

        y_true = term_gt.squeeze(-1)       # (B, T)
        y_prob = term_prob                # (B, T)
        y_pred = (y_prob >= threshold).float()

        valid = masks.squeeze(-1) > 0.5   # (B, T) boolean

        y_true_v = y_true[valid]
        y_prob_v = y_prob[valid]
        y_pred_v = y_pred[valid]

        # BCE loss (termination loss와 같은 것)
        bce = F.binary_cross_entropy(y_prob_v, y_true_v)

        # 메트릭 계산
        acc = (y_pred_v == y_true_v).float().mean().item()

        pos = (y_true_v == 1.0)
        neg = (y_true_v == 0.0)

        tp = ((y_pred_v == 1.0) & pos).sum().item()
        fp = ((y_pred_v == 1.0) & neg).sum().item()
        fn = ((y_pred_v == 0.0) & pos).sum().item()
        tn = ((y_pred_v == 0.0) & neg).sum().item()

        precision = tp / (tp + fp + 1e-8)
        recall = tp / (tp + fn + 1e-8)
        f1 = 2 * precision * recall / (precision + recall + 1e-8)

    print("====== VAE Termination Detection Evaluation ======")
    print(f"Valid steps          : {y_true_v.numel()}")
    print(f"Positive(=switch)    : {int(pos.sum().item())}")
    print(f"Negative(=no switch) : {int(neg.sum().item())}\n")

    print(f"BCE loss (term)      : {bce.item():.6f}")
    print(f"Accuracy             : {acc:.4f}")
    print(f"Precision            : {precision:.4f}")
    print(f"Recall               : {recall:.4f}")
    print(f"F1-score             : {f1:.4f}\n")

    print("Confusion matrix (on valid steps):")
    print(f"  TP (true switch & predicted switch)   : {tp}")
    print(f"  FP (no switch but predicted switch)   : {fp}")
    print(f"  FN (switch but predicted no switch)   : {fn}")
    print(f"  TN (no switch & predicted no switch)  : {tn}\n")

    return {
        "bce": bce.item(),
        "acc": acc,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
    }


# ---------------------------------------------------
# 3. 예시 에피소드 몇 개에서 스위치 타이밍 시각적으로 확인
# ---------------------------------------------------
def show_example_episodes(vae, rollouts, device, num_examples=3):
    vae.eval()

    states = rollouts["states"]
    prev_actions = rollouts["prev_actions"]
    rewards = rollouts["rewards"]
    term_gt = rollouts["terminations"]      # (B, T, 1)
    masks = rollouts["masks"]              # (B, T, 1)

    B, T, _ = states.shape

    with torch.no_grad():
        mu, logvar, term_logits, _ = vae.encode(states, prev_actions, rewards)
        term_prob = torch.sigmoid(term_logits.squeeze(-1))   # (B, T)

    print("====== Example Episodes (true switch vs predicted prob) ======")
    for b in range(min(num_examples, B)):
        valid_len = int(masks[b].sum().item())
        gt_indices = (term_gt[b, :valid_len, 0] > 0.5).nonzero(as_tuple=False).flatten().cpu().numpy().tolist()

        probs = term_prob[b, :valid_len].cpu().numpy()

        # 상위 K 타임스텝 (예: 상위 5개)
        top_k = min(5, valid_len)
        top_indices = np.argsort(probs)[::-1][:top_k]

        print(f"\n--- Episode {b} ---")
        print(f"Valid length T      : {valid_len}")
        print(f"True switch indices : {gt_indices}")
        print("Top-{} predicted switch indices (by prob):".format(top_k))
        for idx in top_indices:
            print(f"  t = {idx:2d}, p(dt=1|·) = {probs[idx]:.4f}")


# ---------------------------------------------------
# 4. 메인 스크립트
# ---------------------------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--vae-checkpoint", type=str, required=True,
                        help="torch.save로 저장한 VAE 체크포인트 경로")
    parser.add_argument("--num-episodes", type=int, default=64,
                        help="평가에 사용할 에피소드 수")
    parser.add_argument("--threshold", type=float, default=0.5,
                        help="dt=1 판정에 사용할 시그모이드 임계값")
    args = parser.parse_args()

    device = Config.device

    # --- 환경/모델 준비 ---
    env = make_env()

    obs_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n
    latent_dim = Config.latent_dim      # Config에 있다고 가정
    embed_dim = Config.embedding_dim        # Config에 있다고 가정
    hidden_size = Config.vae_hidden_size    # Config에 있다고 가정

    vae = DynaMITE_VAE(
        obs_dim=obs_dim,
        action_dim=action_dim,
        latent_dim=latent_dim,
        embed_dim=embed_dim,
        hidden_size=hidden_size,
    ).to(device)

    # --- 체크포인트 로드 ---
    ckpt = torch.load(args.vae_checkpoint, map_location=device)
    # 저장 방식에 따라 분기 (그냥 state_dict 저장 vs dict["vae"] 등)
    if isinstance(ckpt, dict) and "vae" in ckpt:
        vae.load_state_dict(ckpt["vae"])
    else:
        vae.load_state_dict(ckpt)

    print(f"[INFO] Loaded VAE checkpoint from {args.vae_checkpoint}")

    # --- 롤아웃 수집 ---
    rollouts = collect_rollouts_for_vae(env, args.num_episodes, device)

    # --- termination detection 성능 평가 ---
    metrics = evaluate_termination_prediction(vae, rollouts, device, threshold=args.threshold)

    # --- 예시 에피소드 몇 개 출력 ---
    show_example_episodes(vae, rollouts, device, num_examples=5)


if __name__ == "__main__":
    main()
