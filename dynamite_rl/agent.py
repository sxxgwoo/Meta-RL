import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from torch.distributions import Normal, kl_divergence


class DynaMITEAgent:
    def __init__(self, vae, actor_critic, config):
        self.vae = vae.to(config.device)
        self.actor_critic = actor_critic.to(config.device)
        self.cfg = config

        self.opt_vae = optim.Adam(self.vae.parameters(), lr=config.lr_vae)
        self.opt_rl = optim.Adam(self.actor_critic.parameters(), lr=config.lr_policy)

    def update(self, rollouts):
        """
        rollouts: dict containing tensors of shape (B, T, ·)
        keys:
          states       : (B, T, obs_dim)
          actions      : (B, T, act_dim)  (one-hot)
          rewards      : (B, T, 1)
          prev_actions : (B, T, act_dim)
          masks        : (B, T, 1)   (1 for valid, 0 for padding/terminal 이후)
          log_probs    : (B, T)
          terminations : (B, T, 1)   (session_changed d_t)
        """
        states = rollouts["states"]         # (B, T, obs_dim)
        actions = rollouts["actions"]       # (B, T, act_dim)
        rewards = rollouts["rewards"]       # (B, T, 1)
        prev_actions = rollouts["prev_actions"]  # (B, T, act_dim)
        masks = rollouts["masks"]           # (B, T, 1)
        old_log_probs_all = rollouts["log_probs"]  # (B, T)
        terminations = rollouts["terminations"]    # (B, T, 1)

        B, T, _ = states.shape

        # term / session_id는 모든 VAE epoch에서 공통으로 쓰이므로 미리 계산
        term = terminations.squeeze(-1) * masks.squeeze(-1)  # (B, T)
        session_id = torch.cumsum(term, dim=1).long()        # (B, T)

        # ---------------------------------------------
        # 1. Train VAE (Inference Model) - Multi Epoch
        #    (같은 batch로 vae_epochs번 업데이트)
        # ---------------------------------------------
        last_vae_loss_value = None

        for _ in range(self.cfg.vae_epochs):  # ### [변경] 여러 epoch
            mu, logvar, term_logits, _ = self.vae.encode(states, prev_actions, rewards)
            z = self.vae.reparameterize(mu, logvar)

            # --------------------------------------
            # A. Session-based Reconstruction Mask
            # --------------------------------------
            same_session = (session_id.unsqueeze(2) == session_id.unsqueeze(1))  # (B, T, T)

            valid = masks.squeeze(-1) > 0.5                      # (B, T)
            pair_valid = (
                valid.unsqueeze(2) &  # latent index i
                valid.unsqueeze(1)    # target index j
            )  # (B, T, T)

            pair_mask = (same_session & pair_valid)              # (B, T, T), bool
            pair_mask_f = pair_mask.to(states.dtype)

            # z_i : (B, T, Dz) → (B, T, T, Dz)
            z_pair = z.unsqueeze(2).expand(B, T, T, z.size(-1))
            states_pair = states.unsqueeze(1).expand(B, T, T, states.size(-1))
            actions_pair = actions.unsqueeze(1).expand(B, T, T, actions.size(-1))
            rewards_pair = rewards.unsqueeze(1).expand(B, T, T, 1)

            BT2 = B * T * T
            states_flat = states_pair.reshape(BT2, -1)
            actions_flat = actions_pair.reshape(BT2, -1)
            z_flat = z_pair.reshape(BT2, -1)
            rewards_flat = rewards_pair.reshape(BT2, 1)

            pred_rewards_flat = self.vae.decode_reward(states_flat, actions_flat, z_flat)
            loss_recon_flat = F.mse_loss(
                pred_rewards_flat, rewards_flat, reduction="none"
            ).reshape(B, T, T, 1)  # (B, T, T, 1)

            loss_recon_pair = loss_recon_flat.squeeze(-1) * pair_mask_f  # (B, T, T)
            denom_recon = pair_mask_f.sum().clamp_min(1.0)
            loss_recon = loss_recon_pair.sum() / denom_recon

            # --------------------------------------
            # B. KL with latent belief conditioning
            #    - t = 0: prior = N(0, I)
            #    - t ≥ 1: prior = q_{t-1}
            # --------------------------------------
            mu_shift = torch.zeros_like(mu)
            logvar_shift = torch.zeros_like(logvar)
            mu_shift[:, 1:, :] = mu[:, :-1, :].detach()
            logvar_shift[:, 1:, :] = logvar[:, :-1, :].detach()
            # t=0 은 (0,0) → N(0,I) prior

            prior_mu = mu_shift
            prior_logvar = logvar_shift

            var_q = logvar.exp()
            var_p = prior_logvar.exp()  # t=0에서는 1, 그 외에는 이전 posterior 분산

            kl_elements = prior_logvar - logvar \
                          + (var_q + (mu - prior_mu) ** 2) / (var_p + 1e-8) \
                          - 1.0
            loss_kl_t = 0.5 * kl_elements.sum(-1)   # (B, T)
            loss_kl = (loss_kl_t * masks.squeeze(-1)).mean()

            # --------------------------------------
            # C. Consistency Loss (논문식)
            #    θ_t = KL(q_t || q_end(session))
            #    L_cons = max(θ_{t+1} - θ_t, 0)
            # --------------------------------------
            session_last_idx = torch.zeros_like(session_id)  # (B, T)
            session_last_idx[:, -1] = T - 1

            for t in range(T - 2, -1, -1):
                same_as_next = (session_id[:, t] == session_id[:, t + 1])
                session_last_idx[:, t] = torch.where(
                    same_as_next,
                    session_last_idx[:, t + 1],
                    torch.full_like(session_last_idx[:, t], t)
                )

            device = mu.device
            batch_indices = torch.arange(B, device=device).unsqueeze(1).expand(B, T)
            mu_end = mu[batch_indices, session_last_idx]          # (B, T, D)
            logvar_end = logvar[batch_indices, session_last_idx]  # (B, T, D)

            var_end = logvar_end.exp()

            theta_elements = logvar_end - logvar \
                             + (logvar.exp() + (mu - mu_end) ** 2) / (var_end + 1e-8) \
                             - 1.0
            theta = 0.5 * theta_elements.sum(-1)  # (B, T)

            theta_next = theta[:, 1:]      # (B, T-1)
            theta_curr = theta[:, :-1]     # (B, T-1)
            delta_theta = theta_next - theta_curr

            same_session_next = (session_id[:, 1:] == session_id[:, :-1])  # (B, T-1)
            valid_cons = same_session_next & (masks[:, 1:].squeeze(-1) > 0.5)
            valid_cons_f = valid_cons.to(theta.dtype)

            loss_consistency = (
                F.relu(delta_theta) * valid_cons_f
            ).sum() / valid_cons_f.sum().clamp_min(1.0)

            # --------------------------------------
            # D. Termination Loss (d_t)
            # --------------------------------------
            term_logits_flat = term_logits.squeeze(-1)      # (B, T)
            d_t = terminations.squeeze(-1)                 # (B, T)
            loss_term = F.binary_cross_entropy_with_logits(
                term_logits_flat, d_t, reduction="none"
            )
            loss_term = (loss_term * masks.squeeze(-1)).mean()

            # --------------------------------------
            # 최종 VAE Loss
            # --------------------------------------
            loss_vae = (
                loss_recon
                + self.cfg.beta_consistency * loss_consistency
                + 0.1 * loss_kl
                + self.cfg.lambda_term * loss_term
            )

            self.opt_vae.zero_grad()
            loss_vae.backward()
            nn.utils.clip_grad_norm_(self.vae.parameters(), self.cfg.max_grad_norm)
            self.opt_vae.step()

            last_vae_loss_value = loss_vae.item()  # 로그용 값 갱신

        # ---------------------------------------------------
        # 2. Train Policy (PPO)
        #    - 업데이트된 VAE로 다시 z를 뽑아서 사용
        # ---------------------------------------------------
        with torch.no_grad():  # ### [변경] 최신 VAE로 latent 다시 계산
            mu, logvar, _, _ = self.vae.encode(states, prev_actions, rewards)
            z = self.vae.reparameterize(mu, logvar)

        z_detached = z.detach()

        # --- GAE 계산 ---
        with torch.no_grad():
            flat_states_v = states.reshape(B * T, -1)
            flat_z_v = z_detached.reshape(B * T, -1)
            _, _, _, values_flat = self.actor_critic.get_action_and_value(
                flat_states_v, flat_z_v
            )
            values = values_flat.view(B, T, 1)  # (B, T, 1)

            advantages = torch.zeros_like(rewards)  # (B, T, 1)
            lastgaelam = torch.zeros_like(rewards[:, -1])  # (B, 1)

            for t in reversed(range(T)):
                if t == T - 1:
                    nextnonterminal = torch.zeros_like(rewards[:, t])  # (B,1)
                    nextvalues = torch.zeros_like(rewards[:, t])       # (B,1)
                else:
                    nextnonterminal = masks[:, t + 1]      # (B,1)
                    nextvalues = values[:, t + 1]          # (B,1)

                delta = (
                    rewards[:, t]
                    + self.cfg.gamma * nextvalues * nextnonterminal
                    - values[:, t]
                )
                lastgaelam = (
                    delta
                    + self.cfg.gamma
                    * self.cfg.gae_lambda
                    * nextnonterminal
                    * lastgaelam
                )
                advantages[:, t] = lastgaelam

            returns = advantages + values  # (B, T, 1)

            advantages_flat = advantages.view(-1)
            valid_mask_flat = masks.view(-1) > 0.5
            valid_adv = advantages_flat[valid_mask_flat]
            norm_adv = (valid_adv - valid_adv.mean()) / (valid_adv.std() + 1e-8)
            advantages_flat[valid_mask_flat] = norm_adv
            advantages = advantages_flat.view(B, T, 1)
            advantages = advantages.squeeze(-1)  # (B, T)

        # --- Flatten data for mini-batch PPO ---
        flat_states = states.reshape(B * T, -1)
        flat_z = z_detached.reshape(B * T, -1)
        flat_actions = actions.reshape(B * T, -1)
        flat_log_probs = old_log_probs_all.reshape(B * T)
        flat_advantages = advantages.reshape(B * T)
        flat_returns = returns.reshape(B * T)
        flat_masks = masks.reshape(B * T)

        valid = flat_masks > 0.5
        flat_states = flat_states[valid]
        flat_z = flat_z[valid]
        flat_actions = flat_actions[valid]
        flat_log_probs = flat_log_probs[valid]
        flat_advantages = flat_advantages[valid]
        flat_returns = flat_returns[valid]

        total_samples = flat_states.shape[0]
        idxs = np.arange(total_samples)

        # --- PPO Epochs Loop ---
        for _ in range(self.cfg.ppo_epochs):
            np.random.shuffle(idxs)

            for start in range(0, total_samples, self.cfg.mini_batch_size):
                end = start + self.cfg.mini_batch_size
                mb_idx = idxs[start:end]

                mb_states = flat_states[mb_idx]
                mb_z = flat_z[mb_idx]
                mb_actions = flat_actions[mb_idx]
                mb_old_log_probs = flat_log_probs[mb_idx]
                mb_advantages = flat_advantages[mb_idx]
                mb_returns = flat_returns[mb_idx]

                action_idx = mb_actions.argmax(-1)  # (mb,)
                _, new_log_probs, entropy, new_values = self.actor_critic.get_action_and_value(
                    mb_states, mb_z, action=action_idx
                )
                new_values = new_values.squeeze(-1)  # (mb,)

                ratio = torch.exp(new_log_probs - mb_old_log_probs)
                surr1 = ratio * mb_advantages
                surr2 = torch.clamp(
                    ratio,
                    1.0 - self.cfg.ppo_clip_eps,
                    1.0 + self.cfg.ppo_clip_eps,
                ) * mb_advantages

                loss_actor = -torch.min(surr1, surr2).mean()
                loss_critic = self.cfg.value_loss_coef * F.mse_loss(
                    new_values, mb_returns
                )
                loss_entropy = -self.cfg.entropy_coef * entropy.mean()

                loss_rl = loss_actor + loss_critic + loss_entropy

                self.opt_rl.zero_grad()
                loss_rl.backward()
                nn.utils.clip_grad_norm_(
                    self.actor_critic.parameters(), self.cfg.max_grad_norm
                )
                self.opt_rl.step()

        return {
            "loss_vae": last_vae_loss_value if last_vae_loss_value is not None else 0.0,
            "loss_rl": loss_rl.item(),
            "reward": rewards.sum().item(),
        }
