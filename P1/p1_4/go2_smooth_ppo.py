"""
go2_smooth_ppo.py - [P1.4] PPO with optional action-smoothness regularizers, for rsl-rl-lib 5.5.1.

go2_train.py only uses this class when it is given --lcp, --caps-t or --caps-s; otherwise training uses rsl_rl's
PPO unchanged. All three terms act on the policy network, not on the reward:

- LCP, Lipschitz-constrained policy (Chen et al. 2024): gradient penalty
      L_gp = E[ ||d log pi(a|s) / ds||^2 ]   over the PPO mini-batch,
  as in their code (ppo_rma.py, _calc_grad_penalty), where every robot uses a coefficient of 0.002.
- CAPS (Mysore et al. 2021), with mu(s) the mean action:
      L_T = E[ ||mu(s_t) - mu(s_t+1)|| ]       temporal: consecutive steps of the same episode
      L_S = E[ ||mu(s) - mu(s + eps)|| ]       spatial: eps ~ N(0, sigma)
  Their PPO settings for Ant: lambda_T = 0.1, lambda_S = 0.5. They choose sigma "based on expected measurement
  noise": go2_train.py passes the observation-noise scales of P1.2.

update() is a copy of rsl_rl 5.5.1 PPO.update(); the additions are marked [P1.4]. If you upgrade rsl_rl, re-copy
update() from the new version and re-apply the marked blocks.
"""

from __future__ import annotations

import warnings
from importlib import metadata

import torch
import torch.nn as nn
from tensordict import TensorDict

from rsl_rl.algorithms import PPO

RSL_RL_VERSION = "5.5.1"


class SmoothPPO(PPO):
    def __init__(
        self,
        *args,
        grad_penalty_coef: float = 0.0,
        caps_temporal_coef: float = 0.0,
        caps_spatial_coef: float = 0.0,
        caps_sigma: float | list[float] = 0.0,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        if metadata.version("rsl-rl-lib") != RSL_RL_VERSION:
            warnings.warn(
                f"go2_smooth_ppo.SmoothPPO.update() is a copy of rsl-rl-lib {RSL_RL_VERSION}; you have "
                f"{metadata.version('rsl-rl-lib')}. Re-copy PPO.update() and re-apply the [P1.4] blocks."
            )
        if self.actor.is_recurrent or self.critic.is_recurrent:
            raise NotImplementedError("SmoothPPO is written for MLP actors and critics.")
        self.grad_penalty_coef = grad_penalty_coef
        self.caps_temporal_coef = caps_temporal_coef
        self.caps_spatial_coef = caps_spatial_coef
        # per actor-input dimension (list) or one value for all
        self.caps_sigma = torch.as_tensor(caps_sigma, dtype=torch.float, device=self.device)

    # [P1.4] ---------------------------------------------------------------------------------------------------
    def _grad_penalty(self, actions_log_prob: torch.Tensor, actor_inputs: list[torch.Tensor]) -> torch.Tensor:
        """LCP: mean over the batch of the squared norm of d log pi(a|s) / d(actor input)."""
        grads = torch.autograd.grad(actions_log_prob.sum(), actor_inputs, create_graph=True)
        return sum(g.square().sum(dim=-1) for g in grads).mean()

    def _caps_temporal(self, n_samples: int) -> torch.Tensor:
        """CAPS L_T on random (step, env) pairs of the rollout whose next observation is in the same episode."""
        st = self.storage
        t = torch.randint(0, st.num_transitions_per_env - 1, (n_samples,), device=self.device)
        e = torch.randint(0, st.num_envs, (n_samples,), device=self.device)
        same_episode = 1.0 - st.dones[t, e].float().view(-1)  # done at t -> observation t+1 is a new episode
        mu_t = self.actor(st.observations[t, e])
        mu_next = self.actor(st.observations[t + 1, e])
        dist = torch.sqrt((mu_t - mu_next).square().sum(dim=-1) + 1e-8)
        return (dist * same_episode).sum() / same_episode.sum().clamp(min=1.0)

    def _caps_spatial(self, obs: TensorDict) -> torch.Tensor:
        """CAPS L_S: distance between the mean actions at s and at s + N(0, sigma)."""
        noisy = obs.clone()
        start = 0
        for group in self.actor.obs_groups:
            dim = obs[group].shape[-1]
            sigma = self.caps_sigma if self.caps_sigma.ndim == 0 else self.caps_sigma[start : start + dim]
            noisy[group] = obs[group] + sigma * torch.randn_like(obs[group])
            start += dim
        dist = torch.sqrt((self.actor(obs) - self.actor(noisy)).square().sum(dim=-1) + 1e-8)
        return dist.mean()

    # ------------------------------------------------------------------------------------------------------------
    def update(self) -> dict[str, float]:
        """Run optimization epochs over stored batches and return mean losses."""
        mean_value_loss = 0
        mean_surrogate_loss = 0
        mean_entropy = 0
        # RND loss
        mean_rnd_loss = 0 if self.rnd else None
        # Symmetry loss
        mean_symmetry_loss = 0 if self.symmetry else None
        # [P1.4] smoothness losses
        mean_smooth = {"grad_penalty": 0.0, "caps_temporal": 0.0, "caps_spatial": 0.0}

        # Get mini-batch generator
        if self.actor.is_recurrent or self.critic.is_recurrent:
            generator = self.storage.recurrent_mini_batch_generator(self.num_mini_batches, self.num_learning_epochs)
        else:
            generator = self.storage.mini_batch_generator(self.num_mini_batches, self.num_learning_epochs)

        # Iterate over mini-batches
        for batch in generator:
            original_batch_size = batch.observations.batch_size[0]

            # Check if we should normalize advantages per mini-batch
            if self.normalize_advantage_per_mini_batch:
                with torch.no_grad():
                    batch.advantages = (batch.advantages - batch.advantages.mean()) / (batch.advantages.std() + 1e-8)  # type: ignore

            # Perform symmetric augmentation if enabled
            if self.symmetry:
                self.symmetry.augment_batch(batch, original_batch_size)

            # [P1.4] the actor gets its own copy of the observations, which requires grad for the gradient penalty
            actor_obs = batch.observations
            actor_inputs = []
            if self.grad_penalty_coef > 0.0:
                actor_obs = batch.observations.clone()
                for group in self.actor.obs_groups:
                    actor_obs[group] = actor_obs[group].detach().requires_grad_(True)
                    actor_inputs.append(actor_obs[group])

            # Optionally use mixed precision for the forward pass and loss computation
            with torch.amp.autocast(  # type: ignore
                device_type=torch.device(self.device).type, enabled=self.use_mixed_precision, dtype=torch.bfloat16
            ):
                # Recompute actions log prob and entropy for current batch of transitions
                # Note: We need to do this because we updated the policy with new parameters
                self.actor(
                    actor_obs,  # [P1.4] was batch.observations
                    masks=batch.masks,
                    hidden_state=batch.hidden_states[0],
                    stochastic_output=True,
                )
                actions_log_prob = self.actor.get_output_log_prob(batch.actions)  # type: ignore
                values = self.critic(batch.observations, masks=batch.masks, hidden_state=batch.hidden_states[1])
                # Note: We only keep the following tensors for the original samples in case of symmetry augmentation
                distribution_params = tuple(p[:original_batch_size] for p in self.actor.output_distribution_params)
                entropy = self.actor.output_entropy[:original_batch_size]

                # Compute KL divergence and adapt the learning rate
                if self.desired_kl is not None and self.schedule == "adaptive":
                    with torch.inference_mode():
                        kl = self.actor.get_kl_divergence(batch.old_distribution_params, distribution_params)  # type: ignore
                        kl_mean = torch.mean(kl)

                        # Reduce the KL divergence across all GPUs
                        if self.is_multi_gpu:
                            torch.distributed.all_reduce(kl_mean, op=torch.distributed.ReduceOp.SUM)
                            kl_mean /= self.gpu_world_size

                        # Update the learning rate only on the main process
                        if self.gpu_global_rank == 0:
                            if kl_mean > self.desired_kl * 2.0:
                                self.learning_rate = max(1e-5, self.learning_rate / 1.5)
                            elif kl_mean < self.desired_kl / 2.0 and kl_mean > 0.0:
                                self.learning_rate = min(1e-2, self.learning_rate * 1.5)

                        # Update the learning rate for all GPUs
                        if self.is_multi_gpu:
                            lr_tensor = torch.tensor(self.learning_rate, device=self.device)
                            torch.distributed.broadcast(lr_tensor, src=0)
                            self.learning_rate = lr_tensor.item()

                        # Update the learning rate for all parameter groups
                        for param_group in self.optimizer.param_groups:
                            param_group["lr"] = self.learning_rate

                # Surrogate loss
                ratio = torch.exp(actions_log_prob - torch.squeeze(batch.old_actions_log_prob))  # type: ignore
                surrogate = -torch.squeeze(batch.advantages) * ratio  # type: ignore
                surrogate_clipped = -torch.squeeze(batch.advantages) * torch.clamp(  # type: ignore
                    ratio, 1.0 - self.clip_param, 1.0 + self.clip_param
                )
                surrogate_loss = torch.max(surrogate, surrogate_clipped).mean()

                # Value function loss
                if self.use_clipped_value_loss:
                    value_clipped = batch.values + (values - batch.values).clamp(-self.clip_param, self.clip_param)
                    value_losses = (values - batch.returns).pow(2)
                    value_losses_clipped = (value_clipped - batch.returns).pow(2)
                    value_loss = torch.max(value_losses, value_losses_clipped).mean()
                else:
                    value_loss = (batch.returns - values).pow(2).mean()

                loss = surrogate_loss + self.value_loss_coef * value_loss - self.entropy_coef * entropy.mean()

                # [P1.4] smoothness regularizers (after the PPO terms: the extra actor passes below are
                # deterministic and do not touch the distribution used above)
                if self.grad_penalty_coef > 0.0:
                    grad_penalty = self._grad_penalty(actions_log_prob, actor_inputs)
                    loss = loss + self.grad_penalty_coef * grad_penalty
                    mean_smooth["grad_penalty"] += grad_penalty.item()
                if self.caps_temporal_coef > 0.0:
                    caps_temporal = self._caps_temporal(original_batch_size)
                    loss = loss + self.caps_temporal_coef * caps_temporal
                    mean_smooth["caps_temporal"] += caps_temporal.item()
                if self.caps_spatial_coef > 0.0:
                    caps_spatial = self._caps_spatial(batch.observations[:original_batch_size])
                    loss = loss + self.caps_spatial_coef * caps_spatial
                    mean_smooth["caps_spatial"] += caps_spatial.item()

                # RND loss
                rnd_loss = self.rnd.compute_loss(batch.observations[:original_batch_size]) if self.rnd else None  # type: ignore

                # Symmetry loss
                if self.symmetry:
                    symmetry_loss = self.symmetry.compute_loss(self.actor, batch, original_batch_size)
                    if self.symmetry.use_mirror_loss:
                        loss = loss + self.symmetry.mirror_loss_coeff * symmetry_loss

            # Compute the gradients for PPO
            self.optimizer.zero_grad()
            loss.backward()
            # Compute the gradients for RND
            if self.rnd:
                self.rnd.optimizer.zero_grad()
                rnd_loss.backward()

            # Collect gradients from all GPUs
            if self.is_multi_gpu:
                self.reduce_parameters()

            # Apply the gradients for PPO
            nn.utils.clip_grad_norm_(self.actor.parameters(), self.max_grad_norm)
            nn.utils.clip_grad_norm_(self.critic.parameters(), self.max_grad_norm)
            self.optimizer.step()
            # Apply the gradients for RND
            if self.rnd:
                self.rnd.optimizer.step()

            # Store the losses
            mean_value_loss += value_loss.item()
            mean_surrogate_loss += surrogate_loss.item()
            mean_entropy += entropy.mean().item()
            # RND loss
            if mean_rnd_loss is not None:
                mean_rnd_loss += rnd_loss.item()
            # Symmetry loss
            if mean_symmetry_loss is not None:
                mean_symmetry_loss += symmetry_loss.item()

        # Update the normalizers
        obs = self.storage.observations.flatten(0, 1)
        self.actor.update_normalization(obs)  # type: ignore
        self.critic.update_normalization(obs)  # type: ignore
        if self.rnd:
            self.rnd.update_normalization(obs)  # type: ignore

        # Divide the losses by the number of updates
        num_updates = self.num_learning_epochs * self.num_mini_batches
        mean_value_loss /= num_updates
        mean_surrogate_loss /= num_updates
        mean_entropy /= num_updates
        if mean_rnd_loss is not None:
            mean_rnd_loss /= num_updates
        if mean_symmetry_loss is not None:
            mean_symmetry_loss /= num_updates

        # Construct the loss dictionary
        loss_dict = {
            "value": mean_value_loss,
            "surrogate": mean_surrogate_loss,
            "entropy": mean_entropy,
        }
        if self.rnd:
            loss_dict["rnd"] = mean_rnd_loss
        if self.symmetry:
            loss_dict["symmetry"] = mean_symmetry_loss
        # [P1.4] logged as Loss/grad_penalty, Loss/caps_temporal, Loss/caps_spatial
        for name, coef in (
            ("grad_penalty", self.grad_penalty_coef),
            ("caps_temporal", self.caps_temporal_coef),
            ("caps_spatial", self.caps_spatial_coef),
        ):
            if coef > 0.0:
                loss_dict[name] = mean_smooth[name] / num_updates

        # Clear the storage
        self.storage.clear()

        return loss_dict
