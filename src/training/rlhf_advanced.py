"""
RLHF 2.0: Reinforcement Learning from Everything
Implements multi-objective optimization with various feedback sources
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
from typing import Dict, List, Tuple, Optional, Any, Union
from dataclasses import dataclass, field
from enum import Enum
import numpy as np
from collections import deque
import logging
from tqdm import tqdm

logger = logging.getLogger(__name__)


class FeedbackSource(Enum):
    """Sources of feedback for RLHF"""
    HUMAN = "human"
    AI_FEEDBACK = "ai_feedback"
    TOOL_EXECUTION = "tool_execution"
    CONSTITUTIONAL = "constitutional"
    SELF_CONSISTENCY = "self_consistency"
    PROCESS_SUPERVISION = "process_supervision"


class RewardObjective(Enum):
    """Different reward objectives to optimize"""
    HELPFULNESS = "helpfulness"
    HARMLESSNESS = "harmlessness"
    HONESTY = "honesty"
    ACCURACY = "accuracy"
    COHERENCE = "coherence"
    CREATIVITY = "creativity"
    EFFICIENCY = "efficiency"


@dataclass
class RLHFConfig:
    """Configuration for RLHF 2.0"""
    # Optimization
    ppo_epochs: int = 4
    ppo_batch_size: int = 32
    ppo_mini_batch_size: int = 8
    gradient_accumulation_steps: int = 4
    
    # PPO hyperparameters
    clip_range: float = 0.2
    value_clip_range: float = 0.2
    max_grad_norm: float = 1.0
    target_kl: float = 0.01
    gae_lambda: float = 0.95
    discount_factor: float = 0.99
    
    # Reward weights
    reward_weights: Dict[RewardObjective, float] = field(default_factory=lambda: {
        RewardObjective.HELPFULNESS: 1.0,
        RewardObjective.HARMLESSNESS: 1.5,
        RewardObjective.HONESTY: 1.2,
        RewardObjective.ACCURACY: 1.3,
        RewardObjective.COHERENCE: 0.8,
        RewardObjective.CREATIVITY: 0.5,
        RewardObjective.EFFICIENCY: 0.3
    })
    
    # DPO settings
    use_dpo: bool = True
    dpo_beta: float = 0.1
    
    # Process supervision
    enable_process_supervision: bool = True
    step_reward_weight: float = 0.5
    
    # Multi-objective
    use_pareto_optimization: bool = True
    num_preference_samples: int = 100
    
    # Buffer settings
    buffer_size: int = 10000
    min_buffer_size: int = 1000


@dataclass
class Experience:
    """Single experience in RLHF training"""
    states: torch.Tensor
    actions: torch.Tensor
    rewards: Dict[RewardObjective, float]
    next_states: torch.Tensor
    dones: torch.Tensor
    log_probs: torch.Tensor
    values: torch.Tensor
    advantages: Optional[torch.Tensor] = None
    returns: Optional[torch.Tensor] = None
    feedback_source: FeedbackSource = FeedbackSource.HUMAN
    metadata: Dict[str, Any] = field(default_factory=dict)


class MultiObjectiveRewardModel(nn.Module):
    """Multi-objective reward model for different aspects"""
    
    def __init__(self, hidden_dim: int = 768, num_objectives: int = 7):
        super().__init__()
        
        self.objectives = list(RewardObjective)
        
        # Shared encoder
        self.shared_encoder = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU()
        )
        
        # Per-objective reward heads
        self.reward_heads = nn.ModuleDict({
            obj.value: nn.Sequential(
                nn.Linear(hidden_dim // 2, 256),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(256, 64),
                nn.ReLU(),
                nn.Linear(64, 1)
            )
            for obj in RewardObjective
        })
        
        # Preference predictor for Pareto optimization
        self.preference_net = nn.Sequential(
            nn.Linear(num_objectives * 2, 128),
            nn.ReLU(),
            nn.Linear(128, num_objectives),
            nn.Softmax(dim=-1)
        )
        
    def forward(
        self,
        states: torch.Tensor,
        return_all_objectives: bool = True
    ) -> Union[torch.Tensor, Dict[RewardObjective, torch.Tensor]]:
        """Compute rewards for all objectives"""
        
        # Pool if needed
        if len(states.shape) == 3:
            states = states.mean(dim=1)
        
        # Shared encoding
        encoded = self.shared_encoder(states)
        
        # Per-objective rewards
        rewards = {}
        for objective in RewardObjective:
            reward = self.reward_heads[objective.value](encoded)
            rewards[objective] = reward.squeeze(-1)
        
        if return_all_objectives:
            return rewards
        else:
            # Return weighted sum
            return self.combine_rewards(rewards)
    
    def combine_rewards(
        self,
        rewards: Dict[RewardObjective, torch.Tensor],
        weights: Optional[Dict[RewardObjective, float]] = None
    ) -> torch.Tensor:
        """Combine multi-objective rewards"""
        if weights is None:
            # Equal weighting
            weights = {obj: 1.0 / len(RewardObjective) for obj in RewardObjective}
        
        combined = torch.zeros_like(next(iter(rewards.values())))
        for obj, reward in rewards.items():
            combined += weights.get(obj, 0.0) * reward
        
        return combined
    
    def predict_preferences(
        self,
        rewards1: Dict[RewardObjective, torch.Tensor],
        rewards2: Dict[RewardObjective, torch.Tensor]
    ) -> torch.Tensor:
        """Predict human preference between two sets of rewards"""
        # Stack rewards
        r1 = torch.stack([rewards1[obj] for obj in RewardObjective], dim=-1)
        r2 = torch.stack([rewards2[obj] for obj in RewardObjective], dim=-1)
        
        # Concatenate for preference prediction
        combined = torch.cat([r1, r2], dim=-1)
        
        # Predict preference weights
        preferences = self.preference_net(combined)
        
        return preferences


class ProcessSupervisor(nn.Module):
    """Process supervision for step-by-step reasoning evaluation"""
    
    def __init__(self, hidden_dim: int = 768):
        super().__init__()
        
        # Step evaluator
        self.step_evaluator = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )
        
        # Coherence checker
        self.coherence_checker = nn.Sequential(
            nn.Linear(hidden_dim * 3, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        )
        
        # Error detector
        self.error_detector = nn.Sequential(
            nn.Linear(hidden_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )
        
    def evaluate_step(
        self,
        current_step: torch.Tensor,
        previous_step: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Evaluate quality of reasoning step"""
        if previous_step is None:
            # First step
            previous_step = torch.zeros_like(current_step)
        
        # Concatenate current and previous
        combined = torch.cat([current_step, previous_step], dim=-1)
        
        # Evaluate step quality
        step_quality = self.step_evaluator(combined)
        
        return step_quality
    
    def check_coherence(
        self,
        steps: List[torch.Tensor]
    ) -> torch.Tensor:
        """Check coherence across multiple steps"""
        if len(steps) < 2:
            return torch.ones(1)
        
        # Check consecutive pairs
        coherence_scores = []
        for i in range(len(steps) - 1):
            if i == 0:
                # Include context for first pair
                context = torch.zeros_like(steps[0])
            else:
                context = steps[i - 1]
            
            combined = torch.cat([context, steps[i], steps[i + 1]], dim=-1)
            score = self.coherence_checker(combined)
            coherence_scores.append(score)
        
        # Average coherence
        return torch.stack(coherence_scores).mean()
    
    def detect_errors(self, step: torch.Tensor) -> torch.Tensor:
        """Detect errors in reasoning step"""
        error_probability = self.error_detector(step)
        return error_probability


class DirectPreferenceOptimization(nn.Module):
    """DPO for direct preference learning without explicit reward model"""
    
    def __init__(self, beta: float = 0.1):
        super().__init__()
        self.beta = beta
        
    def compute_dpo_loss(
        self,
        policy_chosen_logps: torch.Tensor,
        policy_rejected_logps: torch.Tensor,
        reference_chosen_logps: torch.Tensor,
        reference_rejected_logps: torch.Tensor
    ) -> torch.Tensor:
        """Compute DPO loss"""
        # Compute log ratios
        chosen_ratio = policy_chosen_logps - reference_chosen_logps
        rejected_ratio = policy_rejected_logps - reference_rejected_logps
        
        # DPO loss
        loss = -F.logsigmoid(self.beta * (chosen_ratio - rejected_ratio)).mean()
        
        return loss


class ExperienceBuffer:
    """Experience replay buffer for RLHF"""
    
    def __init__(self, capacity: int = 10000):
        self.capacity = capacity
        self.buffer = deque(maxlen=capacity)
        self.priorities = deque(maxlen=capacity)
        
    def push(self, experience: Experience, priority: float = 1.0):
        """Add experience to buffer"""
        self.buffer.append(experience)
        self.priorities.append(priority)
        
    def sample(self, batch_size: int, prioritized: bool = True) -> List[Experience]:
        """Sample batch from buffer"""
        if len(self.buffer) < batch_size:
            return list(self.buffer)
        
        if prioritized and self.priorities:
            # Prioritized sampling
            priorities = np.array(self.priorities)
            probabilities = priorities / priorities.sum()
            indices = np.random.choice(len(self.buffer), batch_size, p=probabilities)
        else:
            # Uniform sampling
            indices = np.random.choice(len(self.buffer), batch_size)
        
        return [self.buffer[i] for i in indices]
    
    def __len__(self):
        return len(self.buffer)


class PPOTrainer:
    """PPO trainer for RLHF optimization"""
    
    def __init__(
        self,
        policy_model: nn.Module,
        value_model: nn.Module,
        reward_model: MultiObjectiveRewardModel,
        config: RLHFConfig
    ):
        self.policy = policy_model
        self.value = value_model
        self.reward_model = reward_model
        self.config = config
        
        # Optimizers
        self.policy_optimizer = torch.optim.Adam(
            policy_model.parameters(), lr=3e-5, eps=1e-8
        )
        self.value_optimizer = torch.optim.Adam(
            value_model.parameters(), lr=1e-4, eps=1e-8
        )
        
        # DPO if enabled
        self.dpo = DirectPreferenceOptimization(config.dpo_beta) if config.use_dpo else None
        
        # Process supervisor
        self.process_supervisor = ProcessSupervisor() if config.enable_process_supervision else None
        
        # Experience buffer
        self.buffer = ExperienceBuffer(config.buffer_size)
        
        # Stats tracking
        self.stats = {
            'policy_loss': [],
            'value_loss': [],
            'rewards': [],
            'kl_divergence': []
        }
        
    def compute_advantages(
        self,
        rewards: torch.Tensor,
        values: torch.Tensor,
        next_values: torch.Tensor,
        dones: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute GAE advantages and returns"""
        advantages = torch.zeros_like(rewards)
        last_advantage = 0
        
        for t in reversed(range(rewards.shape[0])):
            if t == rewards.shape[0] - 1:
                next_value = next_values[t]
            else:
                next_value = values[t + 1]
            
            delta = rewards[t] + self.config.discount_factor * next_value * (1 - dones[t]) - values[t]
            advantages[t] = delta + self.config.discount_factor * self.config.gae_lambda * (1 - dones[t]) * last_advantage
            last_advantage = advantages[t]
        
        returns = advantages + values
        
        # Normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        return advantages, returns
    
    def train_step(self, experiences: List[Experience]) -> Dict[str, float]:
        """Single PPO training step"""
        # Stack experiences
        states = torch.stack([e.states for e in experiences])
        actions = torch.stack([e.actions for e in experiences])
        old_log_probs = torch.stack([e.log_probs for e in experiences])
        advantages = torch.stack([e.advantages for e in experiences])
        returns = torch.stack([e.returns for e in experiences])
        
        # PPO epochs
        total_policy_loss = 0
        total_value_loss = 0
        total_kl = 0
        
        for epoch in range(self.config.ppo_epochs):
            # Get new log probs and values
            policy_output = self.policy(states)
            new_log_probs = self.compute_log_probs(policy_output, actions)
            new_values = self.value(states).squeeze(-1)
            
            # Compute ratios
            ratios = torch.exp(new_log_probs - old_log_probs)
            
            # Clipped surrogate loss
            surr1 = ratios * advantages
            surr2 = torch.clamp(ratios, 1 - self.config.clip_range, 1 + self.config.clip_range) * advantages
            policy_loss = -torch.min(surr1, surr2).mean()
            
            # Value loss
            value_pred_clipped = old_values + torch.clamp(
                new_values - old_values,
                -self.config.value_clip_range,
                self.config.value_clip_range
            )
            value_loss1 = F.mse_loss(new_values, returns)
            value_loss2 = F.mse_loss(value_pred_clipped, returns)
            value_loss = torch.max(value_loss1, value_loss2)
            
            # KL divergence for early stopping
            kl = (old_log_probs - new_log_probs).mean()
            
            if kl > self.config.target_kl:
                logger.info(f"Early stopping at epoch {epoch} due to KL divergence")
                break
            
            # Backward pass
            self.policy_optimizer.zero_grad()
            policy_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.policy.parameters(), self.config.max_grad_norm)
            self.policy_optimizer.step()
            
            self.value_optimizer.zero_grad()
            value_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.value.parameters(), self.config.max_grad_norm)
            self.value_optimizer.step()
            
            total_policy_loss += policy_loss.item()
            total_value_loss += value_loss.item()
            total_kl += kl.item()
        
        return {
            'policy_loss': total_policy_loss / (epoch + 1),
            'value_loss': total_value_loss / (epoch + 1),
            'kl_divergence': total_kl / (epoch + 1)
        }
    
    def compute_log_probs(self, logits: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        """Compute log probabilities of actions"""
        probs = F.softmax(logits, dim=-1)
        dist = Categorical(probs)
        return dist.log_prob(actions)
    
    def collect_experience(
        self,
        env,
        num_steps: int = 1000
    ) -> List[Experience]:
        """Collect experience from environment"""
        experiences = []
        
        state = env.reset()
        
        for step in range(num_steps):
            # Get action from policy
            with torch.no_grad():
                policy_output = self.policy(state.unsqueeze(0))
                probs = F.softmax(policy_output, dim=-1)
                dist = Categorical(probs)
                action = dist.sample()
                log_prob = dist.log_prob(action)
                
                # Get value estimate
                value = self.value(state.unsqueeze(0)).squeeze()
            
            # Step environment
            next_state, reward_dict, done, info = env.step(action.item())
            
            # Create experience
            exp = Experience(
                states=state,
                actions=action,
                rewards=reward_dict,
                next_states=next_state,
                dones=torch.tensor(done, dtype=torch.float),
                log_probs=log_prob,
                values=value,
                feedback_source=info.get('feedback_source', FeedbackSource.HUMAN),
                metadata=info
            )
            
            experiences.append(exp)
            
            if done:
                state = env.reset()
            else:
                state = next_state
        
        # Compute advantages and returns
        self._compute_experience_advantages(experiences)
        
        return experiences
    
    def _compute_experience_advantages(self, experiences: List[Experience]):
        """Compute advantages for collected experiences"""
        # Convert rewards to combined scalar
        rewards = []
        for exp in experiences:
            combined_reward = self.reward_model.combine_rewards(
                exp.rewards,
                self.config.reward_weights
            )
            rewards.append(combined_reward)
        
        rewards = torch.stack(rewards)
        values = torch.stack([e.values for e in experiences])
        
        # Get next values
        next_values = values.clone()
        next_values[:-1] = values[1:]
        next_values[-1] = 0  # Terminal state
        
        dones = torch.stack([e.dones for e in experiences])
        
        # Compute advantages
        advantages, returns = self.compute_advantages(rewards, values, next_values, dones)
        
        # Update experiences
        for i, exp in enumerate(experiences):
            exp.advantages = advantages[i]
            exp.returns = returns[i]


class RLHF2System:
    """Complete RLHF 2.0 system with all components"""
    
    def __init__(
        self,
        base_model: nn.Module,
        config: RLHFConfig
    ):
        self.base_model = base_model
        self.config = config
        
        # Components
        self.reward_model = MultiObjectiveRewardModel()
        
        # Value model (can be separate or shared with base model)
        self.value_model = nn.Sequential(
            nn.Linear(base_model.config.n_embd, 512),
            nn.ReLU(),
            nn.Linear(512, 1)
        )
        
        # PPO trainer
        self.ppo_trainer = PPOTrainer(
            base_model,
            self.value_model,
            self.reward_model,
            config
        )
        
        # Process supervisor if enabled
        if config.enable_process_supervision:
            self.process_supervisor = ProcessSupervisor()
        
        # AI feedback models
        self.ai_feedback_models = {}
        
    def add_ai_feedback_model(self, name: str, model: nn.Module):
        """Add AI model for feedback generation"""
        self.ai_feedback_models[name] = model
        
    def train_reward_model(
        self,
        preference_data,
        num_epochs: int = 3
    ):
        """Train reward model on preference data"""
        optimizer = torch.optim.Adam(self.reward_model.parameters(), lr=1e-4)
        
        for epoch in range(num_epochs):
            total_loss = 0
            num_batches = 0
            
            for batch in preference_data:
                chosen = batch['chosen']
                rejected = batch['rejected']
                
                # Get rewards
                chosen_rewards = self.reward_model(chosen)
                rejected_rewards = self.reward_model(rejected)
                
                # Preference loss
                combined_chosen = self.reward_model.combine_rewards(
                    chosen_rewards, self.config.reward_weights
                )
                combined_rejected = self.reward_model.combine_rewards(
                    rejected_rewards, self.config.reward_weights
                )
                
                # Bradley-Terry loss
                loss = -F.logsigmoid(combined_chosen - combined_rejected).mean()
                
                # Backward
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
                num_batches += 1
                
            avg_loss = total_loss / num_batches
            logger.info(f"Reward model epoch {epoch}: loss = {avg_loss:.4f}")
    
    def train_with_dpo(
        self,
        preference_data,
        reference_model: nn.Module,
        num_epochs: int = 3
    ):
        """Train using Direct Preference Optimization"""
        if not self.config.use_dpo:
            logger.warning("DPO not enabled in config")
            return
        
        optimizer = torch.optim.Adam(self.base_model.parameters(), lr=3e-5)
        
        for epoch in range(num_epochs):
            total_loss = 0
            num_batches = 0
            
            for batch in preference_data:
                chosen = batch['chosen']
                rejected = batch['rejected']
                
                # Get log probs from policy
                with torch.no_grad():
                    ref_chosen_logps = reference_model(chosen)['logits'].log_softmax(-1)
                    ref_rejected_logps = reference_model(rejected)['logits'].log_softmax(-1)
                
                policy_chosen_logps = self.base_model(chosen)['logits'].log_softmax(-1)
                policy_rejected_logps = self.base_model(rejected)['logits'].log_softmax(-1)
                
                # DPO loss
                loss = self.ppo_trainer.dpo.compute_dpo_loss(
                    policy_chosen_logps.mean(),
                    policy_rejected_logps.mean(),
                    ref_chosen_logps.mean(),
                    ref_rejected_logps.mean()
                )
                
                # Backward
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
                num_batches += 1
            
            avg_loss = total_loss / num_batches
            logger.info(f"DPO epoch {epoch}: loss = {avg_loss:.4f}")
    
    def run_rlhf_loop(
        self,
        env,
        num_iterations: int = 100,
        steps_per_iteration: int = 1000
    ):
        """Main RLHF training loop"""
        
        for iteration in range(num_iterations):
            logger.info(f"RLHF iteration {iteration}")
            
            # Collect experience
            experiences = self.ppo_trainer.collect_experience(env, steps_per_iteration)
            
            # Add to buffer
            for exp in experiences:
                priority = abs(exp.advantages.item()) if exp.advantages is not None else 1.0
                self.ppo_trainer.buffer.push(exp, priority)
            
            # Train if buffer is ready
            if len(self.ppo_trainer.buffer) >= self.config.min_buffer_size:
                # Sample from buffer
                batch = self.ppo_trainer.buffer.sample(self.config.ppo_batch_size)
                
                # PPO training step
                metrics = self.ppo_trainer.train_step(batch)
                
                logger.info(f"Iteration {iteration} metrics: {metrics}")
                
                # Update stats
                for key, value in metrics.items():
                    self.ppo_trainer.stats[key].append(value)
            
            # Periodic evaluation
            if iteration % 10 == 0:
                self.evaluate(env)
    
    def evaluate(self, env, num_episodes: int = 10):
        """Evaluate current policy"""
        total_rewards = {obj: 0.0 for obj in RewardObjective}
        
        for episode in range(num_episodes):
            state = env.reset()
            done = False
            episode_rewards = {obj: 0.0 for obj in RewardObjective}
            
            while not done:
                with torch.no_grad():
                    action = self.base_model(state.unsqueeze(0)).argmax(dim=-1)
                
                next_state, reward_dict, done, _ = env.step(action.item())
                
                for obj, reward in reward_dict.items():
                    episode_rewards[obj] += reward
                
                state = next_state
            
            for obj in RewardObjective:
                total_rewards[obj] += episode_rewards[obj]
        
        # Average rewards
        avg_rewards = {obj: total / num_episodes for obj, total in total_rewards.items()}
        
        logger.info(f"Evaluation results: {avg_rewards}")
        
        return avg_rewards
