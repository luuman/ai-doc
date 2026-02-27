# 强化学习

强化学习（Reinforcement Learning，RL）是机器学习的第三大范式，与监督学习和无监督学习根本不同：它通过智能体与环境的持续交互，依据奖励信号自主学习最优行为策略。RL 是 AlphaGo 战胜人类围棋冠军的技术核心，也是 ChatGPT 等大语言模型对齐人类价值观的关键方法（RLHF）。

## RL 基本框架

强化学习的核心要素：

- **智能体（Agent）**：做出决策的主体（机器人、游戏玩家、语言模型）
- **环境（Environment）**：智能体交互的外部世界（棋盘、物理世界、用户）
- **状态（State, s）**：当前时刻对环境的完整描述（棋盘局面、机器人位置、对话历史）
- **动作（Action, a）**：智能体可执行的操作（移动棋子、控制关节、输出 token）
- **奖励（Reward, r）**：环境对智能体行为的即时反馈（赢棋 +1，输棋 -1，每步 -0.01）
- **策略（Policy, π）**：从状态到动作的映射，`π(a|s)` 表示在状态 s 下选择动作 a 的概率

**交互循环**：

```
时间步 t：
1. 智能体观察状态 s_t
2. 依据策略选择动作 a_t = π(s_t)（或 a_t ~ π(·|s_t)）
3. 环境返回下一状态 s_{t+1} 和奖励 r_t
4. 智能体更新策略以最大化累积奖励
```

**累积折扣奖励（Return）**：

```
G_t = r_t + γ·r_{t+1} + γ²·r_{t+2} + ...
    = Σ_{k=0}^∞ γᵏ · r_{t+k}

折扣因子 γ ∈ [0,1]：控制对未来奖励的重视程度
- γ接近0：只关心即时奖励（短视）
- γ接近1：充分考虑长期收益（有远见）
```

## 马尔可夫决策过程（MDP）

RL 的数学形式化框架是马尔可夫决策过程，由五元组 `(S, A, P, R, γ)` 定义：

- **S**：状态空间
- **A**：动作空间
- **P(s'|s,a)**：状态转移概率（环境动态）
- **R(s,a,s')**：奖励函数
- **γ**：折扣因子

**马尔可夫性质**：下一状态只依赖当前状态和动作，与历史无关。

```
P(s_{t+1} | s_t, a_t, s_{t-1}, a_{t-1}, ...) = P(s_{t+1} | s_t, a_t)
```

这一假设简化了分析，但现实中有时需要部分可观测 MDP（POMDP）来处理不完整信息。

**值函数**：评估状态或状态-动作对的长期价值：

```
状态值函数 V^π(s) = E_π[G_t | s_t = s]   ← 从状态 s 出发，遵循策略 π 的期望回报
动作值函数 Q^π(s,a) = E_π[G_t | s_t=s, a_t=a]  ← 在 s 执行 a 后，遵循 π 的期望回报

贝尔曼方程（递推关系）：
V^π(s) = Σ_a π(a|s) Σ_{s'} P(s'|s,a) [R(s,a,s') + γV^π(s')]
```

## Q-learning

Q-learning 是最基础的无模型值迭代算法，直接学习动作值函数 Q(s,a)：

**Q 表更新规则（Bellman 更新）**：

```
Q(s_t, a_t) ← Q(s_t, a_t) + α [r_t + γ · max_a Q(s_{t+1}, a) - Q(s_t, a_t)]

其中：
- α：学习率（0 到 1）
- r_t + γ·max_a Q(s_{t+1}, a)：TD 目标（Temporal Difference Target）
- [TD 目标 - Q(s_t, a_t)]：TD 误差
```

**ε-greedy 探索策略**：

```python
import random

def epsilon_greedy(Q, state, epsilon=0.1):
    if random.random() < epsilon:
        return env.action_space.sample()  # 随机探索
    else:
        return Q[state].argmax()          # 贪心利用
```

Q-learning 适用于状态空间和动作空间都是离散且有限的简单环境（如格子世界、简单游戏）。

## 深度 Q 网络（DQN）

当状态空间过大（如 Atari 游戏的像素图像：84×84×4）时，Q 表无法存储，需要用神经网络近似 Q 函数：

```python
import torch.nn as nn

class DQN(nn.Module):
    def __init__(self, state_dim, action_dim):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(4, 32, 8, stride=4), nn.ReLU(),
            nn.Conv2d(32, 64, 4, stride=2), nn.ReLU(),
            nn.Conv2d(64, 64, 3, stride=1), nn.ReLU(),
        )
        self.fc = nn.Sequential(
            nn.Linear(64*7*7, 512), nn.ReLU(),
            nn.Linear(512, action_dim)
        )

    def forward(self, x):
        x = self.conv(x / 255.0)
        return self.fc(x.flatten(1))
```

DeepMind 的 DQN（2015, Nature）的关键技术：

- **经验回放（Experience Replay）**：将 `(s, a, r, s')` 存入回放缓冲区，随机采样小批量打破时间相关性
- **目标网络（Target Network）**：用缓慢更新的副本计算 TD 目标，防止训练不稳定
- **帧堆叠**：将连续 4 帧叠加作为状态，让网络感知运动方向

DQN 在 49 款 Atari 游戏中超越人类水平，是深度 RL 的里程碑。

## Policy Gradient——REINFORCE 算法

与值函数方法不同，策略梯度方法直接对策略 π_θ 进行参数化和优化：

**目标**：最大化期望总回报 `J(θ) = E_π[G_t]`

**策略梯度定理**：

```
∇_θ J(θ) = E_π [∇_θ log π_θ(a_t|s_t) · G_t]
```

**REINFORCE 算法**：

```python
def compute_policy_gradient(states, actions, returns, policy):
    log_probs = policy.log_prob(states, actions)
    # 减去基线（baseline）减少方差
    baseline = returns.mean()
    loss = -(log_probs * (returns - baseline)).mean()
    return loss
```

REINFORCE 的缺点：方差高（回报估计噪声大），样本效率低（每次 rollout 只用一次）。

## Actor-Critic 架构

Actor-Critic 结合了策略梯度（Actor，负责选择动作）和值函数（Critic，评估动作质量）：

```
Actor π_θ(a|s)：参数化策略，决定动作
Critic V_φ(s)：估计状态值，计算优势函数

优势函数（Advantage）：A(s,a) = Q(s,a) - V(s)
  ← 动作 a 比平均水平好多少？正值说明当前动作优于平均

Actor 更新：∇_θ J ∝ ∇_θ log π_θ(a|s) · A(s,a)
Critic 更新：最小化 (V_φ(s) - G_t)²
```

优势函数减少了梯度估计的方差，使训练更稳定。

## PPO——近端策略优化

PPO（Proximal Policy Optimization，Schulman et al., 2017）是目前最广泛使用的深度 RL 算法，被 OpenAI 用于 RLHF 对齐大语言模型：

**核心思想**：限制每次策略更新的幅度，防止更新过大导致训练崩溃：

```
PPO-Clip 目标：
L_CLIP(θ) = E_t [min(r_t(θ)·A_t, clip(r_t(θ), 1-ε, 1+ε)·A_t)]

其中：
r_t(θ) = π_θ(a_t|s_t) / π_θ_old(a_t|s_t)  ← 新旧策略的概率比
ε = 0.2（通常）  ← 裁剪范围
```

当概率比 r_t 超出 `[1-ε, 1+ε]` 时，梯度被截断，防止策略变化过剧。

**PPO 在 RLHF 中的应用**：

```
RLHF 训练流程：
1. 预训练语言模型 LM_ref（SFT 阶段）
2. 训练奖励模型 R_φ（从人类偏好对比数据学习）
3. 用 PPO 优化语言模型 LM_θ：
   目标 = E[R_φ(x,y)] - β·KL(LM_θ(y|x) || LM_ref(y|x))
   ← KL 惩罚防止语言模型偏离参考模型太远（避免奖励欺骗）
```

## AlphaGo 与 AlphaZero 技术解析

### AlphaGo（2016）

DeepMind 的 AlphaGo 结合了三项核心技术：

- **监督学习策略网络**：从人类棋谱学习合理落子，提供高质量搜索先验
- **自博弈强化学习**：在监督学习的基础上自我对弈，持续改进
- **蒙特卡罗树搜索（MCTS）**：结合策略网络（指导搜索方向）和价值网络（评估棋局胜率），比传统暴力搜索高效百倍

### AlphaZero（2017）

AlphaZero 完全抛弃人类棋谱，从零开始纯靠自博弈训练，在围棋、国际象棋、将棋三项游戏上全面超越各自领域最强引擎：

- 只告诉规则，不告诉任何人类知识
- 单一网络同时输出策略（各落点的概率）和价值（当前局面的胜率估计）
- 证明了在规则明确的复杂博弈中，自监督 RL 可以超越人类积累的所有知识

AlphaZero 的思路被 AlphaFold（蛋白质结构预测）、MuZero（无需规则的 MCTS）等后续工作广泛继承和扩展。
