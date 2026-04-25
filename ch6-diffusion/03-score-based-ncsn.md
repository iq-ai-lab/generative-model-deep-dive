# 03. Score-Based Model — NCSN (Song & Ermon 2019)

## 🎯 핵심 질문

- **Score** $s(x) = \nabla_x \log p(x)$ 의 의미와 왜 generative modeling 에 사용되는가?
- Langevin dynamics $x_{t+1} = x_t + (\delta/2) s_\theta(x_t) + \sqrt{\delta} z$ 가 어떻게 sampling 을 가능케 하는가?
- Score matching (Hyvärinen 2005) 의 implicit form 과 그 한계 — 왜 직접 사용 어려운가?
- Denoising Score Matching (Vincent 2011): $\mathbb{E}_{q_\sigma(\tilde x | x)}[\|s_\theta - \nabla_{\tilde x} \log q_\sigma\|^2]$ 의 정당성?
- **DDPM = weighted Denoising Score Matching** 의 동치 — 두 framework 의 통합?

---

## 🔍 왜 Score-Based Model 이 결정적인가

Song & Ermon 2019 "Generative Modeling by Estimating Gradients of the Data Distribution" — diffusion 과 다른 출발점이지만 **같은 framework 으로 통합** 되는 generative model:

1. **Score** $\nabla \log p$ 직접 학습 — partition function $Z$ 회피
2. **Langevin dynamics** 로 sampling — score 만 알면 가능
3. **Multi-scale noise**: NCSN (Noise-Conditioned Score Network)
4. **DDPM 과 equivalence**: $\epsilon_\theta$ 학습 = score 학습 (weighted)

이 framework 의 통합으로 **Score-SDE** (Song 2021, Ch6-04) 가 등장 — DDPM, NCSN, 모든 변형이 하나의 SDE 로 표현. 이 문서에서는 score 의 의미, score matching 의 수학, 그리고 DDPM 과의 동치를 다룹니다.

---

## 📐 수학적 선행 조건

- 이전 문서들: 01-ddpm, 02-simple-loss
- [Stochastic Differential Equations Deep Dive](https://github.com/iq-ai-lab/sde-deep-dive): Langevin dynamics, Fokker-Planck
- [Information Theory Deep Dive](https://github.com/iq-ai-lab/information-theory-deep-dive): KL, Fisher information

---

## 📖 직관적 이해

### "Score 가 데이터 manifold 로의 방향"

**Score** $\nabla_x \log p(x)$: density 가 증가하는 방향 (data manifold 쪽으로 향하는 vector field).

- $x$ 가 high-density region: score 작음 (이미 manifold 위)
- $x$ 가 low-density: score 큼, manifold 쪽으로 strong push

**Sampling**: random $x$ 에서 시작, score 따라 step → eventually data manifold.

### Langevin Dynamics 의 직관

$$x_{t+1} = x_t + \frac{\delta}{2} \nabla_x \log p(x_t) + \sqrt{\delta} z_t, \quad z_t \sim \mathcal{N}(0, I)$$

- **Drift** $(\delta/2) \nabla \log p$: density 가 큰 곳으로 push
- **Noise** $\sqrt{\delta} z$: exploration, 평형분포 유지

$\delta \to 0$ 의 limit: continuous-time SDE $dx = (1/2) \nabla \log p \, dt + dW$. **stationary distribution = $p$**.

따라서 "score 만 알면 sampling 가능" — partition function $Z$ 회피.

### Score Matching 의 어려움

표준 score matching (Hyvärinen 2005):

$$\mathcal{L} = \mathbb{E}_p[\|\nabla_x \log p_\theta(x) - \nabla_x \log p(x)\|^2]$$

문제: $\nabla \log p$ 모름 (학습하려는 것). 그러나 integration by parts 로:

$$\mathcal{L} = \mathbb{E}_p[\|\nabla \log p_\theta\|^2 + 2 \nabla \cdot \nabla \log p_\theta] + \text{const}$$

(divergence term — Jacobian trace).

$\nabla \cdot s_\theta = \text{tr}(\partial s_\theta / \partial x)$ — high-dim 에서 cost $O(d^2)$ (Hutchinson 으로 $O(d)$ 가능, FFJORD 와 유사).

**한계**: high-dim 에서 expensive, score 가 low-density region 에서 부정확.

### Denoising Score Matching: 우회 길

$q_\sigma(\tilde x | x) = \mathcal{N}(x, \sigma^2 I)$ (Gaussian noise). Perturbed distribution $q_\sigma(\tilde x) = \int q_\sigma(\tilde x | x) p_d(x) dx$.

**Vincent 2011**:

$$\mathcal{L}_\text{DSM} = \mathbb{E}_{p_d, q_\sigma(\tilde x | x)}[\|s_\theta(\tilde x) - \nabla_{\tilde x} \log q_\sigma(\tilde x | x)\|^2]$$

$\nabla_{\tilde x} \log q_\sigma(\tilde x | x) = -(\tilde x - x)/\sigma^2$ — **closed form** (Gaussian의 score).

따라서:

$$\mathcal{L}_\text{DSM} = \mathbb{E}\left[\left\|s_\theta(\tilde x) + \frac{\tilde x - x}{\sigma^2}\right\|^2\right]$$

이게 efficient — divergence 계산 회피.

---

## ✏️ 엄밀한 정의·정리

### 정의 3.1 — Score

분포 $p(x)$ ($p > 0$, smooth) 의 score:

$$s(x) := \nabla_x \log p(x) = \frac{\nabla_x p(x)}{p(x)}$$

**Important**: score 는 $\log Z$ (normalization) 에 의존 안 함. $p(x) = e^{-E(x)} / Z \Rightarrow s = -\nabla E$.

### 정의 3.2 — Langevin Dynamics

Discrete-time:

$$x_{t+1} = x_t + \frac{\delta}{2} s(x_t) + \sqrt{\delta} z_t, \quad z_t \sim \mathcal{N}(0, I)$$

Continuous-time SDE:

$$dx = \frac{1}{2} \nabla \log p(x) dt + dW$$

**Stationary distribution**: $p$ (Fokker-Planck).

**Convergence**: under regularity, $x_T \to p$ as $T \to \infty$.

### 정리 3.3 — Score Matching (Hyvärinen 2005)

$$\mathcal{L}_\text{SM}(\theta) = \mathbb{E}_p[\|s_\theta(x) - \nabla \log p(x)\|^2]$$

이를 (after integration by parts):

$$\mathcal{L}_\text{SM} = \mathbb{E}_p[\|s_\theta(x)\|^2 + 2 \nabla \cdot s_\theta(x)] + \text{const}$$

여기서 $\nabla \cdot s_\theta = \text{tr}(\partial s_\theta / \partial x)$ — divergence.

**증명** (sketch): $\mathbb{E}_p[\|s_\theta - \nabla \log p\|^2] = \mathbb{E}[\|s_\theta\|^2] - 2 \mathbb{E}[s_\theta \cdot \nabla \log p] + \text{const}$.

Cross term: $\mathbb{E}_p[s_\theta \cdot \nabla \log p] = \int p \cdot s_\theta \cdot (\nabla p / p) dx = \int s_\theta \cdot \nabla p dx = -\int p \cdot \nabla \cdot s_\theta dx = -\mathbb{E}_p[\nabla \cdot s_\theta]$ (integration by parts, assuming $p \to 0$ at boundary).

따라서 $\mathcal{L}_\text{SM} = \mathbb{E}_p[\|s_\theta\|^2 + 2 \nabla \cdot s_\theta] + \text{const}$. $\square$

### 정리 3.4 — Denoising Score Matching (Vincent 2011)

Smoothed distribution $q_\sigma(\tilde x) = \int q_\sigma(\tilde x | x) p_d(x) dx$. 그러면:

$$\mathcal{L}_\text{DSM}(\theta) = \mathbb{E}_{p_d(x) q_\sigma(\tilde x | x)}\left[\|s_\theta(\tilde x) - \nabla_{\tilde x} \log q_\sigma(\tilde x | x)\|^2\right]$$

**같은 minimum** as $\mathcal{L}_\text{SM}^{q_\sigma}$ (score matching for $q_\sigma$).

**증명**: expectation 의 manipulation.

$\mathbb{E}_{q_\sigma}[\|s_\theta(\tilde x) - \nabla \log q_\sigma(\tilde x)\|^2] = \mathbb{E}[\|s_\theta\|^2] - 2 \mathbb{E}[s_\theta \cdot \nabla \log q_\sigma] + \text{const}$.

Cross: $\mathbb{E}_{q_\sigma}[s_\theta \cdot \nabla \log q_\sigma] = \int q_\sigma s_\theta \cdot \frac{\nabla q_\sigma}{q_\sigma} = \int s_\theta \nabla q_\sigma$.

$q_\sigma(\tilde x) = \int q_\sigma(\tilde x | x) p_d(x) dx$, $\nabla q_\sigma = \int \nabla q_\sigma(\tilde x | x) p_d(x) dx$.

따라서:

$\mathbb{E}_{q_\sigma}[s_\theta \cdot \nabla \log q_\sigma] = \int s_\theta(\tilde x) \int \nabla q_\sigma(\tilde x | x) p_d(x) dx d\tilde x$

$= \int p_d(x) \int s_\theta(\tilde x) \nabla q_\sigma(\tilde x | x) d\tilde x dx$

$= \int p_d(x) \int q_\sigma(\tilde x | x) s_\theta(\tilde x) \nabla \log q_\sigma(\tilde x | x) d\tilde x dx$

$= \mathbb{E}_{p_d(x) q_\sigma(\tilde x | x)}[s_\theta(\tilde x) \cdot \nabla \log q_\sigma(\tilde x | x)]$

이를 $\mathcal{L}_\text{DSM}$ 에 대입하면 (after expansion):

$\mathcal{L}_\text{DSM} = \mathbb{E}_{p_d, q_\sigma(\tilde x | x)}[\|s_\theta(\tilde x)\|^2] - 2 \mathbb{E}[\ldots] + \text{const}$

= $\mathbb{E}_{q_\sigma}[\|s_\theta(\tilde x) - \nabla \log q_\sigma(\tilde x)\|^2]$ + const (after combining).

따라서 **same minimum**. $\square$

### 정리 3.5 — DDPM = Weighted DSM

DDPM 의 forward $q(x_t | x_0) = \mathcal{N}(\sqrt{\bar\alpha_t} x_0, (1-\bar\alpha_t) I)$.

**Score of forward**: $\nabla_{x_t} \log q(x_t | x_0) = -\frac{x_t - \sqrt{\bar\alpha_t} x_0}{1 - \bar\alpha_t}$.

$x_t = \sqrt{\bar\alpha_t} x_0 + \sqrt{1-\bar\alpha_t} \epsilon$ 대입:

$$\nabla_{x_t} \log q(x_t | x_0) = -\frac{\sqrt{1-\bar\alpha_t} \epsilon}{1 - \bar\alpha_t} = -\frac{\epsilon}{\sqrt{1-\bar\alpha_t}}$$

따라서 score $\propto -\epsilon$. **NN 의 $\epsilon$-prediction $\equiv$ score-prediction up to scaling**:

$$s_\theta(x_t, t) = -\frac{\epsilon_\theta(x_t, t)}{\sqrt{1-\bar\alpha_t}}$$

**$L_\text{simple}$ 과의 관계**:

$L_\text{simple} = \mathbb{E}[\|\epsilon - \epsilon_\theta\|^2]$. 위 관계 대입:

$$L_\text{simple} = \mathbb{E}[(1 - \bar\alpha_t) \|s_\theta - \nabla \log q\|^2]$$

= **weighted DSM** with weight $(1-\bar\alpha_t)$.

따라서 DDPM 과 NCSN 은 **equivalent up to weighting**.

### 정의 3.6 — NCSN (Song & Ermon 2019)

Multi-scale noise: $\sigma_1 > \sigma_2 > \cdots > \sigma_L$ 의 sequence (geometric: $\sigma_i = \sigma_1 \rho^{i-1}$).

NN $s_\theta(x, \sigma)$ — noise level conditioned.

Loss:

$$\mathcal{L}_\text{NCSN} = \frac{1}{L} \sum_{i=1}^L \sigma_i^2 \cdot \mathbb{E}_{p_d, q_{\sigma_i}}[\|s_\theta(\tilde x, \sigma_i) - \nabla \log q_{\sigma_i}\|^2]$$

(가중치 $\sigma_i^2$ 가 standard 약수).

**Annealed Langevin Sampling**:
1. $x \sim \mathcal{N}(0, \sigma_1^2 I)$ (large noise)
2. For $i = 1$ to $L$:
   - $T$ Langevin steps with $s_\theta(\cdot, \sigma_i)$, step size $\propto \sigma_i^2$
3. Output $x_T$

**효과**: 큰 noise 부터 점진적 denoising — DDPM 의 reverse 와 유사.

---

## 🔬 증명 및 수학적 유도

### 유도 1 — Langevin Dynamics 의 Stationary Distribution

Continuous SDE: $dx = (1/2) \nabla \log p \, dt + dW$.

**Fokker-Planck**: density $p_t(x)$ 의 시간 진화:

$$\frac{\partial p_t}{\partial t} = -\nabla \cdot \left(\frac{1}{2} \nabla \log p \cdot p_t\right) + \frac{1}{2} \nabla^2 p_t$$

**Stationary**: $\partial p_t / \partial t = 0$. Try $p_t = p$:

$-\nabla \cdot \left(\frac{p}{2} \cdot \frac{\nabla p}{p}\right) + \frac{1}{2} \nabla^2 p = -\frac{1}{2} \nabla^2 p + \frac{1}{2} \nabla^2 p = 0$. ✓

따라서 $p$ 가 stationary. Convergence under regularity.

### 유도 2 — Score Matching 의 Integration by Parts

$\mathbb{E}_p[s_\theta \cdot \nabla \log p] = \int p \cdot s_\theta \cdot (\nabla p / p) dx = \int s_\theta \cdot \nabla p \, dx$.

Multivariate integration by parts: $\int s_\theta \cdot \nabla p \, dx = [s_\theta \cdot p]_\text{boundary} - \int p \cdot \nabla \cdot s_\theta \, dx$.

Boundary $= 0$ (assuming $p$ vanishes at $\infty$):

$= -\mathbb{E}_p[\nabla \cdot s_\theta]$

**시사점**: data 의 score 를 직접 사용 안 함, only $s_\theta$ 의 norm 과 divergence — tractable.

### 유도 3 — DSM 의 Closed-Form Target

$q_\sigma(\tilde x | x) = \mathcal{N}(x, \sigma^2 I)$, density:

$$\log q_\sigma(\tilde x | x) = -\frac{\|\tilde x - x\|^2}{2\sigma^2} + \text{const}$$

$$\nabla_{\tilde x} \log q_\sigma(\tilde x | x) = -\frac{\tilde x - x}{\sigma^2}$$

직접 연산 가능. DSM:

$$\mathcal{L}_\text{DSM} = \mathbb{E}\left[\left\|s_\theta(\tilde x, \sigma) + \frac{\tilde x - x}{\sigma^2}\right\|^2\right]$$

Reparameterize: $\tilde x = x + \sigma \epsilon$, $\epsilon \sim \mathcal{N}(0, I)$. $\nabla \log q = -\epsilon / \sigma$:

$$\mathcal{L}_\text{DSM} = \mathbb{E}\left[\left\|s_\theta(x + \sigma\epsilon, \sigma) + \epsilon/\sigma\right\|^2\right]$$

또는 $\sigma$-multiplied:

$$\sigma^2 \mathcal{L}_\text{DSM} = \mathbb{E}\left[\left\|\sigma s_\theta + \epsilon\right\|^2\right]$$

NN 이 $\sigma s_\theta = -\epsilon$ 학습 — DDPM 과 같은 form.

### 유도 4 — DDPM ↔ NCSN Mapping

DDPM 의 noise level: $\sigma_t = \sqrt{1 - \bar\alpha_t}$ (effective noise std).

DDPM forward: $x_t = \sqrt{\bar\alpha_t} x_0 + \sigma_t \epsilon$ — scaled signal + noise.

**Modification**: $\hat x_0 = x_t / \sqrt{\bar\alpha_t}$ (rescale to unit signal). 그러면:

$$\hat x_0 = x_0 + \frac{\sigma_t}{\sqrt{\bar\alpha_t}} \epsilon = x_0 + \tilde\sigma_t \epsilon$$

NCSN 의 form ($\tilde\sigma_t = \sigma_t / \sqrt{\bar\alpha_t}$).

**Equivalence**: DDPM 의 schedule = scaled NCSN schedule. Score equivalence:

$$s_\text{NCSN}(\hat x_0, \tilde\sigma) = \sqrt{\bar\alpha_t} s_\text{DDPM}(x_t, t)$$

(둘 다 $-\epsilon / (\sigma \cdot \sqrt{\bar\alpha_t})$ form).

### 유도 5 — Annealed Langevin 의 Effectiveness

Pure Langevin from random init: data manifold 에 도달하려면 매우 많은 step (slow mixing).

**Multi-scale**:
1. Large $\sigma_1$: smoothed distribution $q_{\sigma_1}$, manifold 가 fat — score 가 globally informative
2. Decrease $\sigma$: 점진적 sharpening, score 가 local detail

각 scale 에서 충분한 mixing → final scale 에서 data manifold 도달.

DDPM 의 reverse process 와 algorithmically 유사 — same idea, different parameterization.

---

## 💻 실험 재현 / PyTorch 구현

### 실험 1 — 1D Score Matching

```python
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

# Target: bimodal mixture
def true_score(x):
    """True score of 0.5*N(-2, 1) + 0.5*N(2, 1)"""
    p1 = torch.exp(-0.5 * (x + 2)**2)
    p2 = torch.exp(-0.5 * (x - 2)**2)
    p = 0.5 * (p1 + p2)
    grad_p = 0.5 * (p1 * (-(x + 2)) + p2 * (-(x - 2)))
    return grad_p / (p + 1e-12)

# Score model
class ScoreNet(nn.Module):
    def __init__(self, hidden=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(1, hidden), nn.SiLU(),
            nn.Linear(hidden, hidden), nn.SiLU(),
            nn.Linear(hidden, 1),
        )
    def forward(self, x):
        return self.net(x.unsqueeze(-1)).squeeze(-1)

s_theta = ScoreNet()
opt = torch.optim.Adam(s_theta.parameters(), lr=1e-3)

# DSM training
sigma = 0.5
for step in range(3000):
    x_real = torch.where(torch.rand(256) > 0.5,
                         torch.randn(256) * 0.5 + 2,
                         torch.randn(256) * 0.5 - 2)
    eps = torch.randn(256)
    x_tilde = x_real + sigma * eps
    target = -eps / sigma
    pred = s_theta(x_tilde)
    loss = ((pred - target) ** 2).mean()
    opt.zero_grad(); loss.backward(); opt.step()

# Visualize learned score
x = torch.linspace(-5, 5, 100)
with torch.no_grad():
    s_learned = s_theta(x)
s_true = true_score(x)
plt.plot(x, s_true, label='True score')
plt.plot(x, s_learned, '--', label='Learned (DSM)')
plt.xlabel('x'); plt.ylabel('∇log p')
plt.legend(); plt.show()
```

### 실험 2 — Langevin Sampling

```python
def langevin_sample(score_fn, n=500, T=1000, delta=0.01):
    x = torch.randn(n)   # init from prior
    for t in range(T):
        z = torch.randn_like(x)
        x = x + (delta / 2) * score_fn(x) + np.sqrt(delta) * z
    return x

@torch.no_grad()
def s_fn(x):
    return s_theta(x)

samples = langevin_sample(s_fn).numpy()
plt.hist(samples, bins=50, density=True, alpha=0.5, label='Langevin samples')
real = torch.where(torch.rand(2000) > 0.5,
                   torch.randn(2000) * 0.5 + 2,
                   torch.randn(2000) * 0.5 - 2)
plt.hist(real.numpy(), bins=50, density=True, alpha=0.5, label='True dist')
plt.legend(); plt.show()
# Bimodal mixture 잘 sampled
```

### 실험 3 — Multi-Scale NCSN (Annealed)

```python
class NCSNNet(nn.Module):
    """Conditioned on σ"""
    def __init__(self, hidden=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2, hidden), nn.SiLU(),  # x and log_sigma
            nn.Linear(hidden, hidden), nn.SiLU(),
            nn.Linear(hidden, 1),
        )
    def forward(self, x, sigma):
        log_sigma = sigma.log()
        if log_sigma.dim() == 0:
            log_sigma = log_sigma.expand(x.size(0))
        return self.net(torch.stack([x, log_sigma], dim=-1)).squeeze(-1)

L = 10
sigmas = torch.linspace(0.01, 5.0, L).flip(0)   # decreasing

ncsn = NCSNNet()
opt = torch.optim.Adam(ncsn.parameters(), lr=1e-3)

for step in range(5000):
    x_real = torch.where(torch.rand(256) > 0.5,
                         torch.randn(256) * 0.5 + 2,
                         torch.randn(256) * 0.5 - 2)
    sigma_idx = torch.randint(0, L, (256,))
    sigma = sigmas[sigma_idx]
    eps = torch.randn(256)
    x_tilde = x_real + sigma * eps
    target = -eps / sigma
    pred = ncsn(x_tilde, sigma)
    loss = (sigma ** 2 * (pred - target) ** 2).mean()
    opt.zero_grad(); loss.backward(); opt.step()

# Annealed Langevin sampling
def annealed_langevin(ncsn, sigmas, T=100, eps=2e-5, n=500):
    x = torch.randn(n) * sigmas[0]
    for sigma in sigmas:
        delta = eps * (sigma / sigmas[-1]) ** 2
        for _ in range(T):
            z = torch.randn_like(x)
            with torch.no_grad():
                s = ncsn(x, sigma * torch.ones(n))
            x = x + (delta / 2) * s + np.sqrt(delta) * z
    return x

samples = annealed_langevin(ncsn, sigmas).numpy()
```

### 실험 4 — DDPM ↔ NCSN Equivalence Check

```python
# Same data, train DDPM 와 NCSN 비교
# Compare learned score (after appropriate scaling)
# DDPM ε prediction with rescaling = NCSN score prediction
# 두 sample distribution 이 비슷
```

---

## 🔗 이론과 실전의 간극

### 1. Score-Based 의 Theoretical Elegance

NCSN 의 framework 가 EBM (energy-based model) 와 자연스럽게 연결:
- $p(x) = e^{-E(x)} / Z$
- $\nabla \log p = -\nabla E$
- Score 학습 = energy 의 derivative 학습 — partition 회피

이것이 EBM 의 부흥 (Ch7-03) 의 한 형태.

### 2. NCSN 의 Sample Quality

NCSN 의 first generation: image generation 에서 reasonable 결과. 그러나 DDPM 이 약간 우월. 후속 NCSN++ (Song 2020), Score-SDE (Song 2021) 가 SOTA 달성.

### 3. Score-SDE 의 통합

Ch6-04 의 Score-SDE 가 DDPM, NCSN, 그리고 새 변형들을 하나의 SDE framework 으로 통합. 이로 모든 diffusion variant 의 mathematical unification.

---

## ⚖️ 가정과 한계

| 가정 | 한계 |
|------|------|
| Score 가 well-defined | $p(x) = 0$ 인 region 에서 부정확 |
| Langevin 이 mixing | High-dim 에서 slow, multi-scale 필요 |
| DSM 이 standard SM 과 동치 | 다른 $\sigma$ 에서 다른 minimum |
| Multi-scale 이 충분 | $\sigma$ schedule 의 선택이 결정적 |
| Continuous 도메인 | Discrete data 에는 dequantization |

---

## 📌 핵심 정리

$$\boxed{\text{Score: } s(x) = \nabla_x \log p(x) \quad - \text{ density 의 gradient}}$$

$$\boxed{\text{Langevin: } x_{t+1} = x_t + \frac{\delta}{2} s_\theta(x_t) + \sqrt{\delta} z_t \to p}$$

$$\boxed{\text{DSM: } \mathcal{L} = \mathbb{E}\left[\left\|s_\theta(\tilde x) + \frac{\tilde x - x}{\sigma^2}\right\|^2\right]}$$

| 관계 | DDPM | NCSN |
|------|------|------|
| Target | $\epsilon$ prediction | Score prediction |
| Loss | $L_\text{simple}$ | weighted DSM |
| Forward | $x_t = \sqrt{\bar\alpha_t} x_0 + \sqrt{1-\bar\alpha_t}\epsilon$ | $\tilde x = x + \sigma \epsilon$ |
| Sampling | Reverse process | Annealed Langevin |
| Equivalence | $s_\theta = -\epsilon_\theta / \sqrt{1-\bar\alpha_t}$ |  |

| Score Matching Variants | Pros | Cons |
|------------------------|------|------|
| **Standard SM** (Hyvärinen 2005) | Direct | $O(d^2)$ divergence |
| **Sliced SM** (Song 2019) | $O(d)$ via Hutchinson | Stochastic variance |
| **Denoising SM** (Vincent 2011) | Closed-form target | Different objective |
| **Multi-scale (NCSN)** | Captures all scales | Hyperparameter |

---

## 🤔 생각해볼 문제

**문제 1** (기초): 1D Gaussian $p(x) = \mathcal{N}(x; \mu, \sigma^2)$ 의 score 를 closed form 으로 계산하라.

<details>
<summary>해설</summary>

$$\log p(x) = -\frac{(x - \mu)^2}{2\sigma^2} - \frac{1}{2}\log(2\pi\sigma^2)$$

$$\nabla_x \log p = -\frac{x - \mu}{\sigma^2}$$

**해석**:
- $x = \mu$: score = 0 (peak, 이미 high density)
- $x > \mu$: score < 0 (push toward $\mu$)
- $x < \mu$: score > 0 (push toward $\mu$)

**Langevin step**: $x_{t+1} = x_t - (\delta/(2\sigma^2))(x_t - \mu) + \sqrt{\delta} z$. Linear AR process — converges to $\mathcal{N}(\mu, \sigma^2)$.

**시사점**: simple Gaussian 의 score 가 affine — Langevin 이 빠르게 convergence. 복잡한 분포에서는 score 가 nonlinear, mixing 더 어려움.

</details>

**문제 2** (심화): DDPM 의 $\epsilon$-prediction 이 score 학습과 동치임을 보였다. 그러면 $L_\text{simple}$ 의 weight $w_t = 1$ 이 NCSN 의 weight $\sigma^2$ 과 어떻게 연결되는가?

<details>
<summary>해설</summary>

DDPM 의 $\sigma_t = \sqrt{1 - \bar\alpha_t}$ (effective noise std).

$L_\text{simple}$: $\mathbb{E}[\|\epsilon - \epsilon_\theta\|^2]$ — weight $w_t^{L_\text{simple}} = 1$.

**Score equivalence**: $\epsilon = -\sigma_t \cdot s$, 따라서 $\epsilon - \epsilon_\theta = -\sigma_t (s - s_\theta)$.

$\|\epsilon - \epsilon_\theta\|^2 = \sigma_t^2 \|s - s_\theta\|^2$.

**Score-form $L_\text{simple}$**:

$$L_\text{simple} = \mathbb{E}[\sigma_t^2 \|s_\theta - \nabla \log q_t\|^2]$$

**비교 with NCSN weight**: $\mathcal{L}_\text{NCSN}^{(t)} = \sigma_t^2 \cdot \mathbb{E}[\|s_\theta - \nabla \log q_{\sigma_t}\|^2]$.

→ **Same**! DDPM $L_\text{simple}$ = NCSN 의 standard weighted DSM.

**시사점**:
- DDPM 과 NCSN 이 정확히 same loss (different parameterization)
- $w_t = 1$ in $\epsilon$-space ⟺ $w_t = \sigma_t^2$ in score-space — natural choice
- "Simple" weighting 이 사실은 "$\sigma^2$-weighted" — score-based 관점에서 표준

이는 DDPM 과 NCSN 의 mathematical equivalence 의 명확한 demonstration.

</details>

**문제 3** (논문 비평): Score-based (NCSN) vs Likelihood-based (Flow, AR) 의 비교 — 두 패러다임의 장단점과, diffusion 이 어떻게 두 framework 의 advantages 를 결합하는지 논하라.

<details>
<summary>해설</summary>

**Likelihood-based (AR, Flow)**:
- ✅ Tractable likelihood — NLL evaluation
- ✅ MLE training — stable
- ❌ Architectural restriction (causal mask, invertibility)
- ❌ Sample quality 가 GAN 보다 못 함

**Score-based (NCSN, EBM)**:
- ✅ No architectural restriction — free-form NN
- ✅ EBM 의 expressiveness
- ❌ Partition function $Z$ intractable
- ❌ Likelihood evaluation 직접 안 됨
- ✅ Score matching 으로 $Z$ 회피

**Diffusion 의 결합**:

1. **Likelihood**: ELBO (lower bound, like VAE) + Score-SDE 의 exact likelihood via Probability Flow ODE
2. **Sample quality**: Score-based training + iterative refinement → SOTA FID
3. **No architectural restriction**: U-Net 이 free-form
4. **Stable training**: per-timestep supervised loss (like AR)
5. **No partition function**: score 직접 학습

**Trade-off**:
- 느린 sampling (50+ steps)
- Computational cost 높음

**Summary table**:

| 측면 | AR | Flow | VAE | GAN | NCSN | Diffusion |
|------|-----|------|-----|-----|------|-----------|
| Likelihood | Exact | Exact | Lower | None | Implicit | Lower (ELBO) + Exact (PF-ODE) |
| Sample quality | Good | Medium | Blurry | Sharp | Good | SOTA |
| Training stability | Stable | Stable | Stable | Unstable | Stable | Stable |
| Architecture | AR mask | Coupling | Encoder/decoder | Free | Free | Free (U-Net) |

**시사점**: Diffusion 이 모든 family 의 advantages 결합:
- AR 의 stable training
- Flow 의 exact likelihood (via PF-ODE)
- Score-based 의 free architecture
- GAN-level sample quality

**Trade-off**: sampling speed. 이것이 Consistency Model, Rectified Flow 등 후속 작업의 motivation.

**현대적 위치**: 2020 이후 image generation 의 mainstream. Future direction: 1-step generation (Consistency), better likelihood (VDM), multimodal (DALL-E 3, Sora).

</details>

---

<div align="center">

[◀ 이전 (02. L_simple)](./02-ddpm-simple-loss.md) | [📚 README](../README.md) | [다음 ▶ (04. Score-SDE)](./04-score-sde.md)

</div>
