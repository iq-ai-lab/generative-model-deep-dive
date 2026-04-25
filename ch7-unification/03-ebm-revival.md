# 03. Generative Model 과 Energy-Based Model

## 🎯 핵심 질문

- Energy-Based Model $p_\theta(x) = \frac{1}{Z(\theta)} \exp(-E_\theta(x))$ 의 의미와 partition function $Z$ 의 intractability 문제?
- Contrastive Divergence (Hinton 2002) 의 mechanism — MCMC negative samples 와 positive 의 contrast?
- Score Matching (Hyvärinen 2005) 가 어떻게 $\nabla_x \log Z = 0$ 을 활용해 partition function 을 회피하는가?
- JEM (Grathwohl 2020): "Your classifier is secretly an EBM" 의 통찰 — softmax classifier 가 EBM 임을 보임?
- Score-based diffusion 이 EBM 의 derivative 를 학습하여 $Z$ 를 우회하는 방식?

---

## 🔍 왜 EBM 의 부흥이 흥미로운가

Energy-Based Model 은 generative model 의 가장 일반적 form:

$$p_\theta(x) \propto \exp(-E_\theta(x))$$

**Maximum flexibility**: $E_\theta$ 가 임의 NN — architectural restriction 없음. 그러나 **partition function $Z$ 의 intractability** 가 historically 큰 obstacle.

2020년대 부흥의 이유:
1. **Score-based bridge**: Diffusion 이 $\nabla \log p$ 학습 — EBM 의 derivative
2. **JEM** (Grathwohl 2020): classifier 의 logit 이 EBM 으로 해석
3. **Compositionality**: 다른 EBM 결합으로 새 generative model
4. **Hybrid**: EBM + flow, EBM + VAE

이 문서에서는 EBM 의 수학, 훈련 방법들 (CD, score matching), JEM 의 통찰, 그리고 diffusion 과의 연결을 다룹니다.

---

## 📐 수학적 선행 조건

- Ch6-03 (Score-based)
- [Probability Theory Deep Dive](https://github.com/iq-ai-lab/probability-theory-deep-dive): Distribution, MCMC
- [Information Theory Deep Dive](https://github.com/iq-ai-lab/information-theory-deep-dive): KL divergence

---

## 📖 직관적 이해

### "Energy = Negative Log-Density"

$E_\theta(x)$ 가 작으면 $p(x)$ 큼 (high density region). 큰 $E$ = 데이터에서 멀음.

**Boltzmann distribution** (statistical physics): $p(x) \propto e^{-E(x)/T}$. T = temperature.

ML 의 EBM: $T = 1$ — $p(x) \propto e^{-E(x)}$.

### Partition Function 의 함정

$$Z(\theta) = \int e^{-E_\theta(x)} dx$$

문제: high-dim integral, 일반 closed-form 없음.

**Direct MLE**:

$$\nabla_\theta \log p_\theta(x) = -\nabla_\theta E_\theta(x) - \nabla_\theta \log Z(\theta)$$

$\nabla_\theta \log Z = -\mathbb{E}_{p_\theta}[\nabla E_\theta]$ (computation via Monte Carlo).

**Issue**: $p_\theta$ 에서 sampling 이 어려움 (MCMC), expensive per gradient step.

### Contrastive Divergence (CD)

Idea: "$p_\theta$ 에서 sampling" 을 **MCMC** 로 짧게 (몇 step Langevin 또는 HMC):

$$\nabla_\theta \log p(x) \approx -\nabla E(x_\text{data}) + \nabla E(x_\text{MCMC})$$

**Contrast**: data sample 의 energy 낮춤, MCMC sample 의 energy 올림.

**Hinton 2002 의 trick**: MCMC chain 이 짧아도 (k=1 step) 작동 — 정확한 sampling 불필요.

### Score Matching: Partition Function 우회

$\nabla_x \log p = -\nabla_x E - \underbrace{\nabla_x \log Z}_{= 0}$ ($Z$ 가 $x$ 와 무관)

= $-\nabla_x E$ — **Z 와 무관**.

따라서 score 학습 = $-E_\theta$ 의 gradient 학습. NN 이 $E$ 자체가 아닌 $\nabla E$ 직접 학습 → score-based generative model.

**Diffusion 이 사실 EBM 의 score 학습**.

---

## ✏️ 엄밀한 정의·정리

### 정의 3.1 — Energy-Based Model

$$p_\theta(x) = \frac{e^{-E_\theta(x)}}{Z(\theta)}, \quad Z(\theta) = \int e^{-E_\theta(x)} dx$$

$E_\theta: \mathbb{R}^d \to \mathbb{R}$ — "energy function", NN.

### 정리 3.2 — MLE Gradient

$$\nabla_\theta \log p_\theta(x) = -\nabla_\theta E_\theta(x) + \mathbb{E}_{p_\theta}[\nabla_\theta E_\theta(x')]$$

**증명**:

$\log p_\theta(x) = -E_\theta(x) - \log Z(\theta)$

$\nabla_\theta \log Z = (1/Z) \nabla_\theta Z = (1/Z) \int e^{-E} \cdot (-\nabla E) dx = -\mathbb{E}_{p_\theta}[\nabla E]$

따라서:

$\nabla_\theta \log p = -\nabla E(x) - (-\mathbb{E}_{p_\theta}[\nabla E]) = -\nabla E(x) + \mathbb{E}_{p_\theta}[\nabla E]$

**Interpretation**:
- "Positive phase" $-\nabla E(x)$: data 의 energy 감소
- "Negative phase" $\mathbb{E}_{p_\theta}[\nabla E]$: model 의 high-density region 의 energy 증가

### 정의 3.3 — Contrastive Divergence (CD-k, Hinton 2002)

Approximate negative phase via $k$-step MCMC starting from data:

$$\nabla_\theta \mathcal{L}_\text{CD-k} \approx -\nabla E(x_\text{data}) + \nabla E(x_k)$$

$x_k$: $k$ steps of MCMC (Langevin, HMC) from $x_\text{data}$.

$k = 1$: Hinton 의 famous CD-1 (works empirically).

### 정의 3.4 — Score Matching (Hyvärinen 2005)

$$\mathcal{L}_\text{SM}(\theta) = \mathbb{E}_{p_d}\left[\frac{1}{2}\|s_\theta(x)\|^2 + \nabla \cdot s_\theta(x)\right] + \text{const}$$

여기서 $s_\theta(x) = -\nabla_x E_\theta(x)$.

**No partition function**: $Z$ 가 등장 안 함 (Ch6-03 derivation).

### 정의 3.5 — Joint Energy-Based Model (JEM, Grathwohl 2020)

Classifier $p_\theta(y | x) = \exp(f_\theta(x)[y]) / \sum_{y'} \exp(f_\theta(x)[y'])$ — softmax over logits.

**Reinterpretation**:

$$p_\theta(x, y) = \exp(f_\theta(x)[y]) / Z(\theta)$$

$$p_\theta(x) = \sum_y p_\theta(x, y) = \frac{\sum_y \exp(f_\theta(x)[y])}{Z(\theta)} = \frac{\text{LogSumExp}_\theta(x)}{Z(\theta)}$$

**Energy**:

$$E_\theta(x) = -\text{LogSumExp}_y(f_\theta(x)[y]) = -\log \sum_y \exp(f_\theta(x)[y])$$

따라서 **classifier 의 logit 이 implicit EBM 정의**. JEM 이 이를 명시화하여 generative + discriminative joint training.

### 정리 3.6 — Score-Based Generative = Implicit EBM

NCSN, DDPM 등 score-based 가:

$$s_\theta(x, t) \approx \nabla_x \log p_t(x) = -\nabla_x E_t(x)$$

$E_t(x)$ 는 implicit (NN 이 derivative 만 학습). $E$ 자체는 reconstruction 가능 (line integral):

$$E_\theta(x) = -\int_0^1 s_\theta(\alpha x) \cdot x \, d\alpha + \text{const}$$

따라서 **diffusion model 은 sequence 의 EBM** (각 noise level $t$).

---

## 🔬 증명 및 수학적 유도

### 유도 1 — CD-1 의 Approximation Quality

$k = 1$ Langevin step:

$$x_1 = x_0 + \frac{\delta}{2} \nabla \log p_\theta(x_0) + \sqrt\delta z = x_0 - \frac{\delta}{2} \nabla E(x_0) + \sqrt\delta z$$

(small step from data).

**Negative phase**: $\nabla E(x_1) \approx \nabla E(x_0) + \nabla^2 E(x_0) \cdot (x_1 - x_0)$.

$\mathbb{E}[\nabla E(x_1) - \nabla E(x_0)] \approx -\frac{\delta}{2} \nabla^2 E \cdot \nabla E$ (drift term, noise mean 0).

이게 $\nabla_\theta D_\text{KL}(p_d \| p_\theta)$ 와 비례 (Hinton 2002 의 increment) — partial gradient.

**효과**: CD-1 이 정확한 MLE 가 아니지만 **valid descent direction** for KL — empirically 잘 작동.

### 유도 2 — Score Matching 의 Z-Free Form

$\mathcal{L}_\text{SM} = \mathbb{E}_{p_d}[\|s_\theta - \nabla \log p_d\|^2]$ (true).

Integration by parts (Ch6-03):

$$\mathcal{L}_\text{SM} = \mathbb{E}_{p_d}[\|s_\theta\|^2 + 2 \nabla \cdot s_\theta] + \text{const}$$

**$s_\theta = -\nabla E_\theta$**:

$$\mathcal{L}_\text{SM} = \mathbb{E}_{p_d}[\|\nabla E_\theta\|^2 - 2 \nabla^2 E_\theta] + \text{const}$$

(divergence of $-\nabla E$ = $-\nabla^2 E$).

**No Z**: $E_\theta$ 만 있고 $Z$ 없음 — partition function 자동 회피.

**Cost**: $\nabla^2 E$ (Laplacian) 계산이 high-dim 에서 expensive ($O(d^2)$ 또는 Hutchinson $O(d)$).

### 유도 3 — JEM 의 Joint Training

**Cross-Entropy** (classification): $\mathcal{L}_\text{CE} = -\log p_\theta(y | x)$.

**EBM Loss** (generation): $\mathcal{L}_\text{EBM} = -\log p_\theta(x)$.

**Joint**: $\mathcal{L}_\text{JEM} = \mathcal{L}_\text{CE} + \lambda \mathcal{L}_\text{EBM}$.

**$p_\theta(y|x)$**: standard cross-entropy, easy.

**$p_\theta(x) = \text{LogSumExp}(f_\theta(x)) / Z$**: EBM, $Z$ intractable. Use CD or score matching.

**Empirical** (Grathwohl 2020):
- CIFAR-10 classification accuracy: 92.9% (vs 95% standard)
- FID: 38.4 (모자라지만 generation 가능)
- **Adversarial robustness 우월**: standard CIFAR 의 $\epsilon = 8/255$ FGSM 에 robust

**시사점**: classifier 가 EBM 임을 인식하고 joint training 하면 **dual benefits** — classification accuracy slight loss + generation ability + robustness.

### 유도 4 — EBM Compositionality

EBM 의 elegance: $p(x) \propto \exp(-E(x))$ → $E$ 들의 합 = product of distributions:

$$E_\text{combined}(x) = E_1(x) + E_2(x) \Rightarrow p_\text{combined}(x) \propto p_1(x) \cdot p_2(x)$$

**예**:
- $p_1$: "이미지 분포" learned
- $p_2$: "특정 attribute 의 분포" learned
- Combined: "특정 attribute 가진 이미지 분포"

**Du 2020** 의 "Compositional Generation with Energy-Based Models": multiple constraints 의 product 으로 controllable generation.

**Diffusion 에서의 형태**: classifier-free guidance 가 사실 EBM compositionality 의 implicit form — $p(x|y) \propto p(x) p(y|x)^w$ 가 product of distributions.

### 유도 5 — Diffusion ↔ EBM 의 정확한 관계

Diffusion 의 forward: $q_t(x) = \int q(x | x_0) p_d(x_0) dx_0$ — 각 $t$ 에서 noisy distribution.

**Score**: $\nabla_x \log q_t = -\nabla_x E_t$ for some implicit $E_t$.

NN 이 $s_\theta(x, t) \approx \nabla_x \log q_t = -\nabla_x E_t$ 학습.

따라서 diffusion model = **time-conditional EBM 의 score**.

**Generation via Langevin**: $x \leftarrow x + (\delta/2) s_\theta + \sqrt\delta z$ — EBM 에서 sampling 의 standard 방법.

**Reverse SDE = annealed Langevin**: $t = T \to 0$ 의 progression.

**시사점**: diffusion 은 사실 multi-scale EBM — 각 noise level 의 EBM 을 sequence 로 해결. Score 학습이 efficient algorithm.

---

## 💻 실험 재현 / PyTorch 구현

### 실험 1 — 1D EBM with CD

```python
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

# Target: bimodal
def sample_target(n):
    return torch.where(torch.rand(n) > 0.5,
                       torch.randn(n) * 0.5 + 2,
                       torch.randn(n) * 0.5 - 2)

# Energy NN
class EnergyNet(nn.Module):
    def __init__(self, hidden=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(1, hidden), nn.SiLU(),
            nn.Linear(hidden, hidden), nn.SiLU(),
            nn.Linear(hidden, 1),
        )
    def forward(self, x):
        return self.net(x.unsqueeze(-1)).squeeze(-1)

E_theta = EnergyNet()
opt = torch.optim.Adam(E_theta.parameters(), lr=1e-3)

# Langevin sampling for negative samples
def langevin_step(x, E, delta=0.05, n_steps=10):
    x = x.clone().detach().requires_grad_(True)
    for _ in range(n_steps):
        e = E(x)
        grad = torch.autograd.grad(e.sum(), x)[0]
        with torch.no_grad():
            x = x - delta * grad + np.sqrt(2 * delta) * torch.randn_like(x)
        x = x.detach().requires_grad_(True)
    return x.detach()

# CD-k training
for step in range(3000):
    x_data = sample_target(128)
    # Initialize MCMC from data (CD)
    x_init = x_data + torch.randn_like(x_data) * 0.1
    x_neg = langevin_step(x_init, E_theta, n_steps=10)

    e_data = E_theta(x_data).mean()
    e_neg = E_theta(x_neg).mean()
    loss = e_data - e_neg   # data energy down, negative energy up
    # + L2 regularization on energy
    loss = loss + 0.01 * (e_data**2 + e_neg**2)
    opt.zero_grad(); loss.backward(); opt.step()

# Sampling: Langevin from random init
@torch.no_grad()
def sample(E, n=500, n_steps=1000):
    x = torch.randn(n)
    for _ in range(n_steps):
        x = x.requires_grad_(True)
        e = E(x).sum()
        grad = torch.autograd.grad(e, x)[0]
        with torch.no_grad():
            x = x - 0.05 * grad + np.sqrt(0.1) * torch.randn_like(x)
    return x

samples = sample(E_theta).numpy()
plt.hist(samples, bins=50, density=True, alpha=0.5, label='EBM samples')
real = sample_target(2000).numpy()
plt.hist(real, bins=50, density=True, alpha=0.5, label='Real')
plt.legend(); plt.show()
```

### 실험 2 — Score Matching for EBM

```python
def score_matching_loss(E, x):
    """Score matching: ||∇E||^2 + 2 * ∇²E (Hutchinson)"""
    x = x.clone().detach().requires_grad_(True)
    e = E(x).sum()
    grad = torch.autograd.grad(e, x, create_graph=True)[0]
    # |grad|^2 / 2 (per sample)
    norm_sq = grad.pow(2).sum(-1) / 2
    # Hutchinson trace estimator for ∇²E (Laplacian)
    eps = torch.randn_like(x)
    grad_eps = (grad * eps).sum()
    hess_eps = torch.autograd.grad(grad_eps, x)[0]
    laplacian = (hess_eps * eps).sum(-1)
    return (norm_sq + laplacian).mean()

# Train E with score matching
for step in range(2000):
    x = sample_target(128)
    loss = score_matching_loss(E_theta, x)
    opt.zero_grad(); loss.backward(); opt.step()

# Pros: no MCMC during training, faster
# Cons: divergence calculation (Laplacian)
```

### 실험 3 — JEM (Joint Energy-Based Model)

```python
class JEM(nn.Module):
    def __init__(self, n_classes=10, in_ch=1, hidden=64):
        super().__init__()
        # Standard classifier (like ResNet)
        self.backbone = nn.Sequential(
            nn.Conv2d(in_ch, hidden, 3, padding=1), nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(hidden, hidden*2, 3, padding=1), nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
        )
        self.fc = nn.Linear(hidden*2, n_classes)

    def logits(self, x):
        h = self.backbone(x)
        return self.fc(h)   # [B, K] — logits

    def energy(self, x):
        return -torch.logsumexp(self.logits(x), dim=-1)   # E(x) = -LogSumExp

    def class_prob(self, x):
        return torch.softmax(self.logits(x), dim=-1)

# Training: cross-entropy + EBM (CD)
def jem_loss(model, x, y, alpha=1.0):
    # Discriminative
    logits = model.logits(x)
    loss_clf = nn.functional.cross_entropy(logits, y)

    # Generative (CD)
    e_pos = model.energy(x).mean()
    x_neg = langevin_sample_image(model, ...)   # Langevin from random
    e_neg = model.energy(x_neg).mean()
    loss_gen = e_pos - e_neg

    return loss_clf + alpha * loss_gen

# JEM: generation + classification + robustness
```

### 실험 4 — Compositionality

```python
# 두 EBM 결합으로 conditional generation
class EBM1(nn.Module):
    """Image distribution learned"""
    pass
class EBM2(nn.Module):
    """Attribute (e.g., 'bright') distribution"""
    pass

def composed_energy(x, ebm1, ebm2, alpha=1.0):
    return ebm1(x) + alpha * ebm2(x)

# Sample from composition: Langevin with composed gradient
# x ← x - δ ∇[E1 + α E2] + √2δ z
# Result: image distribution AND attribute simultaneously
```

---

## 🔗 이론과 실전의 간극

### 1. EBM 의 잔존 한계

EBM 이 elegant 하지만 mainstream 으로 안 된 이유:
- **Sampling 비용**: Langevin/HMC 가 high-dim 에서 slow
- **Training instability**: positive/negative phase balance 어려움
- **Image quality**: pure EBM 의 sample 이 GAN/Diffusion 보다 떨어짐

해결책으로 hybrid (EBM + Flow, EBM + VAE) 등 시도, 큰 breakthrough 없음.

### 2. Diffusion 이 EBM 의 De Facto Implementation

Diffusion model 이 사실 EBM 의 score 를 학습 — 다음의 advantages:
- Multi-scale EBM (annealed)
- Score-based (no $Z$)
- Stable training

**EBM 의 부흥** 은 사실 diffusion 의 부흥 — 같은 mathematical framework, 다른 algorithmic approach.

### 3. JEM 의 Industrial Use

JEM 의 robustness advantage 가 specific applications 에 가치:
- Adversarial defense
- OOD detection
- Calibrated uncertainty
- Active learning

순수 generation 에서는 less competitive.

---

## ⚖️ 가정과 한계

| 가정 | 한계 |
|------|------|
| EBM 의 expressiveness 보편 | Partition $Z$ 의 intractability fundamental |
| CD 가 sufficient training | Bias 있음 (정확한 MLE 아님) |
| Score matching 이 efficient | Laplacian 계산 high-dim 에서 비쌈 |
| JEM 이 두 task 모두 잘 함 | Classifier accuracy 약간 손해 |
| Compositionality 가 자연 | Multiple EBMs 의 product 가 단순 sum |

---

## 📌 핵심 정리

$$\boxed{p_\theta(x) = \frac{e^{-E_\theta(x)}}{Z(\theta)}}$$

$$\boxed{\nabla_\theta \log p_\theta(x) = -\nabla_\theta E_\theta(x) + \mathbb{E}_{p_\theta}[\nabla_\theta E_\theta]}$$

| Method | Mechanism | Complexity |
|--------|-----------|------------|
| **CD-k** (Hinton 2002) | $k$-step MCMC for negative | Slow |
| **Score Matching** (Hyvärinen 2005) | $Z$ 회피, divergence based | Laplacian cost |
| **Diffusion / NCSN** | Multi-scale score | Practical, scalable |
| **JEM** (Grathwohl 2020) | Classifier as EBM | Joint training |

| EBM 의 Advantages | Limitations |
|------------------|-------------|
| Maximum flexibility ($E$ free-form) | $Z$ intractable |
| Compositionality (sum of $E$) | Sampling slow (MCMC) |
| Adversarial robustness (JEM) | Image quality 부족 (pure EBM) |
| Classifier 와 unified | Training unstable |

---

## 🤔 생각해볼 문제

**문제 1** (기초): EBM 의 MLE gradient 의 두 항 ("positive" $-\nabla E(x)$ 와 "negative" $\mathbb{E}_{p_\theta}[\nabla E]$) 의 직관적 의미를 설명하라.

<details>
<summary>해설</summary>

**Positive phase** $-\nabla E(x)$ for $x \sim p_d$:
- Direction: data point 의 energy 줄이는 방향
- 효과: data 의 likelihood 증가
- Computation: trivial (single forward + backward)

**Negative phase** $\mathbb{E}_{p_\theta}[\nabla E]$:
- Average over model's current distribution
- "Model 이 high density 라고 생각하는 곳" 의 energy 올리기
- 효과: model 의 잘못된 mode 제거
- Computation: $p_\theta$ 에서 sampling 필요 (MCMC)

**비유**: "data 점들을 끌어내리고, 모델이 잘못 만든 가짜 점들을 끌어올린다".

**Equilibrium**: positive 와 negative 가 같으면 ($\mathbb{E}_{p_d}[\nabla E] = \mathbb{E}_{p_\theta}[\nabla E]$) → MLE optimum, $p_\theta = p_d$.

**시사점**: EBM 의 fundamental difficulty 는 negative phase. Sampling 이 어려운 이유 = $p_\theta$ 가 expressive 일수록 MCMC 가 mixing 안 됨.

</details>

**문제 2** (심화): JEM (Grathwohl 2020) 의 $E_\theta(x) = -\text{LogSumExp}_y(f_\theta(x)[y])$ 가 왜 standard classifier 의 logit 으로부터 자연스럽게 유도되는지 설명하라.

<details>
<summary>해설</summary>

**Standard Classifier**:
$p(y | x) = e^{f(x)[y]} / Z_y(x)$, $Z_y(x) = \sum_{y'} e^{f(x)[y']}$.

**Joint Distribution Form**:

만약 unnormalized $p(x, y) = e^{f(x)[y]}$ 로 정의하면:

$Z(\theta) = \sum_y \int e^{f(x)[y]} dx$ (joint partition).

$p(y | x) = p(x, y) / p(x) = \frac{e^{f(x)[y]}}{\sum_{y'} e^{f(x)[y']}} = $ standard softmax classifier ✓.

**Marginal $p(x)$**:
$p(x) = \sum_y p(x, y) / Z = \frac{\sum_y e^{f(x)[y]}}{Z} = \frac{Z_y(x)}{Z}$

여기서 $Z_y(x) = \text{LogSumExp}_y(f(x)[y])$ — input-dependent.

**Energy form**:
$p(x) \propto Z_y(x) = e^{\text{LogSumExp}(f(x)[y])}$

→ $E_\theta(x) = -\text{LogSumExp}_y(f_\theta(x)[y])$.

**해석**:
- LogSumExp = soft-max with smooth maximum
- 어떤 class 에든 strongly classified 되면 LogSumExp 큼 (low energy, high $p(x)$)
- 모든 class 에 weak classification (uniform) 이면 LogSumExp 작음 (high energy, low $p(x)$)

**시사점**:
- Standard classifier 가 implicit EBM 정의
- "Confident classification" 이 "high data density" 와 align
- Adversarial example 이 종종 low confidence → low $p(x)$ → OOD-like 동작

**JEM 의 Joint Training**:
- Cross-entropy: $p(y|x)$ 학습 (discriminative)
- EBM loss: $p(x)$ 학습 (generative)
- 둘 다 같은 $f_\theta$ 사용 — parameter sharing

**Empirical 결과**:
- Generation 가능 (Sample MNIST, CIFAR digits)
- Classification accuracy 약간 손해 (95% → 92%)
- Adversarial robustness 향상 (FGSM 에 대해 더 robust)

</details>

**문제 3** (논문 비평): EBM 의 부흥이 사실 Diffusion 의 부흥의 다른 form 인지 — 두 framework 의 mathematical equivalence 와 algorithmic difference 를 분석하라.

<details>
<summary>해설</summary>

**Mathematical Equivalence**:

**EBM**: $p(x) \propto e^{-E(x)}$, score = $-\nabla E$.

**Diffusion**: 각 noise level $t$ 에서 implicit EBM:
- $q_t(x) \propto e^{-E_t(x)}$ (일종의 EBM with $t$-conditional energy)
- NN $s_\theta(x, t) = -\nabla E_t(x)$ 학습

**Common framework**: score-based generative modeling. EBM 과 diffusion 이 same mathematical objects:
- Distribution: $\propto e^{-E}$
- Score: $\nabla \log p = -\nabla E$
- Sampling: Langevin or annealed Langevin

**Algorithmic Differences**:

**1. Training**:
- EBM: CD or score matching (single distribution)
- Diffusion: weighted DSM across noise levels (multiple distributions)

**2. Sampling**:
- EBM: Langevin from random init (slow, mixing 어려움)
- Diffusion: annealed Langevin / reverse SDE (fast, mode-coverage 보장)

**3. NN Architecture**:
- EBM: $E_\theta(x)$ scalar output
- Diffusion: $s_\theta(x, t)$ vector output (same dim as $x$), $t$-conditional

**4. Stability**:
- EBM: positive-negative phase balance 어려움
- Diffusion: per-timestep supervised (stable)

**Why Diffusion 이 Mainstream 이 됨**:

1. **Sampling efficiency**: annealed schedule 이 mode-coverage 보장
2. **Stability**: per-step training, no MCMC during training
3. **Quality**: SOTA image generation
4. **Theoretical**: ELBO bound, exact likelihood (PF-ODE)

**EBM 의 Unique Contribution**:
- **Compositionality**: $E_1 + E_2$ for product distribution
- **Classifier integration** (JEM)
- **Adversarial robustness**
- **Theoretical clarity**: explicit energy

**Hybrid 의 Possibility**:
- EBM-based diffusion: explicit energy + diffusion training
- Compositional diffusion: multiple diffusion models combined
- Diffusion-based JEM: adversarial robust diffusion classifier

**시사점**:
- Diffusion = practical instantiation of EBM principles
- EBM 의 부흥은 conceptual — generative modeling 의 unified view
- Future: 두 패러다임의 cross-pollination 으로 새 advances

**Modern view**: diffusion, score-based, EBM, NCSN 모두 같은 수학적 framework 의 different views. "Energy-based generative modeling" 이라는 broader category 안에 통합.

</details>

---

<div align="center">

[◀ 이전 (02. Consistency / Rectified Flow)](./02-consistency-rectified-flow.md) | [📚 README](../README.md) | [다음 ▶ (04. Frontier)](./04-frontier.md)

</div>
