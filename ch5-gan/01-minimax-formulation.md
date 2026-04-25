# 01. GAN 의 수학적 정식화 (Goodfellow 2014)

## 🎯 핵심 질문

- $\min_G \max_D V(D, G) = \mathbb{E}_{p_d}[\log D(x)] + \mathbb{E}_{p_z}[\log(1-D(G(z)))]$ 의 의미와 game-theoretic 해석은?
- "Two-player zero-sum game" 의 Nash equilibrium 이 GAN training 의 목적인 이유는?
- Generator 의 saturating loss $\log(1-D(G(z)))$ 가 초기 훈련에서 saturate 하는 문제 — non-saturating $-\log D(G(z))$ 의 정당성?
- Generator 와 discriminator 의 alternating update 가 왜 simultaneous 보다 일반적인가?
- GAN 이 implicit (no $p_\theta(x)$ closed form) 한 이유는? Manifold push-forward 의 의미?

---

## 🔍 왜 GAN 이 결정적인가

Goodfellow 2014 "Generative Adversarial Nets" 는 generative modeling 의 패러다임을 바꿨습니다:

1. **Adversarial training** — likelihood 없이 sampling 만으로 학습
2. **Implicit density** — $p_\theta(x)$ closed-form 불필요
3. **High-quality samples** — VAE 보다 sharp, 최초 photorealistic GAN samples
4. **Game-theoretic foundation** — minimax + Nash equilibrium 의 NN 적용

이 단순한 아이디어 — generator 와 discriminator 의 adversarial 게임 — 이 모든 GAN family (DCGAN, WGAN, StyleGAN, BigGAN) 의 출발. 이론적으로는 minimax 의 해가 $p_g = p_d$ (정확한 분포 학습), 실전에서는 mode collapse, training instability 등의 문제.

이 문서에서는 GAN 의 **수학적 formulation**, **Nash equilibrium**, **non-saturating loss 의 정당성** 을 다룹니다.

---

## 📐 수학적 선행 조건

- [Optimization Theory Deep Dive](https://github.com/iq-ai-lab/optimization-theory-deep-dive): Minimax, saddle point, Nash equilibrium
- [Probability Theory Deep Dive](https://github.com/iq-ai-lab/probability-theory-deep-dive): Push-forward measure
- 이전 챕터들: Ch1 (explicit/implicit, KL)

---

## 📖 직관적 이해

### "위조지폐범과 경찰의 게임"

**Generator $G$**: 가짜 돈을 만드는 위조지폐범. Latent noise $z$ 에서 가짜 데이터 $G(z)$ 생성.

**Discriminator $D$**: 가짜 vs 진짜를 구별하는 경찰. $D(x) \in [0, 1]$ 가 "진짜일 확률".

**Adversarial training**:
- $D$ 는 진짜와 가짜를 정확히 구별하려 함 — $D(x_\text{real}) = 1$, $D(G(z)) = 0$
- $G$ 는 $D$ 를 속이려 함 — $D(G(z)) = 1$ 만들기

**Equilibrium**: $D$ 가 더 이상 구별 못 하는 상태 — $D(x) = 0.5$ for all $x$. 이는 $p_g = p_d$ 일 때만 가능.

### 수학적 두 단계

**Step 1**: Discriminator $D$ 최적화 (fixed $G$):

$$D^* = \arg\max_D V(D, G) = \arg\max_D \mathbb{E}_{p_d}[\log D] + \mathbb{E}_{p_g}[\log(1-D)]$$

진짜에서 $D = 1$, 가짜에서 $D = 0$ 가 최적 — discriminator 의 cross-entropy.

**Step 2**: Generator $G$ 최적화 (fixed $D$):

$$G^* = \arg\min_G V(D, G) = \arg\min_G \mathbb{E}_{p_g}[\log(1 - D(x))]$$

가짜가 $D = 1$ 받게 하는 generator. (다음 챕터: 이를 JSD 로 환원)

### Saturating Loss 의 문제

Generator 의 loss $\log(1 - D(G(z)))$ 가 초기 훈련에서 문제:

- 훈련 초기: $G$ 가 약함 → $D$ 가 쉽게 구별 → $D(G(z)) \approx 0$
- $\log(1 - D) = \log(1 - 0) = 0$ — gradient 거의 0
- $G$ 학습 못 함

**해결**: non-saturating loss

$$L_G = -\mathbb{E}_{p_z}[\log D(G(z))]$$

$D(G(z)) \approx 0$ 에서도 강한 gradient — 빠른 초기 학습.

---

## ✏️ 엄밀한 정의·정리

### 정의 1.1 — GAN Architecture

**Generator** $G_\theta: \mathbb{R}^k \to \mathbb{R}^d$ (NN), latent $z \sim p_z$ (e.g., $\mathcal{N}(0, I_k)$).

**Discriminator** $D_\phi: \mathbb{R}^d \to [0, 1]$ (NN), output 이 "real probability".

### 정의 1.2 — Minimax Objective (Goodfellow 2014)

$$\min_G \max_D V(D, G) = \mathbb{E}_{x \sim p_d}[\log D(x)] + \mathbb{E}_{z \sim p_z}[\log(1 - D(G(z)))]$$

**해석**:
- $D$ 의 관점: classification (real vs fake) 의 negative cross-entropy 최대화
- $G$ 의 관점: $D$ 가 실수하게 하기 (push $D(G(z)) \to 1$)

### 정리 1.3 — Optimal Discriminator

Fixed $G$ (따라서 fixed $p_g$) 에 대해:

$$D^*(x) = \frac{p_d(x)}{p_d(x) + p_g(x)}$$

**증명**:

$V(D, G) = \int p_d(x) \log D(x) + p_g(x) \log(1 - D(x)) dx$

각 $x$ 에서 integrand 를 $D(x)$ 에 대해 최적화:

$\frac{\partial}{\partial D(x)} [a \log D + b \log(1 - D)] = a/D - b/(1 - D) = 0$

$D = a/(a + b) = p_d / (p_d + p_g)$. $\square$

### 정리 1.4 — Generator 의 Optimum

$D = D^*$ 대입한 후 $G$ 의 optimum:

$$G^* = \arg\min_G V(D^*, G)$$

(다음 챕터에서 $V(D^*, G) = 2 \text{JSD}(p_d \| p_g) - \log 4$ 환원, 따라서 $G^* \Leftrightarrow p_g = p_d$)

### 정의 1.5 — Saturating vs Non-Saturating Generator Loss

**Saturating** (Goodfellow 의 원래 minimax):
$$L_G^\text{sat} = \mathbb{E}_{p_z}[\log(1 - D(G(z)))]$$

**Non-Saturating** (Goodfellow 의 실전 권장):
$$L_G^\text{NS} = -\mathbb{E}_{p_z}[\log D(G(z))]$$

**같은 optimum** ($p_g = p_d$ 에서 같은 $G$), **다른 gradient dynamics**.

### 정리 1.6 — Non-Saturating Loss 의 Gradient

$$\nabla_\theta L_G^\text{NS} = -\mathbb{E}_{p_z}\left[\nabla_\theta \log D(G_\theta(z))\right] = -\mathbb{E}_{p_z}\left[\frac{\nabla_\theta D(G_\theta(z))}{D(G_\theta(z))}\right]$$

$D(G(z)) \approx 0$ 일 때 (초기 generator 가 약할 때):
- Saturating: $\nabla \log(1 - D) = -\nabla D / (1 - D) \approx -\nabla D$ — small
- Non-saturating: $\nabla \log D = \nabla D / D$ — **large** (when $D$ small)

따라서 non-saturating 이 초기에 강한 gradient → 빠른 학습.

### 정의 1.7 — GAN Training Algorithm

```
for iteration in range(N):
    # 1. Update discriminator (k steps)
    for k_steps:
        sample x_real ~ p_d (mini-batch)
        sample z ~ p_z
        update φ to maximize V(D_φ, G_θ)
    # 2. Update generator (1 step)
    sample z ~ p_z
    update θ to minimize -log D(G(z))   (non-saturating)
```

$k = 1$ (alternating, simultaneous update) 이 일반적.

---

## 🔬 증명 및 수학적 유도

### 유도 1 — Optimal Discriminator 의 직관

$V(D, G)$ 가 $D$ 의 함수로 cross-entropy:

$D$ 가 $x \sim p_d$ 를 1, $x \sim p_g$ 를 0 으로 분류해야 좋음. 그러나 $p_d$ 와 $p_g$ 가 같은 $x$ 를 모두 generate 가능 — $D$ 는 $x$ 만 보고 origin 결정 못 함.

**Optimal $D^*(x)$** = $x$ 가 real 일 posterior:

$$P(\text{real} | x) = \frac{P(x | \text{real}) P(\text{real})}{P(x | \text{real}) P(\text{real}) + P(x | \text{fake}) P(\text{fake})}$$

$P(\text{real}) = P(\text{fake}) = 1/2$ (training batch 의 절반씩):

$$D^*(x) = \frac{p_d(x)}{p_d(x) + p_g(x)}$$

### 유도 2 — Saturating Loss 의 Saturation 분석

초기 훈련 ($G$ 약함):
- $G(z) \approx $ noise — easily distinguishable
- $D(G(z)) \approx 0$ for most $z$
- $\log(1 - D(G(z))) \approx \log 1 = 0$
- Gradient $\nabla_\theta \log(1 - D) = -\nabla_\theta D / (1 - D) \approx -\nabla_\theta D$ — small (D 자체가 작음)

**Saturation**: gradient 가 0 에 가까워 학습 정체.

### 유도 3 — Non-Saturating Loss 의 Theoretical Issue

Non-saturating 의 정확한 의미: minimize $-\mathbb{E}[\log D(G(z))]$.

이를 JSD 또는 KL 로 환원하면 (Section 3 of Goodfellow 2014):

$$L_G^\text{NS} \approx \text{KL}(p_g \| p_d) - 2 \text{JSD}(p_d \| p_g) + \text{const}$$

**Reverse KL term** $\text{KL}(p_g \| p_d)$ — mode-seeking. Mode collapse 의 수학적 원인 중 하나.

**Trade-off**:
- Non-saturating: 학습 빠름, mode collapse 위험
- Saturating: 이론적 cleaner, 학습 느림

실전에서는 non-saturating + variance reduction 또는 WGAN (Ch5-04) 으로 이동.

### 유도 4 — Push-Forward Measure 의 Implicit Density

$G_\theta: \mathbb{R}^k \to \mathbb{R}^d$, $z \sim p_z$. $G_\theta$ 가 정의하는 push-forward distribution:

$$p_g = G_\theta \# p_z$$

$$P_g(A) = P_z(G^{-1}(A)) \quad \forall \text{ measurable } A$$

만약 $k < d$ ($z$ 차원이 $x$ 차원보다 작음 — 일반적), $p_g$ 는 lower-dim manifold 에 집중된 singular measure — Lebesgue density 정의 안 됨.

따라서 GAN 은 **density 를 직접 표현 안 함** — sampling 만 가능. Likelihood 평가 불가능.

이것이 GAN 의 "implicit" 본질.

### 유도 5 — Alternating vs Simultaneous Update

**Simultaneous** (한 번에 $G, D$ 모두 update): 일반적인 minimax solver. 하지만 NN 은 high-dim, non-convex → simultaneous 가 collapse 또는 unstable.

**Alternating**: $D$ 를 $k$ steps update, 그 후 $G$ 를 1 step update.

**왜 alternating 가 일반적**:
1. **Inner max approximation**: $D$ 가 잘 학습되어야 $\nabla_G V(D, G)$ 가 informative
2. **Escape local minima**: 두 모델이 turn-by-turn 학습 → step 간 escape
3. **Stability**: simultaneous 가 oscillation 발생 → alternating 이 더 안정

**$k$ 의 영향**:
- $k = 1$: 가장 간단, 일반적
- $k > 1$: $D$ 더 잘 학습하지만 cost 증가, $G$ 가 너무 약해질 수 있음
- WGAN 은 $k = 5$ 권장 (critic 이 1-Lipschitz 학습 필요)

---

## 💻 실험 재현 / PyTorch 구현

### 실험 1 — Toy GAN (1D Mixture of Gaussians)

```python
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt

# Target: bimodal mixture
def sample_real(n):
    return torch.where(torch.rand(n, 1) > 0.5,
                       torch.randn(n, 1) * 0.5 + 2,
                       torch.randn(n, 1) * 0.5 - 2)

# Generator: noise → 1D
class Generator(nn.Module):
    def __init__(self, z_dim=2, hidden=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(z_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, 1),
        )
    def forward(self, z):
        return self.net(z)

# Discriminator
class Discriminator(nn.Module):
    def __init__(self, hidden=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(1, hidden), nn.LeakyReLU(0.2),
            nn.Linear(hidden, hidden), nn.LeakyReLU(0.2),
            nn.Linear(hidden, 1),
        )
    def forward(self, x):
        return self.net(x)
        # 나중에 sigmoid 적용 또는 logits 형태로 BCE loss

G = Generator()
D = Discriminator()
opt_G = optim.Adam(G.parameters(), lr=2e-4, betas=(0.5, 0.999))
opt_D = optim.Adam(D.parameters(), lr=2e-4, betas=(0.5, 0.999))

bce = nn.BCEWithLogitsLoss()

for step in range(5000):
    # Train D
    x_real = sample_real(128)
    z = torch.randn(128, 2)
    x_fake = G(z).detach()

    d_real = D(x_real)
    d_fake = D(x_fake)
    loss_D = bce(d_real, torch.ones_like(d_real)) \
           + bce(d_fake, torch.zeros_like(d_fake))
    opt_D.zero_grad(); loss_D.backward(); opt_D.step()

    # Train G (non-saturating)
    z = torch.randn(128, 2)
    x_fake = G(z)
    d_fake = D(x_fake)
    loss_G = bce(d_fake, torch.ones_like(d_fake))   # want D(fake) = 1
    opt_G.zero_grad(); loss_G.backward(); opt_G.step()

    if step % 500 == 0:
        print(f"Step {step}: D_loss = {loss_D.item():.3f}, G_loss = {loss_G.item():.3f}")

# Visualize
with torch.no_grad():
    z = torch.randn(2000, 2)
    samples = G(z).numpy().flatten()
plt.hist(samples, bins=50, density=True, alpha=0.5, label='Generated')
real = sample_real(2000).numpy().flatten()
plt.hist(real, bins=50, density=True, alpha=0.5, label='Real')
plt.legend(); plt.title('GAN: 1D bimodal'); plt.show()
```

### 실험 2 — Saturating vs Non-Saturating Gradient

```python
# Initial G 가 random, D 가 가까이 0 출력 — gradient magnitude 비교
G_random = Generator()   # untrained
D_trained = Discriminator()
# Train D briefly to discriminate between random G samples and real
# ...

z = torch.randn(128, 2)
x_fake = G_random(z)
d_fake = D_trained(x_fake)
print(f"D(G(z)) mean: {torch.sigmoid(d_fake).mean().item():.4f}")
# Expected: ~0 (D easily distinguishes random G from real)

# Saturating gradient
loss_sat = bce(d_fake, torch.zeros_like(d_fake))   # equivalent to log(1-D)
grad_sat = torch.autograd.grad(loss_sat, G_random.parameters(), retain_graph=True)
norm_sat = sum(g.norm() ** 2 for g in grad_sat) ** 0.5

# Non-saturating gradient
loss_ns = bce(d_fake, torch.ones_like(d_fake))    # equivalent to -log D
grad_ns = torch.autograd.grad(loss_ns, G_random.parameters())
norm_ns = sum(g.norm() ** 2 for g in grad_ns) ** 0.5

print(f"Saturating gradient norm:     {norm_sat.item():.4f}")
print(f"Non-saturating gradient norm: {norm_ns.item():.4f}")
# 일반적: non-saturating 이 10-100배 크다 — 빠른 학습의 이유
```

### 실험 3 — Optimal Discriminator 검증

```python
# 1D toy: p_d, p_g 알려진 분포
import scipy.stats as st

# 분포 정의
mu_d, sigma_d = 0.0, 1.0   # data
mu_g, sigma_g = 1.0, 0.5   # generator

x_grid = torch.linspace(-4, 5, 200)
p_d_grid = torch.exp(-0.5 * ((x_grid - mu_d) / sigma_d) ** 2) / (sigma_d * np.sqrt(2*np.pi))
p_g_grid = torch.exp(-0.5 * ((x_grid - mu_g) / sigma_g) ** 2) / (sigma_g * np.sqrt(2*np.pi))

# Theoretical optimal D
D_optimal = p_d_grid / (p_d_grid + p_g_grid)

# Sample from each, train D
def sample_d(n): return torch.randn(n, 1) * sigma_d + mu_d
def sample_g(n): return torch.randn(n, 1) * sigma_g + mu_g

D_test = Discriminator()
opt_D = optim.Adam(D_test.parameters(), lr=1e-3)
for step in range(3000):
    x_d = sample_d(128)
    x_g = sample_g(128)
    loss = bce(D_test(x_d), torch.ones(128, 1)) + bce(D_test(x_g), torch.zeros(128, 1))
    opt_D.zero_grad(); loss.backward(); opt_D.step()

with torch.no_grad():
    D_learned = torch.sigmoid(D_test(x_grid.unsqueeze(-1))).squeeze()

plt.plot(x_grid, D_optimal, '--', label='D* (theoretical)')
plt.plot(x_grid, D_learned, label='D (learned)')
plt.title('Optimal D vs Learned D')
plt.legend(); plt.show()
# 두 곡선이 거의 일치 — D 가 optimal posterior 에 수렴
```

---

## 🔗 이론과 실전의 간극

### 1. Theoretical Convergence vs Practical Stability

**이론**: 두 모델이 alternating optimization 으로 Nash equilibrium 에 수렴.

**실전**:
- Mode collapse (Ch5-03)
- Discriminator 과학습
- Generator 의 gradient vanishing
- Hyperparameter 민감

해결: WGAN, Spectral Norm, GP, BigGAN 의 architectural choices.

### 2. Architecture 의 영향

GAN 의 quality 는 architecture 에 강하게 의존:
- **DCGAN** (Radford 2016): 첫 본격 CNN GAN, transposed conv 의 도입
- **Progressive growing** (Karras 2018): 저해상도에서 점진적 high-res
- **StyleGAN** (Karras 2019): mapping network + AdaIN, FFHQ 1024×1024
- **BigGAN** (Brock 2019): orthogonal regularization + truncation, ImageNet conditional

### 3. GAN 의 후속 영향

GAN family 가 image generation 의 mainstream 이었음 (2014-2020), Diffusion 의 등장 후 secondary 위치. 단:
- StyleGAN 이 face editing, identity preservation 에서 여전히 SOTA
- GAN-based vocoder (HiFi-GAN) 가 real-time TTS 의 표준
- Adversarial loss 가 다른 model 의 보조 (Diffusion + GAN 의 hybrid)

---

## ⚖️ 가정과 한계

| 가정 | 한계 |
|------|------|
| Two-player zero-sum game 의 Nash 수렴 | NN 의 non-convex 에서 saddle point 문제, oscillation |
| Optimal $D$ 도달 가능 | Capacity 제한 + alternating update 로 부정확 |
| Implicit density | Likelihood 평가 불가능, FID 등 proxy metric |
| Push-forward measure | Manifold 의 dim mismatch 시 singular |
| Alternating 이 simultaneous 보다 안정 | 항상 그런 건 아님, problem dependent |

---

## 📌 핵심 정리

$$\boxed{\min_G \max_D V(D, G) = \mathbb{E}_{p_d}[\log D(x)] + \mathbb{E}_{p_z}[\log(1 - D(G(z)))]}$$

$$\boxed{D^*(x) = \frac{p_d(x)}{p_d(x) + p_g(x)} \quad \text{(Optimal discriminator)}}$$

| 측면 | 값 |
|------|------|
| **Architecture** | $G$ (NN, $z \to x$), $D$ (NN, $x \to [0, 1]$) |
| **Training** | Alternating SGD (k=1 standard) |
| **Optimal $D$** | $p_d / (p_d + p_g)$ |
| **Optimal $G$** | $p_g = p_d$ (다음 챕터: JSD 환원) |
| **Saturating loss** | $\log(1 - D)$ — vanishing gradient at start |
| **Non-saturating** | $-\log D$ — strong initial gradient |
| **Implicit** | $p_g$ closed form 없음 — push-forward |

---

## 🤔 생각해볼 문제

**문제 1** (기초): $V(D, G)$ 의 expectation 들을 expand 해서, fixed $G$ 에서 $D$ 의 optimum 을 직접 미분으로 derive 하라.

<details>
<summary>해설</summary>

$$V(D, G) = \int p_d(x) \log D(x) dx + \int p_g(x) \log(1 - D(x)) dx$$

각 $x$ 에서 integrand: $f(D) = p_d(x) \log D + p_g(x) \log(1 - D)$.

$\partial f / \partial D = p_d / D - p_g / (1 - D) = 0$

$p_d (1 - D) = p_g D$

$p_d - p_d D = p_g D$

$p_d = D(p_g + p_d)$

$D = p_d / (p_d + p_g) = D^*(x)$.

**Second derivative**: $\partial^2 f / \partial D^2 = -p_d / D^2 - p_g / (1 - D)^2 < 0$ — concave, $D^*$ 가 maximum. ✓

**시사점**: optimal $D$ 가 closed-form. 그러나 NN 으로 $D^*$ 를 정확히 표현 못 할 수 있음 — discriminator 의 capacity 가 충분하면 근사 가능.

</details>

**문제 2** (심화): Generator 의 saturating loss 의 gradient 가 작아지는 이유를 더 정량적으로 — $D(G(z)) = \sigma(s(z))$ 로 두고 $s$ (logits) 에 대한 gradient 비교.

<details>
<summary>해설</summary>

**Sigmoid output**: $D = \sigma(s)$, $s$ = logits. $1 - D = \sigma(-s)$.

**Saturating loss**: $L_G = \mathbb{E}[\log(1 - D)] = \mathbb{E}[\log \sigma(-s)]$.

$$\frac{\partial L_G}{\partial s} = \frac{\partial \log \sigma(-s)}{\partial s} = -\sigma(s) = -D$$

(using $d \log \sigma(-s) / ds = -1 + \sigma(s) = -\sigma(-s)$, wait let me redo:

$\log \sigma(-s) = -\log(1 + e^s)$. $d/ds = -e^s / (1 + e^s) = -\sigma(s) = -D$.

$\partial L_G / \partial \theta = -D \cdot \partial s / \partial \theta$ — gradient 는 $D$ 에 비례.

**초기 ($D \approx 0$)**: gradient $\approx 0$. Saturation.

**Non-saturating**: $L_G^\text{NS} = -\mathbb{E}[\log D] = -\mathbb{E}[\log \sigma(s)]$.

$d \log \sigma(s) / ds = \sigma(-s) = 1 - D$.

$\partial L_G^\text{NS} / \partial \theta = -(1 - D) \cdot \partial s / \partial \theta$ — gradient 는 $1 - D$ 에 비례.

**초기 ($D \approx 0$)**: gradient $\approx -1 \cdot \partial s / \partial \theta$ — **strong**.

**비교**:
- Saturating: $\nabla \propto D$ — saturate when $D \to 0$
- Non-saturating: $\nabla \propto 1 - D$ — saturate when $D \to 1$ (생성이 잘 되어 D 가 속을 때)

**Trade-off**: 둘 다 어느 쪽에서 saturate. 초기 학습은 $D \approx 0$ 가 흔하므로 non-saturating 우월. 후기 ($D \approx 1$, 잘 생성) 는 saturating 도 OK.

</details>

**문제 3** (논문 비평): GAN training 의 alternating update 가 simultaneous 보다 안정적이라는 것이 일반적 권고. Goodfellow 2014 의 주장은 "이론적으로는 simultaneous 가 minimax 의 직접 해". 두 접근의 trade-off 를 saddle point dynamics 측면에서 분석하라.

<details>
<summary>해설</summary>

**Simultaneous Update**:
$$\theta^{(t+1)} = \theta^{(t)} - \eta \nabla_\theta V$$
$$\phi^{(t+1)} = \phi^{(t)} + \eta \nabla_\phi V$$

(Generator parameter $\theta$, discriminator $\phi$, both updated using current state)

**문제**: Saddle point 에서 oscillation. 예를 들어 1D quadratic $V(x, y) = xy$ 의 saddle (0, 0) 에서:
$$\dot x = -y, \dot y = x \quad \text{(continuous limit)}$$

Solution: $(x, y) = (\cos t, \sin t)$ — 무한 회전, 수렴 안 함.

**Alternating (Gauss-Seidel-style)**:
$$\theta^{(t+1)} = \theta^{(t)} - \eta \nabla_\theta V(\theta^{(t)}, \phi^{(t)})$$
$$\phi^{(t+1)} = \phi^{(t)} + \eta \nabla_\phi V(\theta^{(t+1)}, \phi^{(t)})$$

(Generator update first, then discriminator with new generator)

**효과**: $\phi$ update 시 새로운 $\theta$ 사용 → 일종의 "implicit" damping. Local minima 회피, oscillation 감소.

**Empirical**:
- Simultaneous: 종종 mode collapse, oscillation
- Alternating with $k = 1$: most common, balanced
- Alternating with $k > 1$ (D update 더): D 가 더 잘 학습 → G 의 informative gradient
- Alternating with $k > 1$ (G update 더): G 가 D 를 압도, equilibrium 깨짐

**현대적 처치**:
- **Two Time-Scale Update Rule** (Heusel 2017 TTUR): $\eta_D > \eta_G$, simultaneous 가능
- **Spectral normalization**: $D$ 의 Lipschitz 제약으로 stability
- **Gradient penalty** (WGAN-GP): saddle point 의 dynamics 개선

**시사점**:
- Alternating 이 일반적 starting point — implementation 간단, stability 우월
- 그러나 두 모델의 capacity 와 update step 의 balance 가 중요
- Saddle point dynamics 의 더 깊은 이해가 GAN training 의 ongoing research

</details>

---

<div align="center">

[◀ 이전 (Ch4-05. CNF)](../ch4-flow/05-cnf-neural-ode.md) | [📚 README](../README.md) | [다음 ▶ (02. JSD 환원)](./02-jsd-reduction.md)

</div>
