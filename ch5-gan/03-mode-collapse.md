# 03. GAN 훈련 불안정성 · Mode Collapse

## 🎯 핵심 질문

- Mode collapse 의 정확한 정의는? Generator 가 일부 mode 만 학습하는 현상의 수학적 표현?
- JSD 의 non-overlapping support 에서 gradient 가 0 인 이유와 mode collapse 의 관계?
- Reverse KL 성향 (non-saturating loss 의) 이 mode collapse 의 원인 중 하나인 이유?
- Nash equilibrium 의 local 성과 oscillation — 왜 GAN training 이 oscillate 하는가?
- Minibatch discrimination, unrolled GAN, mode-seeking regularization — 각 해결책의 메커니즘?

---

## 🔍 왜 Mode Collapse 가 GAN 의 핵심 문제인가

GAN 의 이론은 깔끔: minimax 의 해가 $p_g = p_d$. 그러나 실전에서:

1. **Mode collapse**: $p_g$ 가 $p_d$ 의 일부 mode 만 cover. 예: 10-class MNIST 에서 GAN 이 "1" 과 "7" 만 생성.
2. **Training instability**: Loss 가 oscillate, 수렴 안 함.
3. **Vanishing gradient**: $D$ 가 너무 강해지면 $G$ 의 gradient 0.

이 문제들이 GAN 의 production 사용을 어렵게 만듦. 모든 후속 GAN 작업 (WGAN, Spectral Norm, StyleGAN) 이 이 문제 해결을 시도. 이 문서에서는 **mode collapse 의 수학적 원인** 과 다양한 해결 시도를 다룹니다.

---

## 📐 수학적 선행 조건

- 이전 문서들: 01-minimax, 02-jsd-reduction
- [Information Theory Deep Dive](https://github.com/iq-ai-lab/information-theory-deep-dive): JSD, KL forward vs reverse

---

## 📖 직관적 이해

### "한 mode 만 cover 해도 D 를 속일 수 있다"

GAN 의 generator 가 데이터의 다양한 mode 를 모두 cover 하지 않아도 $D$ 를 속일 수 있는 **simpler** 전략 발견:

- 데이터: 10 modes (e.g., MNIST 10 digits)
- Generator 가 1 mode (e.g., 모든 digit 1) 만 정확히 생성
- $D(G(z))$ 가 모두 0.99 — successfully fool $D$
- $G$ 의 loss 가 매우 낮음 → 학습 정체

이것이 mode collapse — 일부 mode 의 perfect generation 이 모든 mode 의 partial cover 보다 generator loss 가 낮음.

### Reverse KL 성향의 mathematical 설명

Non-saturating loss $-\log D(G(z))$ 의 환원 (Goodfellow 2014 Section 3):

$$L_G^\text{NS} \approx \text{KL}(p_g \| p_d) - 2 \cdot JSD(p_d \| p_g) + \text{const}$$

**Reverse KL** $\text{KL}(p_g \| p_d)$ — mode-seeking. $p_g > 0$ 인 곳에서 $p_d$ 도 커야 finite KL → $p_g$ 가 $p_d$ 의 high-density 영역에 집중 (일부 mode).

**Forward KL** (MLE-like) 은 mass-covering — 모든 mode cover. Saturating loss 가 더 가깝게 forward KL 에 align.

**Trade-off**: non-saturating 의 빠른 학습 vs reverse KL 의 mode-seeking — practical GAN 의 fundamental tension.

### Nash Equilibrium 의 Local 성

Minimax 의 Nash equilibrium: simultaneous best response 의 fixed point. 그러나:

- **Multiple Nash equilibria**: 다른 ($G, D$) 가 모두 Nash 가능
- **Local Nash**: small neighborhood 에서만 Nash, global 아님
- **Cycle**: $G$ 가 mode A 만, 그 후 mode B 만, 다시 A 만... — non-convergent oscillation

이 dynamics 의 분석이 GAN training 의 핵심 연구 영역.

---

## ✏️ 엄밀한 정의·정리

### 정의 3.1 — Mode Collapse

데이터 분포 $p_d$ 가 multimodal ($k$ modes), generator 분포 $p_g$ 가 그 중 $k' < k$ 의 mode 만 cover 하면 **mode collapse** 라 한다.

**정량적**:
- **Coverage**: $\sum_i \mathbb{1}[p_g \text{ covers mode } i] / k$
- **Diversity**: sample 의 perceptual diversity (LPIPS variance, etc.)
- **Recall** (Kynkäänniemi 2019): 진짜 manifold 가 fake manifold 로 cover 되는 비율

### 정의 3.2 — Training Instability

$V(D_\phi, G_\theta)$ 또는 individual losses 가 시간에 따라 oscillate 하거나 발산:

$$\text{Var}_t(L_G(t)) \text{ 또는 } \text{Var}_t(L_D(t)) \text{ 가 큼}$$

수렴 안 함 (vs DCGAN/WGAN 의 안정적 수렴).

### 정리 3.3 — Vanishing Gradient at Disjoint Support

$\text{supp}(p_d) \cap \text{supp}(p_g) = \emptyset$ 이면:
$$JSD(p_d \| p_g) = \log 2 \quad \text{(constant)}$$
$$\nabla_\theta JSD = 0 \Rightarrow \nabla_\theta V(D^*, G_\theta) = 0$$

Generator 학습 정지.

### 정리 3.4 — Non-Saturating Loss 의 Reverse KL 환원

Generator 의 non-saturating loss $L_G^\text{NS} = -\mathbb{E}_{p_g}[\log D(x)]$ 에서, optimal $D = p_d / (p_d + p_g)$ 대입:

$$L_G^\text{NS} = -\mathbb{E}_{p_g}\left[\log \frac{p_d}{p_d + p_g}\right] = \mathbb{E}_{p_g}[\log(p_d + p_g) - \log p_d]$$

수정 (Goodfellow 2014 Section 3):

$$L_G^\text{NS} = \text{KL}(p_g \| p_d) - 2 \text{JSD}(p_d \| p_g) + \log 4$$

**정확한 계산** (위 식 합쳐서):

$L_G^\text{NS}$ 의 minimization 은 $\text{KL}(p_g \| p_d)$ minimization (mode-seeking) + $\text{JSD}$ maximization (다른 영역 push). Mode collapse 의 wirksamkeit.

### 정리 3.5 — Nash Equilibrium 의 Existence (Minimax)

Continuous game $V: \Theta \times \Phi \to \mathbb{R}$ 에서, $\Theta, \Phi$ compact + $V$ continuous + concave/convex 의 일부 조건 하에 Nash equilibrium 존재 (Sion's minimax theorem).

**GAN 의 경우**:
- $\Theta, \Phi$ = NN parameter 공간 — non-compact
- $V$ non-convex, non-concave (NN)

**결과**: Nash equilibrium 의 existence 보장 안 됨, optimization 이 어려움.

### 정의 3.6 — Mitigation 방법들

**Minibatch Discrimination** (Salimans 2016): $D$ 가 batch 내 sample 들 사이의 거리 또는 statistics 도 input 으로 받음. Mode collapse 시 batch 의 sample 들이 비슷 → $D$ 가 detect.

**Unrolled GAN** (Metz 2017): $G$ update 시 $D$ 의 future $k$-step update 를 고려. $G$ 가 "$D$ 가 다음 update 에서 어떻게 반응할지" 예측 → mode collapse 회피.

**Mode-Seeking Regularization** (Mao 2019): explicit loss 항으로 generator 의 diversity 강제:

$$L_\text{div} = -\frac{\|G(z_1) - G(z_2)\|}{\|z_1 - z_2\|}$$

다른 latent → 다른 output 강제.

**Spectral Norm + Self-Attention** (BigGAN): architectural mitigations.

---

## 🔬 증명 및 수학적 유도

### 유도 1 — Mode Collapse 의 Loss Landscape

데이터: $p_d = \frac{1}{2} \delta_a + \frac{1}{2} \delta_b$ (two-point distribution).

Generator: $p_g = \delta_a$ (point mass at $a$) — mode collapse to $a$.

**Optimal $D$**:
- $x = a$: $D^*(a) = p_d(a) / (p_d(a) + p_g(a)) = (1/2) / ((1/2) + 1) = 1/3$
- $x = b$: $D^*(b) = (1/2) / (1/2 + 0) = 1$
- $x \neq a, b$: $D^* = 0$

**$V$ 의 값**:
$$V(D^*, G) = \mathbb{E}_{p_d}[\log D^*] + \mathbb{E}_{p_g}[\log(1 - D^*)]$$
$$= \frac{1}{2} \log(1/3) + \frac{1}{2} \log 1 + 1 \cdot \log(1 - 1/3)$$
$$= -\frac{\log 3}{2} + 0 + \log(2/3) = -\frac{\log 3}{2} + \log 2 - \log 3 = \log 2 - \frac{3}{2} \log 3$$

수치: $\approx 0.693 - 1.648 = -0.955$.

비교: $p_g = p_d$ 일 때 $V = -\log 4 \approx -1.386$.

차이: $-0.955 - (-1.386) = 0.431$ — collapsed generator 가 약간 worse than optimal. **그러나 $G$ 의 gradient signal 이 약하면 collapsed 에서 escape 못 함**.

### 유도 2 — Reverse KL 의 Mode-Seeking 분석

$\text{KL}(p_g \| p_d) = \int p_g \log(p_g / p_d) dx$.

만약 $p_d > 0$ 이지만 $p_g = 0$ in some region: integrand = 0 — **no penalty**.

만약 $p_g > 0$ 이지만 $p_d = 0$: integrand → $\infty$ — **infinite penalty**.

따라서:
- $p_g$ 가 $p_d$ 의 작은 부분만 cover 해도 OK (no penalty for missing mass)
- $p_g$ 가 $p_d$ 가 0 인 곳에 mass 있으면 큰 penalty

→ **Mode-seeking**: $p_g$ 가 $p_d$ 의 high-density 영역에 집중 (일부 mode), 데이터 없는 영역 회피.

### 유도 3 — Forward KL 의 Mass-Covering 비교

$\text{KL}(p_d \| p_g) = \int p_d \log(p_d / p_g) dx$.

$p_d > 0$ 이지만 $p_g = 0$: integrand → $\infty$ — **infinite penalty**.

따라서 $p_g$ 가 모든 $p_d$ 의 support 를 cover 해야 함 — **mass-covering**.

**비교**:
- Forward KL (MLE, VAE, Flow): mass-covering — blurry but covers all modes
- Reverse KL (non-saturating GAN): mode-seeking — sharp but misses modes

이 fundamental asymmetry 가 GAN vs other generative models 의 특성 차이 결정.

### 유도 4 — Nash Equilibrium 의 Local Trap

Toy example: $V(\theta, \phi) = \theta \phi$ (saddle at origin).

Simultaneous gradient descent:
$$\theta^{(t+1)} = \theta^{(t)} - \eta \phi^{(t)}, \quad \phi^{(t+1)} = \phi^{(t)} + \eta \theta^{(t)}$$

Recursion: $(\theta^{(t+1)}, \phi^{(t+1)}) = (\theta^{(t)} - \eta \phi^{(t)}, \phi^{(t)} + \eta \theta^{(t)})$.

$\theta^{(t+1) 2} + \phi^{(t+1) 2} = (\theta - \eta \phi)^2 + (\phi + \eta \theta)^2 = \theta^2 + \phi^2 + \eta^2 (\theta^2 + \phi^2) = (1 + \eta^2)(\theta^2 + \phi^2)$

→ **Distance from origin grows** by factor $\sqrt{1 + \eta^2} > 1$ each step. **Divergent** for any $\eta > 0$.

이는 simple linear 의 saddle 이지만, GAN 의 highly non-convex saddle 에서도 유사한 oscillation 발생 — local Nash 의 trap.

**해결**: Two-time-scale (TTUR), gradient penalty, predictive update (Yadav 2018).

### 유도 5 — Minibatch Discrimination 의 Mechanism

표준 $D$: $D(x_i)$ 가 single sample $x_i$ 만 input. Mode collapse 시 batch 의 모든 $x_i$ 가 비슷 (same mode) — $D$ 가 알지 못함.

**Minibatch discrimination**: $D(x_i, \{x_j\}_{j \neq i})$ 가 batch 의 다른 sample 들도 input. Pairwise distance feature:

$$f_i = \exp(-\|M(x_i) - M(x_j)\|_1)$$

(some learned $M$ projection). Mode collapse 시 모든 $f_i$ 가 1 에 가까움 — $D$ 가 detect.

**효과**: $G$ 가 diverse sample 생성 incentive — mode collapse 회피.

---

## 💻 실험 재현 / PyTorch 구현

### 실험 1 — Mode Collapse 재현 (8-Gaussians)

```python
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt

# 8-Gaussian mixture (well-known mode collapse benchmark)
def sample_8gauss(n):
    centers = [(np.cos(t), np.sin(t)) for t in np.linspace(0, 2*np.pi, 9)[:-1]]
    centers = torch.tensor(centers).float() * 2.0
    idx = torch.randint(0, 8, (n,))
    samples = centers[idx] + torch.randn(n, 2) * 0.05
    return samples

class G(nn.Module):
    def __init__(self, z_dim=2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(z_dim, 128), nn.ReLU(),
            nn.Linear(128, 128), nn.ReLU(),
            nn.Linear(128, 2),
        )
    def forward(self, z):
        return self.net(z)

class D(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2, 128), nn.LeakyReLU(0.2),
            nn.Linear(128, 128), nn.LeakyReLU(0.2),
            nn.Linear(128, 1),
        )
    def forward(self, x):
        return self.net(x)

G_, D_ = G(), D()
opt_G = optim.Adam(G_.parameters(), lr=1e-4, betas=(0.5, 0.9))
opt_D = optim.Adam(D_.parameters(), lr=1e-4, betas=(0.5, 0.9))
bce = nn.BCEWithLogitsLoss()

losses_G, losses_D = [], []
for step in range(20000):
    # Train D
    x_real = sample_8gauss(256)
    z = torch.randn(256, 2)
    x_fake = G_(z).detach()
    loss_D = bce(D_(x_real), torch.ones(256, 1)) + bce(D_(x_fake), torch.zeros(256, 1))
    opt_D.zero_grad(); loss_D.backward(); opt_D.step()

    # Train G
    z = torch.randn(256, 2)
    x_fake = G_(z)
    loss_G = bce(D_(x_fake), torch.ones(256, 1))
    opt_G.zero_grad(); loss_G.backward(); opt_G.step()

    losses_G.append(loss_G.item()); losses_D.append(loss_D.item())

# Sample 후 시각화
with torch.no_grad():
    z = torch.randn(2000, 2)
    samples = G_(z).numpy()
real = sample_8gauss(2000).numpy()
plt.scatter(real[:, 0], real[:, 1], s=2, c='red', alpha=0.3, label='Real')
plt.scatter(samples[:, 0], samples[:, 1], s=2, alpha=0.5, label='Generated')
plt.title('Standard GAN (often mode collapses)')
plt.legend(); plt.show()

# 종종 8 modes 중 1-3 개만 cover (mode collapse)
```

### 실험 2 — Mode Coverage Metric 측정

```python
def measure_mode_coverage(samples, centers, threshold=0.3):
    """각 mode 가 sample 에 의해 cover 되는지"""
    samples = torch.tensor(samples) if isinstance(samples, np.ndarray) else samples
    coverage = []
    for c in centers:
        c = torch.tensor(c).float()
        d = (samples - c).norm(dim=-1)
        # sample 이 mode c 의 threshold 안에 있는지
        in_mode = (d < threshold).any().item()
        coverage.append(in_mode)
    return sum(coverage) / len(coverage), coverage

centers = [(np.cos(t)*2, np.sin(t)*2) for t in np.linspace(0, 2*np.pi, 9)[:-1]]
cov, individual = measure_mode_coverage(samples, centers)
print(f"Mode coverage: {cov:.2f} ({sum(individual)}/{len(centers)} modes)")
```

### 실험 3 — Minibatch Discrimination 구현

```python
class MinibatchDiscrimination(nn.Module):
    def __init__(self, in_features, out_features=64, kernel_dims=16):
        super().__init__()
        self.T = nn.Parameter(torch.randn(in_features, out_features, kernel_dims))

    def forward(self, x):
        # x: [B, in_features]
        # Project: x → [B, out_features, kernel_dims]
        M = x.unsqueeze(2).matmul(self.T.permute(1, 0, 2))   # [B, out_features, kernel_dims]
        # Pairwise L1 distance
        diff = (M.unsqueeze(0) - M.unsqueeze(1)).abs().sum(-1)   # [B, B, out_features]
        # Sum exp(-d)
        c = torch.exp(-diff).sum(0) - 1   # subtract self
        return torch.cat([x, c], dim=-1)

class D_with_minibatch(nn.Module):
    def __init__(self):
        super().__init__()
        self.feat = nn.Sequential(
            nn.Linear(2, 128), nn.LeakyReLU(0.2),
            nn.Linear(128, 128), nn.LeakyReLU(0.2),
        )
        self.mbd = MinibatchDiscrimination(128)
        self.out = nn.Linear(128 + 64, 1)

    def forward(self, x):
        f = self.feat(x)
        f = self.mbd(f)
        return self.out(f)

# Train with this D — usually shows better mode coverage
```

### 실험 4 — Loss Curve 의 Oscillation

```python
plt.subplot(2, 1, 1)
plt.plot(losses_G[-2000:], label='G loss')
plt.plot(losses_D[-2000:], label='D loss')
plt.legend()
plt.title('GAN training: oscillating losses')

plt.subplot(2, 1, 2)
# Moving average
window = 100
mavg_G = np.convolve(losses_G, np.ones(window)/window, mode='valid')
mavg_D = np.convolve(losses_D, np.ones(window)/window, mode='valid')
plt.plot(mavg_G, label='G (moving avg)')
plt.plot(mavg_D, label='D (moving avg)')
plt.legend()
plt.show()
# Loss 가 정확히 수렴 안 하고 oscillation — 일반적 GAN training
```

---

## 🔗 이론과 실전의 간극

### 1. Mode Collapse 의 Diagnosis

진단 방법:
- **Mode coverage**: 진짜 mode 들을 cover 하는 비율
- **Sample diversity**: LPIPS, FID 의 recall
- **Latent traversal**: 다른 $z$ 가 비슷한 $G(z)$ 줄 때 collapse

### 2. Stability 향상의 Architectural Choices

- **Spectral Normalization** (Ch5-05): $D$ 의 Lipschitz 제약
- **Self-Attention** (SAGAN): long-range dependency
- **Two-Time-Scale Update Rule** (TTUR): $\eta_D > \eta_G$
- **Orthogonal Regularization**: weight 의 orthogonality

### 3. WGAN 으로의 이동

표준 GAN 의 mode collapse 가 fundamental — JSD 의 본질. **WGAN** (Ch5-04) 이 Wasserstein-1 distance 사용하여 이 문제를 architectural 으로 해결. WGAN-GP, Spectral Norm 이 표준 GAN 의 대안.

---

## ⚖️ 가정과 한계

| 가정 | 한계 |
|------|------|
| Mode collapse 가 항상 reverse KL 때문 | Architectural, optimization 도 영향 |
| Nash equilibrium 도달 가능 | NN 의 non-convex 에서 어려움 |
| Mitigation 이 항상 작동 | Dataset, hyperparameter dependent |
| FID 가 mode coverage 측정 | Imperfect, P/R 또는 explicit count 필요 |
| JSD 가 GAN 의 objective | Non-saturating loss 는 사실 다른 form |

---

## 📌 핵심 정리

$$\boxed{\text{Mode Collapse: } p_g \text{ covers only some modes of } p_d}$$

$$\boxed{\text{JSD saturates at } \log 2 \text{ with disjoint support} \Rightarrow \nabla_G V = 0}$$

| 원인 | 메커니즘 |
|------|---------|
| **Reverse KL** (non-saturating) | $\text{KL}(p_g \| p_d)$ — mode-seeking |
| **JSD saturation** | Disjoint support → vanishing gradient |
| **Local Nash** | Saddle dynamics → cycling |
| **D over-training** | $D$ too strong → $G$ gradient 약함 |

| 해결책 | 메커니즘 |
|--------|---------|
| **Minibatch discrimination** | $D$ 가 batch diversity 봄 |
| **Unrolled GAN** | $G$ 가 future $D$ updates 고려 |
| **Mode-seeking reg** | Explicit diversity loss |
| **Spectral Norm** (Ch5-05) | $D$ Lipschitz 제약 |
| **Wasserstein** (Ch5-04) | Different distance, no saturation |
| **TTUR** | $\eta_D > \eta_G$ |

---

## 🤔 생각해볼 문제

**문제 1** (기초): Two-point data $p_d = (1/2)\delta_a + (1/2)\delta_b$ 에서 mode collapse 된 $p_g = \delta_a$ 의 $V(D^*, G)$ 와 $JSD(p_d \| p_g)$ 를 직접 계산하라.

<details>
<summary>해설</summary>

**JSD**:
$$m = (p_d + p_g)/2 = (1/4) \delta_a + (1/4) \delta_b + (1/2) \delta_a = (3/4) \delta_a + (1/4) \delta_b$$

$\text{KL}(p_d \| m)$:
- At $a$: $p_d(a) \log(p_d(a)/m(a)) = (1/2) \log((1/2)/(3/4)) = (1/2) \log(2/3)$
- At $b$: $p_d(b) \log(p_d(b)/m(b)) = (1/2) \log((1/2)/(1/4)) = (1/2) \log 2$

Sum: $(1/2)[\log(2/3) + \log 2] = (1/2) \log(4/3)$.

$\text{KL}(p_g \| m)$:
- At $a$: $p_g(a) \log(p_g(a)/m(a)) = 1 \cdot \log(1/(3/4)) = \log(4/3)$
- At $b$: $p_g(b) = 0$, term = 0

Sum: $\log(4/3)$.

$JSD = (1/2)[(1/2)\log(4/3) + \log(4/3)] = (1/2)(3/2)\log(4/3) = (3/4)\log(4/3)$.

수치: $(3/4) \cdot 0.2877 \approx 0.216$.

**$V(D^*, G)$**: $2 JSD - \log 4 = 2 \cdot 0.216 - 1.386 = -0.954$.

이전 직접 계산 ($\log 2 - (3/2) \log 3 \approx -0.955$) 와 일치.

**비교**:
- Optimal ($p_g = p_d$): $V = -\log 4 \approx -1.386$
- Mode collapsed: $V \approx -0.954$
- 차이: $0.432$ — 비교적 작음

**시사점**: collapsed 가 약간 worse, 그러나 gradient signal 약하면 escape 못 함. Mode collapse 의 attractor 성질.

</details>

**문제 2** (심화): Non-saturating loss $L_G = -\mathbb{E}_{p_g}[\log D(x)]$ 에서 optimal $D = p_d / (p_d + p_g)$ 대입하면 reverse KL 항이 등장한다는 정확한 환원을 derive 하라.

<details>
<summary>해설</summary>

$L_G^\text{NS} = -\mathbb{E}_{p_g}[\log D^*] = -\mathbb{E}_{p_g}\left[\log \frac{p_d}{p_d + p_g}\right]$

$$= \mathbb{E}_{p_g}[\log(p_d + p_g) - \log p_d]$$

$$= \mathbb{E}_{p_g}[\log p_d + \log(1 + p_g/p_d) - \log p_d]$$

이 형태는 직접 reverse KL 안 됨. 다른 manipulation:

$$L_G^\text{NS} = \mathbb{E}_{p_g}\left[\log\left(\frac{p_d + p_g}{p_d}\right)\right] = \mathbb{E}_{p_g}\left[\log\left(1 + \frac{p_g}{p_d}\right)\right]$$

이는 reverse KL 의 변형 — $\log(p_g/p_d)$ 가 아닌 $\log(1 + p_g/p_d)$.

**Goodfellow 2014 의 공식적 환원** (Section 3, Theorem 4 의 derivation):

$L_G^\text{NS} - V(D^*, G)$ 의 차이 (saturating 와의 비교):
$$L_G^\text{NS} - L_G^\text{sat} = -\mathbb{E}_{p_g}[\log D^*] - \mathbb{E}_{p_g}[\log(1 - D^*)]$$
$$= -\mathbb{E}_{p_g}[\log D^* + \log(1 - D^*)]$$

이 차이를 분석하여 reverse KL 항 등장. 정확한 표현 (Goodfellow 2014):

$$L_G^\text{NS} = \text{KL}(p_g \| p_d) - 2 JSD(p_d \| p_g) + \log 4$$

**핵심**: reverse KL term 이 mode-seeking, JSD term 이 mass-covering. 두 항의 conflict 가 generator dynamics 를 지배.

**시사점**: non-saturating loss 가 JSD 만 minimize 하지 않음 — reverse KL 도 (mode-seeking) + JSD (covering) 의 tension. 실전 GAN 의 mode collapse 의 수학적 origin.

</details>

**문제 3** (논문 비평): WGAN (Ch5-04) 가 standard GAN 의 mode collapse 와 vanishing gradient 를 어떻게 해결하는지 — Wasserstein-1 distance 의 saturation 부재 측면에서 설명하라.

<details>
<summary>해설</summary>

**Standard GAN 의 문제** (above):
1. JSD = $\log 2$ saturation in disjoint support → vanishing gradient
2. Reverse KL → mode-seeking → mode collapse

**WGAN 의 Wasserstein-1**:
$$W_1(p_d, p_g) = \inf_{\gamma \in \Pi(p_d, p_g)} \mathbb{E}_{(x, y) \sim \gamma}[\|x - y\|]$$

**Saturation 부재**:
- Disjoint support 에서도 $W_1 \neq $ constant
- 두 mode 의 거리에 비례 — gradient 가 분포 거리에 따라 informative
- $W_1 = 0 \iff p_d = p_g$ (continuous, 미분 가능 in $G$)

**Mode-Seeking 부재**:
- Wasserstein 은 transport-based, "$p_g$ mass 를 어디 옮길지" — mode 무시 안 됨
- Implicit penalty for missing mode (transport cost)
- Symmetric (forward/reverse 구분 없음) — KL 의 비대칭 회피

**Kantorovich-Rubinstein dual**:
$$W_1 = \sup_{\|f\|_L \leq 1} \mathbb{E}_{p_d}[f] - \mathbb{E}_{p_g}[f]$$

WGAN 은 1-Lipschitz $D$ (called critic) 로 위 sup 추정. Standard GAN 의 sigmoid $D$ 와 다름 — saturation 없음.

**Empirical 효과**:
- Stable training (loss 가 monotone, 측정 가능)
- Mode collapse 거의 없음
- Hyperparameter 덜 민감
- 이론과 실전의 매칭이 tighter

**Trade-off**:
- 1-Lipschitz 강제가 어려움 (weight clipping → bias, gradient penalty → 비용)
- WGAN-GP 가 5-10× slower than standard GAN
- Sample quality 가 항상 우월하지 않음 (Spectral Norm 이 alternative)

**시사점**: GAN training 의 fundamental issues 가 distance metric 의 선택 문제. JSD 의 architectural fit 보다 Wasserstein 이 high-dim NN 에 더 robust. WGAN 이 GAN 의 next-generation 의 표준이 됨.

</details>

---

<div align="center">

[◀ 이전 (02. JSD 환원)](./02-jsd-reduction.md) | [📚 README](../README.md) | [다음 ▶ (04. WGAN)](./04-wgan.md)

</div>
