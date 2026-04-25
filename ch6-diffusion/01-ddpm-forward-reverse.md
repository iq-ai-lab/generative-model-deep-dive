# 01. DDPM — Forward · Reverse (Ho 2020)

## 🎯 핵심 질문

- Forward process $q(x_t | x_{t-1}) = \mathcal{N}(\sqrt{1 - \beta_t} x_{t-1}, \beta_t I)$ 는 어떤 의미인가?
- Closed-form $q(x_t | x_0) = \mathcal{N}(\sqrt{\bar\alpha_t} x_0, (1 - \bar\alpha_t) I)$ 의 derivation — reparameterization chain 은 어떻게 single Gaussian 으로 환원되는가?
- Reverse process $p_\theta(x_{t-1} | x_t)$ 가 왜 Gaussian 으로 modeled? 작은 $\beta_t$ 의 가정?
- ELBO 의 chain decomposition $L = L_T + \sum_{t=2}^T L_{t-1} + L_0$ 의 각 항의 의미는?
- Variance schedule $\beta_t$ (linear vs cosine) 의 선택이 generation quality 에 미치는 영향?

---

## 🔍 왜 DDPM 이 결정적 발견인가

Ho 2020 "Denoising Diffusion Probabilistic Models" 가 diffusion 을 high-quality image generation 의 mainstream 으로 만든 작업:

1. **Simple training objective** — $L_\text{simple} = \mathbb{E}[\|\epsilon - \epsilon_\theta\|^2]$ (Ch6-02)
2. **Unconditional FID 우수** — CIFAR-10 에서 $3.17$, BigGAN ($14.7$) 압도
3. **Stable training** — GAN 의 instability 없음
4. **Theoretical foundation** — Sohl-Dickstein 2015 의 thermodynamics-inspired idea 의 NN 적용

이 작업이 후속 모든 diffusion model (Stable Diffusion, DALL-E 2, Imagen, Sora) 의 기반. 이 문서에서는 DDPM 의 forward/reverse process 의 정확한 수학과 ELBO 분해를 다룹니다.

---

## 📐 수학적 선행 조건

- [Probability Theory Deep Dive](https://github.com/iq-ai-lab/probability-theory-deep-dive): Markov chain, Gaussian distribution
- [Bayesian ML Deep Dive](https://github.com/iq-ai-lab/bayesian-ml-deep-dive): ELBO, KL divergence
- 이전 챕터: Ch3 (VAE 의 ELBO)

---

## 📖 직관적 이해

### "점진적 노이즈 주입과 제거"

**Forward (fixed)**: 데이터 $x_0$ 에 점진적으로 Gaussian noise 추가 — $T$ steps 후 $x_T \approx \mathcal{N}(0, I)$ (pure noise).

$$x_0 \to x_1 \to x_2 \to \cdots \to x_T$$

각 step: $x_t = \sqrt{1 - \beta_t} x_{t-1} + \sqrt{\beta_t} \epsilon$, $\epsilon \sim \mathcal{N}(0, I)$.

**Reverse (learned)**: pure noise $x_T$ 에서 점진적으로 noise 제거 — $T$ steps 후 $x_0$ (data sample).

$$x_T \to x_{T-1} \to \cdots \to x_1 \to x_0$$

각 step: NN $p_\theta(x_{t-1} | x_t)$ 학습 — "noisy 한 이미지에서 약간 cleaner 한 이미지 추정".

### "왜 Forward 가 Fixed?"

VAE: encoder $q_\phi(z|x)$ 학습. Diffusion: forward $q(x_t | x_{t-1})$ **fixed** (no learning).

**장점**:
- No encoder collapse (VAE 의 posterior collapse 같은 문제 없음)
- Closed-form $q(x_t | x_0)$ — 효율적 training
- Simple — only reverse (decoder) 학습

**효과**: chain 의 분해된 ELBO 가 stable한 training signal 제공.

### "Closed-Form Forward"

$x_1, x_2, \ldots, x_t$ 가 Markov chain 이지만 **모든 conditional 이 Gaussian**, reparameterization 으로 single Gaussian 결합 가능:

$$x_t = \sqrt{\bar\alpha_t} x_0 + \sqrt{1 - \bar\alpha_t} \epsilon, \quad \epsilon \sim \mathcal{N}(0, I)$$

여기서 $\bar\alpha_t = \prod_{s=1}^t \alpha_s$, $\alpha_t = 1 - \beta_t$.

이로 임의 $t$ 의 $x_t$ 를 한 step 에 sampling 가능 — training 효율.

---

## ✏️ 엄밀한 정의·정리

### 정의 1.1 — Forward Process (Diffusion Kernel)

Variance schedule $\{\beta_t\}_{t=1}^T$ 가 주어졌을 때 ($0 < \beta_t < 1$):

$$q(x_t | x_{t-1}) = \mathcal{N}(x_t; \sqrt{1 - \beta_t} x_{t-1}, \beta_t I)$$

Markov chain:

$$q(x_{1:T} | x_0) = \prod_{t=1}^T q(x_t | x_{t-1})$$

### 정리 1.2 — Closed-Form Forward

$\alpha_t = 1 - \beta_t$, $\bar\alpha_t = \prod_{s=1}^t \alpha_s$. 그러면:

$$q(x_t | x_0) = \mathcal{N}(x_t; \sqrt{\bar\alpha_t} x_0, (1 - \bar\alpha_t) I)$$

또는 reparameterization:

$$x_t = \sqrt{\bar\alpha_t} x_0 + \sqrt{1 - \bar\alpha_t} \epsilon, \quad \epsilon \sim \mathcal{N}(0, I)$$

### 증명 (정리 1.2)

귀납. $t = 1$: $q(x_1 | x_0) = \mathcal{N}(\sqrt{\alpha_1} x_0, (1 - \alpha_1) I)$ — $\bar\alpha_1 = \alpha_1$, OK.

귀납 가정: $q(x_{t-1} | x_0) = \mathcal{N}(\sqrt{\bar\alpha_{t-1}} x_0, (1 - \bar\alpha_{t-1}) I)$.

$q(x_t | x_0) = \int q(x_t | x_{t-1}) q(x_{t-1} | x_0) dx_{t-1}$.

Two Gaussians 의 마진:
- $x_{t-1} | x_0 \sim \mathcal{N}(\sqrt{\bar\alpha_{t-1}} x_0, (1-\bar\alpha_{t-1}) I)$
- $x_t | x_{t-1} \sim \mathcal{N}(\sqrt{\alpha_t} x_{t-1}, \beta_t I)$

Reparameterize: $x_{t-1} = \sqrt{\bar\alpha_{t-1}} x_0 + \sqrt{1 - \bar\alpha_{t-1}} \epsilon_{t-1}$, $\epsilon_{t-1} \sim \mathcal{N}(0, I)$.

$x_t = \sqrt{\alpha_t} x_{t-1} + \sqrt{\beta_t} \epsilon_t$
$= \sqrt{\alpha_t \bar\alpha_{t-1}} x_0 + \sqrt{\alpha_t (1-\bar\alpha_{t-1})} \epsilon_{t-1} + \sqrt{\beta_t} \epsilon_t$
$= \sqrt{\bar\alpha_t} x_0 + \sqrt{\alpha_t (1-\bar\alpha_{t-1}) + \beta_t} \epsilon$ (combining two independent Gaussians)

$\alpha_t (1 - \bar\alpha_{t-1}) + \beta_t = \alpha_t - \alpha_t \bar\alpha_{t-1} + \beta_t = \alpha_t - \bar\alpha_t + 1 - \alpha_t = 1 - \bar\alpha_t$.

따라서 $x_t = \sqrt{\bar\alpha_t} x_0 + \sqrt{1 - \bar\alpha_t} \epsilon$. $\square$

### 정리 1.3 — Forward Posterior $q(x_{t-1} | x_t, x_0)$

$q(x_{t-1} | x_t, x_0)$ 도 Gaussian (Bayes' rule on Gaussians):

$$q(x_{t-1} | x_t, x_0) = \mathcal{N}(x_{t-1}; \tilde\mu_t(x_t, x_0), \tilde\beta_t I)$$

여기서:

$$\tilde\mu_t(x_t, x_0) = \frac{\sqrt{\bar\alpha_{t-1}} \beta_t}{1 - \bar\alpha_t} x_0 + \frac{\sqrt{\alpha_t}(1 - \bar\alpha_{t-1})}{1 - \bar\alpha_t} x_t$$

$$\tilde\beta_t = \frac{1 - \bar\alpha_{t-1}}{1 - \bar\alpha_t} \beta_t$$

이는 reverse process 의 ground truth — $x_0$ 와 $x_t$ 가 모두 알려졌을 때.

### 정의 1.4 — Reverse Process

NN $\theta$ 로 parameterized:

$$p_\theta(x_{t-1} | x_t) = \mathcal{N}(x_{t-1}; \mu_\theta(x_t, t), \Sigma_\theta(x_t, t))$$

$x_T \sim \mathcal{N}(0, I)$ 에서 시작, $T$ step reverse:

$$p_\theta(x_{0:T}) = p(x_T) \prod_{t=1}^T p_\theta(x_{t-1} | x_t)$$

### 정리 1.5 — Reverse 가 Gaussian 인 정당성

Forward $q(x_t | x_{t-1})$ 가 $\beta_t$ small 인 Gaussian → reverse $q(x_{t-1} | x_t)$ 도 approximately Gaussian (Sohl-Dickstein 2015).

**$\beta_t$ small 가정의 의미**: 각 step 의 noise 가 작을수록 reverse approximation 정확. 그러나 너무 작으면 $T$ 가 커져야 함 → trade-off.

표준: $T = 1000$, $\beta_t \in [10^{-4}, 0.02]$ (linear schedule).

### 정리 1.6 — ELBO Decomposition

DDPM 의 ELBO:

$$\log p(x_0) \geq \mathbb{E}_q\left[\log \frac{p_\theta(x_{0:T})}{q(x_{1:T} | x_0)}\right] = -L$$

$L$ 의 분해:

$$L = \underbrace{\text{KL}(q(x_T | x_0) \| p(x_T))}_{L_T} + \sum_{t=2}^T \underbrace{\text{KL}(q(x_{t-1} | x_t, x_0) \| p_\theta(x_{t-1} | x_t))}_{L_{t-1}} - \underbrace{\mathbb{E}_q[\log p_\theta(x_0 | x_1)]}_{L_0}$$

**해석**:
- $L_T$: forward 의 final 이 prior 와 일치 — $\beta$ schedule 잘 설계되면 거의 0
- $L_{t-1}$: 각 reverse step 의 KL — main learning signal
- $L_0$: discrete pixel reconstruction (continuous → discrete 처리)

---

## 🔬 증명 및 수학적 유도

### 유도 1 — Variance 가 1 - $\bar\alpha_t$ 인 직관

$x_t = \sqrt{\bar\alpha_t} x_0 + \sqrt{1 - \bar\alpha_t} \epsilon$ — signal $\sqrt{\bar\alpha_t} x_0$, noise $\sqrt{1 - \bar\alpha_t} \epsilon$.

$\bar\alpha_t$ 가 0 → 1 사이 schedule:
- $t = 0$: $\bar\alpha_0 = 1$ — $x_t = x_0$ (no noise)
- $t = T$: $\bar\alpha_T \approx 0$ — $x_T \approx \epsilon$ (pure noise)

**Signal-to-Noise Ratio**:

$$\text{SNR}(t) = \frac{\bar\alpha_t}{1 - \bar\alpha_t}$$

큰 SNR (small $t$): signal dominant. 작은 SNR (large $t$): noise dominant.

### 유도 2 — $q(x_{t-1} | x_t, x_0)$ 의 Gaussian Posterior

Bayes:

$$q(x_{t-1} | x_t, x_0) = \frac{q(x_t | x_{t-1}, x_0) q(x_{t-1} | x_0)}{q(x_t | x_0)}$$

$q(x_t | x_{t-1}, x_0) = q(x_t | x_{t-1})$ (Markov property).

세 Gaussian 의 비율 → exponent 의 quadratic form → Gaussian. After algebra:

$$q(x_{t-1} | x_t, x_0) \propto \exp\left[-\frac{1}{2}\left(\frac{(x_t - \sqrt{\alpha_t} x_{t-1})^2}{\beta_t} + \frac{(x_{t-1} - \sqrt{\bar\alpha_{t-1}} x_0)^2}{1 - \bar\alpha_{t-1}}\right)\right]$$

Quadratic 의 complete the square → mean and variance as in 정리 1.3.

### 유도 3 — Variance Schedule 의 영향

**Linear schedule** (Ho 2020): $\beta_t = 10^{-4} + (t - 1) \cdot (0.02 - 10^{-4}) / T$. Simple, works on $32 \times 32$.

**Cosine schedule** (Nichol 2021):

$$\bar\alpha_t = \cos^2\left(\frac{\pi/2 \cdot (t/T + 0.008)}{1 + 0.008}\right)$$

$\bar\alpha_t$ 가 cosine 형태 — initial steps (small $t$) 에서 더 천천히 noise 추가, 더 informative early steps. 64×64+ 에서 우월.

**효과**:
- Linear: end of forward 에서 SNR 빠르게 0 → 후반 steps 가 거의 무용
- Cosine: smooth SNR 변화 → all steps informative
- FID 약간 개선, NLL 비슷

### 유도 4 — ELBO Decomposition 의 Step-by-Step

Standard ELBO:

$$L = -\log p(x_0) \leq \mathbb{E}_q\left[-\log \frac{p(x_{0:T})}{q(x_{1:T} | x_0)}\right]$$

$\log \frac{p(x_{0:T})}{q(x_{1:T}|x_0)}$ 를 분해:

$$= \log p(x_T) + \sum_t \log \frac{p_\theta(x_{t-1}|x_t)}{q(x_t|x_{t-1})}$$

$q(x_t | x_{t-1}) = q(x_t | x_{t-1}, x_0) = q(x_{t-1} | x_t, x_0) \cdot q(x_t | x_0) / q(x_{t-1} | x_0)$ (Bayes).

대입 후 정리:

$$L = \mathbb{E}_q[-\log p(x_T) + \text{KL}(q(x_{T-1}|x_T, x_0) \| p_\theta) + \cdots + \text{KL}(\ldots) - \log p_\theta(x_0|x_1)]$$

(자세한 algebra 은 Ho 2020 Appendix). 결과: 각 step KL + boundary terms.

### 유도 5 — Reverse Process Variance 의 Choice

$\Sigma_\theta(x_t, t)$ 의 선택:
- **Fixed** $\Sigma_\theta = \beta_t I$ (Ho 2020): 최대 entropy 의 forward
- **Fixed** $\Sigma_\theta = \tilde\beta_t I$ (Ho 2020 alternative): forward posterior 의 variance
- **Learned** $\Sigma_\theta(x_t, t)$ (Improved DDPM, Nichol 2021): NN 으로 학습

Ho 2020: 둘 다 비슷한 quality. 단순 fixed $\beta_t$ 선택. $\Sigma_\theta = \tilde\beta_t I$ 가 약간 better NLL.

---

## 💻 실험 재현 / PyTorch 구현

### 실험 1 — Forward Process 구현

```python
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

T = 1000

# Linear beta schedule
betas = torch.linspace(1e-4, 0.02, T)
alphas = 1 - betas
alpha_bars = torch.cumprod(alphas, dim=0)

def q_sample(x_0, t, noise=None):
    """Sample x_t from q(x_t | x_0) — closed form"""
    if noise is None:
        noise = torch.randn_like(x_0)
    sqrt_ab = alpha_bars[t].sqrt()
    sqrt_1mab = (1 - alpha_bars[t]).sqrt()
    while sqrt_ab.dim() < x_0.dim():
        sqrt_ab = sqrt_ab.unsqueeze(-1)
        sqrt_1mab = sqrt_1mab.unsqueeze(-1)
    return sqrt_ab * x_0 + sqrt_1mab * noise

# 시각화: $x_0$ 부터 $x_T$ 까지의 점진적 noising
x_0 = torch.randn(1, 1, 32, 32) * 0.5   # toy
t_list = [0, 100, 500, 999]
fig, axes = plt.subplots(1, len(t_list), figsize=(15, 3))
for ax, t in zip(axes, t_list):
    t_tensor = torch.tensor([t])
    x_t = q_sample(x_0, t_tensor).squeeze().numpy()
    ax.imshow(x_t, cmap='gray')
    ax.set_title(f't = {t}')
plt.show()
# t=0: 원본, t=999: 거의 pure noise
```

### 실험 2 — SNR Schedule 비교

```python
# Linear vs Cosine alpha_bar
T = 1000
linear_alpha_bars = torch.cumprod(1 - torch.linspace(1e-4, 0.02, T), dim=0)

def cosine_alpha_bars(T, s=0.008):
    t_norm = torch.arange(T + 1) / T
    f = torch.cos((t_norm + s) / (1 + s) * np.pi / 2) ** 2
    return f[1:] / f[0]

cosine_ab = cosine_alpha_bars(T)

# SNR(t) = alpha_bar / (1 - alpha_bar)
linear_snr = linear_alpha_bars / (1 - linear_alpha_bars)
cosine_snr = cosine_ab / (1 - cosine_ab)

plt.semilogy(linear_snr.numpy(), label='Linear')
plt.semilogy(cosine_snr.numpy(), label='Cosine')
plt.xlabel('Timestep t'); plt.ylabel('SNR')
plt.legend(); plt.title('Variance Schedule Comparison')
plt.show()
# Linear: SNR 이 빠르게 0 → 후반 steps 무용
# Cosine: 부드러운 SNR 변화
```

### 실험 3 — Reverse Process Sampling

```python
class SimpleEpsilonNet(nn.Module):
    """Toy ε prediction NN — given x_t, t → predict ε"""
    def __init__(self, dim=2, hidden=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim + 1, hidden), nn.SiLU(),
            nn.Linear(hidden, hidden), nn.SiLU(),
            nn.Linear(hidden, dim),
        )
    def forward(self, x, t):
        t_norm = (t.float() / T).unsqueeze(-1)
        return self.net(torch.cat([x, t_norm], dim=-1))

@torch.no_grad()
def sample_loop(model, shape, T_steps=T):
    x = torch.randn(shape)
    for t in reversed(range(T_steps)):
        t_tensor = torch.tensor([t]).expand(x.size(0))
        # μ_θ(x_t, t)
        eps_pred = model(x, t_tensor)
        coef1 = 1 / alphas[t].sqrt()
        coef2 = betas[t] / (1 - alpha_bars[t]).sqrt()
        mu = coef1 * (x - coef2 * eps_pred)
        # Variance
        if t > 0:
            sigma = betas[t].sqrt()
            x = mu + sigma * torch.randn_like(x)
        else:
            x = mu
    return x

# 학습 후 (Ch6-02 의 L_simple), 위 함수로 샘플 생성
```

### 실험 4 — Forward Posterior $q(x_{t-1} | x_t, x_0)$ 의 시각화

```python
# 1D 데이터에 대해 forward posterior 의 mean/variance 시각화
x_0 = torch.tensor([1.0])
t = 100
x_t = q_sample(x_0.unsqueeze(0).unsqueeze(0).unsqueeze(0),
                torch.tensor([t])).squeeze()
print(f"x_t (sample): {x_t.item():.3f}")

# Forward posterior parameters
sqrt_ab_prev = alpha_bars[t-1].sqrt()
mu_tilde = (sqrt_ab_prev * betas[t] / (1 - alpha_bars[t])) * x_0 \
         + (alphas[t].sqrt() * (1 - alpha_bars[t-1]) / (1 - alpha_bars[t])) * x_t
beta_tilde = ((1 - alpha_bars[t-1]) / (1 - alpha_bars[t])) * betas[t]

print(f"q(x_{{t-1}} | x_t, x_0): N({mu_tilde:.3f}, {beta_tilde:.5f})")
# 이 distribution 이 reverse process 의 ground truth
```

---

## 🔗 이론과 실전의 간극

### 1. Variance Schedule 의 Empirical Tuning

Ho 2020 의 linear $[10^{-4}, 0.02]$ for $T = 1000$ 이 $32 \times 32$ image 에서 잘 작동. 더 큰 image (256+) 에서:
- Cosine schedule 우월 (Nichol 2021)
- $T$ 더 길게 (e.g., 2000)
- Adaptive schedule (training 중 update)

### 2. $T$ Steps 의 Choice

$T = 1000$ (Ho 2020) — 표준. 큰 $T$:
- 더 정확한 reverse approximation (small $\beta_t$ 가정)
- Sampling 더 느림
- Training 시 random $t$ sample → 분산

작은 $T$:
- Sampling 빠름
- Reverse approximation 부정확 → quality 손해

DDIM (Ch6-04) 이 sampling 가속, train $T = 1000$ + sample $T' = 50-100$ 가능.

### 3. Forward 가 Fixed 인 것의 의의

- VAE: encoder 학습 → posterior collapse 위험
- Diffusion: forward fixed → 안정적 supervision

이것이 Diffusion 의 fundamental advantage. 단, expressiveness 제약 — forward 가 Gaussian 만 가능. Modeling power 가 reverse 의 NN 에 집중.

---

## ⚖️ 가정과 한계

| 가정 | 한계 |
|------|------|
| Forward 가 Gaussian | 다른 분포 (예: stable, heavy-tailed) 불가 |
| Markov chain | 더 복잡한 dependency 없음 |
| Fixed $\beta_t$ schedule | Adaptive 가능 (research) |
| Reverse Gaussian approximation | $\beta_t$ small 가정 — 큰 step 에서 부정확 |
| $T$ steps 가 충분 | Sampling slow, DDIM 등으로 가속 |

---

## 📌 핵심 정리

$$\boxed{q(x_t | x_{t-1}) = \mathcal{N}(\sqrt{1-\beta_t} x_{t-1}, \beta_t I)}$$

$$\boxed{q(x_t | x_0) = \mathcal{N}(\sqrt{\bar\alpha_t} x_0, (1-\bar\alpha_t) I) \quad — \text{ closed form}}$$

$$\boxed{x_t = \sqrt{\bar\alpha_t} x_0 + \sqrt{1-\bar\alpha_t} \epsilon}$$

| Quantity | Formula |
|----------|---------|
| $\beta_t$ | Variance schedule |
| $\alpha_t$ | $1 - \beta_t$ |
| $\bar\alpha_t$ | $\prod_{s=1}^t \alpha_s$ |
| Forward mean | $\sqrt{1-\beta_t} x_{t-1}$ |
| Forward var | $\beta_t I$ |
| Closed-form mean | $\sqrt{\bar\alpha_t} x_0$ |
| Closed-form var | $(1-\bar\alpha_t) I$ |
| SNR(t) | $\bar\alpha_t / (1-\bar\alpha_t)$ |

---

## 🤔 생각해볼 문제

**문제 1** (기초): $T = 100, \beta_t = 0.01$ for all $t$ 일 때 $\bar\alpha_T$ 의 값을 계산하고, $x_T$ 의 분포를 derive 하라.

<details>
<summary>해설</summary>

$\alpha_t = 1 - \beta_t = 0.99$. $\bar\alpha_T = 0.99^{100} \approx 0.366$.

$x_T | x_0 \sim \mathcal{N}(\sqrt{0.366} x_0, 0.634 I) = \mathcal{N}(0.605 x_0, 0.634 I)$.

**시사점**: $\bar\alpha_T \approx 0.37$ 이므로 $x_T$ 가 still 약간 informative ($x_0$ 의 contribution). Pure noise ($\bar\alpha = 0$) 가 되려면 더 많은 steps 또는 큰 $\beta$.

**Linear schedule** $\beta \in [10^{-4}, 0.02]$, $T = 1000$: $\bar\alpha_T \approx \exp(-\sum \beta_t) \approx \exp(-10) \approx 4.5 \times 10^{-5}$. 거의 0 — pure noise.

</details>

**문제 2** (심화): Forward posterior $q(x_{t-1} | x_t, x_0)$ 의 mean $\tilde\mu$ 가 $x_t$ 와 $x_0$ 의 weighted average 임을 확인하라. 두 weight 의 합이 1 인가?

<details>
<summary>해설</summary>

$\tilde\mu_t(x_t, x_0) = \frac{\sqrt{\bar\alpha_{t-1}} \beta_t}{1 - \bar\alpha_t} x_0 + \frac{\sqrt{\alpha_t}(1 - \bar\alpha_{t-1})}{1 - \bar\alpha_t} x_t$

**Coefficient 합**:

$$\frac{\sqrt{\bar\alpha_{t-1}} \beta_t + \sqrt{\alpha_t}(1 - \bar\alpha_{t-1})}{1 - \bar\alpha_t}$$

이게 1 인가? 분자: $\sqrt{\bar\alpha_{t-1}} \beta_t + \sqrt{\alpha_t}(1 - \bar\alpha_{t-1})$. 일반적으로 $1 - \bar\alpha_t$ 와 같지 않음.

**왜 같지 않은가**: $x_0$ 와 $x_t$ 는 different scales. $x_t = \sqrt{\bar\alpha_t} x_0 + \sqrt{1 - \bar\alpha_t} \epsilon$ — $x_t$ 가 $x_0$ 의 scaled version + noise. Linear combination 의 weights 가 scales 을 보정.

**Intuition**:
- $t \to 0$: $\bar\alpha_t \to 1$, $\beta_t \to 0$. Coefficient of $x_t$ → 1, of $x_0$ → 0. ✓ ($x_{t-1} \approx x_t$ at small $t$)
- $t \to T$: $\bar\alpha_t \to 0$. Coefficient of $x_t$ → small, of $x_0$ → larger. (pure noise 에서 reconstruct 시 $x_0$ 의 영향 큼)

**시사점**: $\tilde\mu$ 가 affine combination of $x_t$ and $x_0$, but coefficients depend on schedule. 이 정확한 form 이 ELBO 의 KL 항 분석의 기반.

</details>

**문제 3** (논문 비평): Cosine vs Linear schedule 의 difference 를 SNR (signal-to-noise ratio) 측면에서 분석하라. 왜 large image 에서 cosine 이 우월한가?

<details>
<summary>해설</summary>

**Linear schedule**: $\beta_t \propto t$. $\bar\alpha_t = \prod_s (1 - \beta_s) \approx \exp(-\sum \beta_s) = \exp(-O(t^2 / T))$.

$\bar\alpha_t$ 가 매우 빠르게 감소 — $T/2$ 에서 이미 거의 0. SNR $= \bar\alpha_t / (1 - \bar\alpha_t) \to 0$ exponentially fast.

**Cosine schedule**: $\bar\alpha_t = \cos^2(\pi/2 \cdot t / T)$. $T/2$ 에서 $\bar\alpha = 0.5$ — gradual decrease. SNR 변화 부드러움.

**Large image 에서의 영향**:
1. **Information 분배**: linear 에서 후반 steps 가 noise-dominant — useful learning signal 적음
2. **Reverse approximation**: linear schedule 의 빠른 SNR drop 이 small $\beta$ 가정 깨뜨림
3. **High-frequency content**: large image 의 fine details 가 high SNR 에서 학습 — cosine 의 gradual schedule 이 더 informative

**Quantitative**:
- Nichol 2021 (Improved DDPM): cosine 으로 64×64 ImageNet FID 6.0 (linear: 8.5)
- 256×256 에서 차이 더 큼

**General principle**: forward schedule 이 "uniform information loss" 를 달성하면 reverse 학습이 efficient. Cosine 이 이에 더 가까움.

**Modern variants**:
- **Cosine** (Nichol 2021): 표준
- **Karras et al. 2022 EDM**: $\sigma$-based parameterization, optimal schedule 찾음
- **Continuous SDE** (Song 2021, Ch6-04): $\beta(t)$ 의 일반화
- **Adaptive**: training 중 schedule update

**시사점**: variance schedule 이 simple hyperparameter 같지만 model quality 에 큰 영향. Information theory 의 관점 (uniform info loss) 이 design principle 제공.

</details>

---

<div align="center">

[◀ 이전 (Ch5-06. StyleGAN)](../ch5-gan/06-stylegan.md) | [📚 README](../README.md) | [다음 ▶ (02. L_simple)](./02-ddpm-simple-loss.md)

</div>
