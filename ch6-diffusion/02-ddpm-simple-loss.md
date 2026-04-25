# 02. DDPM Loss 의 단순화 — $L_\text{simple}$

## 🎯 핵심 질문

- $\mu_\theta(x_t, t) = \frac{1}{\sqrt{\alpha_t}}\left(x_t - \frac{\beta_t}{\sqrt{1-\bar\alpha_t}}\epsilon_\theta\right)$ 의 parameterization 이 어떻게 ELBO 의 각 KL 을 noise prediction loss 로 환원하는가?
- ELBO 의 각 항 $L_{t-1}$ 이 $\frac{\beta_t^2}{2 \sigma_t^2 \alpha_t (1-\bar\alpha_t)} \mathbb{E}[\|\epsilon - \epsilon_\theta\|^2]$ 형태로 되는 정확한 derivation?
- $L_\text{simple} = \mathbb{E}_{t, x_0, \epsilon}[\|\epsilon - \epsilon_\theta(x_t, t)\|^2]$ — **가중치 무시** 가 왜 더 좋은 sample quality 를 주는가?
- "$\epsilon$ prediction" vs "$x_0$ prediction" vs "$v$ prediction" 의 차이?
- Simple loss 의 implicit weighting 이 어떤 frequency 또는 timestep 에 더 attention 을 주는가?

---

## 🔍 왜 $L_\text{simple}$ 이 결정적 발견인가

Ho 2020 의 핵심 contribution 은 forward/reverse 의 framework 보다는 **단순한 loss function** 의 발견:

$$L_\text{simple} = \mathbb{E}_{t, x_0, \epsilon}\left[\|\epsilon - \epsilon_\theta(x_t, t)\|^2\right]$$

**놀라운 점**: 이론적 ELBO 의 weighted version 이지만, 가중치를 **무시** ($w_t = 1$) 하는 것이 sample quality 면에서 우월. 이는:
1. **Simpler training** — 가중치 계산 불필요
2. **Better quality** — empirical FID 향상
3. **Implicit re-weighting** — high-noise step 에 더 attention

이 단순한 발견이 diffusion 의 mainstream 화의 결정적 trigger. 이 문서에서는 ELBO → $L_\text{simple}$ 의 정확한 derivation 과 weighting 의 implicit effect 를 다룹니다.

---

## 📐 수학적 선행 조건

- 이전 문서: 01-ddpm-forward-reverse.md
- [Probability Theory Deep Dive](https://github.com/iq-ai-lab/probability-theory-deep-dive): Gaussian KL
- Ch3-01 (ELBO)

---

## 📖 직관적 이해

### "noise 를 예측하면 reverse 가 따라온다"

DDPM 의 reverse step: $x_{t-1} \approx x_t - \text{noise}$. 즉 NN 이 "현재 noise 의 양" 을 예측하면 reverse 가 결정.

**Reparameterization**: $x_t = \sqrt{\bar\alpha_t} x_0 + \sqrt{1 - \bar\alpha_t} \epsilon$. 알려진 $x_t, t$ 에 대해 $\epsilon$ 만 알면 $x_0$ 도 알 수 있음:

$$x_0 = \frac{x_t - \sqrt{1 - \bar\alpha_t} \epsilon}{\sqrt{\bar\alpha_t}}$$

따라서 **NN 이 $\epsilon$ 만 예측하면 충분** — 다른 모든 quantity (mean, $x_0$ estimate) 가 derived.

### "왜 noise prediction 이 자연스러운가"

대안:
- **$x_0$ prediction**: NN 이 직접 clean image 예측. 가능하지만 small $t$ 에서 trivial ($x_t \approx x_0$), large $t$ 에서 매우 어려움.
- **$\mu$ prediction**: posterior mean 직접. 복잡한 formula.
- **$\epsilon$ prediction** (Ho 2020 의 선택): "noise 의 양" — 모든 $t$ 에서 well-defined task, simple loss.

$\epsilon$ prediction 이 **uniform difficulty** across timesteps — 학습 안정.

### "Weighting 무시 의 마법"

ELBO 의 정확한 형태:

$$L_\text{ELBO} = \sum_t \mathbb{E}\left[\frac{\beta_t^2}{2\sigma_t^2 \alpha_t (1 - \bar\alpha_t)} \|\epsilon - \epsilon_\theta\|^2\right] + \text{const}$$

**$L_\text{simple}$**: 가중치 $\frac{\beta_t^2}{...}$ 를 무시:

$$L_\text{simple} = \sum_t \mathbb{E}[\|\epsilon - \epsilon_\theta\|^2]$$

가중치 무시가 **better empirical quality** 인 이유:
- ELBO weight 가 small $t$ (high SNR) 에서 매우 큼 → small $t$ 에 너무 집중
- $L_\text{simple}$ 이 모든 $t$ 에 같은 attention → balanced
- High $t$ (large noise) 의 학습이 더 중요 (FID 측정)

---

## ✏️ 엄밀한 정의·정리

### 정리 2.1 — Reverse Mean Parameterization

$\mu_\theta(x_t, t)$ 를 다음으로 parameterize:

$$\mu_\theta(x_t, t) = \frac{1}{\sqrt{\alpha_t}}\left(x_t - \frac{\beta_t}{\sqrt{1 - \bar\alpha_t}} \epsilon_\theta(x_t, t)\right)$$

여기서 $\epsilon_\theta(x_t, t)$ 는 NN 의 output (predicted noise).

**Justification**: forward posterior $q(x_{t-1} | x_t, x_0)$ 의 ground-truth mean $\tilde\mu$:

$$\tilde\mu_t(x_t, x_0) = \frac{\sqrt{\bar\alpha_{t-1}} \beta_t}{1 - \bar\alpha_t} x_0 + \frac{\sqrt{\alpha_t}(1 - \bar\alpha_{t-1})}{1 - \bar\alpha_t} x_t$$

$x_0 = (x_t - \sqrt{1-\bar\alpha_t} \epsilon) / \sqrt{\bar\alpha_t}$ 대입:

$$\tilde\mu_t(x_t, \epsilon) = \frac{1}{\sqrt{\alpha_t}}\left(x_t - \frac{\beta_t}{\sqrt{1-\bar\alpha_t}} \epsilon\right)$$

따라서 NN parameterization 이 ground truth 와 같은 form, $\epsilon$ 만 학습.

### 정리 2.2 — ELBO 의 $\epsilon$-Form

위 parameterization 하에서, $L_{t-1}$ (ELBO 의 $t$-th term) 은:

$$L_{t-1} = \frac{\beta_t^2}{2 \sigma_t^2 \alpha_t (1 - \bar\alpha_t)} \cdot \mathbb{E}_{x_0, \epsilon}\left[\|\epsilon - \epsilon_\theta(\sqrt{\bar\alpha_t} x_0 + \sqrt{1-\bar\alpha_t}\epsilon, t)\|^2\right]$$

**증명**: $\text{KL}(q \| p_\theta)$ between two Gaussians $\mathcal{N}(\tilde\mu, \tilde\beta_t I)$ and $\mathcal{N}(\mu_\theta, \sigma_t^2 I)$ ($\sigma_t = \tilde\beta_t$ 또는 fixed):

$$\text{KL} = \frac{1}{2\sigma_t^2}\|\tilde\mu - \mu_\theta\|^2 + \text{const}$$

$\tilde\mu - \mu_\theta = \frac{\beta_t}{\sqrt{\alpha_t (1-\bar\alpha_t)}}(\epsilon - \epsilon_\theta)$ (above parameterization 으로부터).

대입:

$$\text{KL} = \frac{1}{2\sigma_t^2} \cdot \frac{\beta_t^2}{\alpha_t (1-\bar\alpha_t)} \|\epsilon - \epsilon_\theta\|^2 \quad \square$$

### 정의 2.3 — $L_\text{simple}$ (Ho 2020)

$$L_\text{simple} = \mathbb{E}_{t \sim \mathcal{U}(1, T), x_0 \sim p_d, \epsilon \sim \mathcal{N}(0, I)}\left[\|\epsilon - \epsilon_\theta(\sqrt{\bar\alpha_t} x_0 + \sqrt{1-\bar\alpha_t}\epsilon, t)\|^2\right]$$

ELBO 의 $L_{t-1}$ 와의 차이: **가중치 $\beta_t^2 / (2\sigma_t^2 \alpha_t (1-\bar\alpha_t))$ 무시 (= 1 로 set)**.

### 정리 2.4 — Implicit Weighting Comparison

ELBO weight $w_t^\text{ELBO} = \beta_t^2 / (2\sigma_t^2 \alpha_t (1-\bar\alpha_t))$.

Linear schedule 에서:
- Small $t$: $\beta_t$ small, $1 - \bar\alpha_t$ small → $w_t^\text{ELBO}$ 큼 (high-SNR steps 이 ELBO 에 dominate)
- Large $t$: $\beta_t$ moderate, $1 - \bar\alpha_t$ near 1 → $w_t^\text{ELBO}$ moderate

$L_\text{simple}$: $w_t = 1$ uniform — large $t$ 에 relatively 더 attention.

**Empirical**: $L_\text{simple}$ 이 더 좋은 FID. ELBO 가 더 좋은 NLL (likelihood-based).

### 정리 2.5 — 다른 Parameterizations

**$x_0$ prediction**: NN $f_\theta(x_t, t) = \hat x_0$. Loss: $\|x_0 - \hat x_0\|^2$. 동등 form (different scaling).

**$v$ prediction** (Salimans 2022 Progressive Distillation):

$$v = \sqrt{\bar\alpha_t} \epsilon - \sqrt{1 - \bar\alpha_t} x_0$$

$v$ 가 모든 $t$ 에서 well-scaled, distillation 에 유리.

**Score prediction**: $s_\theta(x_t, t) = \nabla_{x_t} \log q(x_t)$. NCSN (Ch6-03) 의 형태.

**관계**: 모두 equivalent (linear transformation 으로 변환), 학습 dynamics 만 다름.

### 정리 2.6 — Boundary Term $L_0$

$L_0 = -\mathbb{E}_q[\log p_\theta(x_0 | x_1)]$ — discrete pixel reconstruction.

**Ho 2020 의 처리**:

$p_\theta(x_0 | x_1)$: each pixel 의 256-bin probability via discretized Gaussian:

$$p_\theta(x_0 | x_1) = \prod_{i=1}^d \int_{x_0^i - 1/255}^{x_0^i + 1/255} \mathcal{N}(z; \mu_\theta^i(x_1, 1), \sigma^2) dz$$

이는 small contribution to total loss, $L_\text{simple}$ 의 $t = 1$ term 으로 absorb.

---

## 🔬 증명 및 수학적 유도

### 유도 1 — Parameterization 의 Algebra

Ground truth $\tilde\mu(x_t, x_0)$:

$$\tilde\mu = \frac{\sqrt{\bar\alpha_{t-1}} \beta_t}{1 - \bar\alpha_t} x_0 + \frac{\sqrt{\alpha_t}(1 - \bar\alpha_{t-1})}{1 - \bar\alpha_t} x_t$$

$x_0 = \frac{x_t - \sqrt{1 - \bar\alpha_t} \epsilon}{\sqrt{\bar\alpha_t}}$ 대입:

$$\tilde\mu = \frac{\sqrt{\bar\alpha_{t-1}} \beta_t}{(1 - \bar\alpha_t) \sqrt{\bar\alpha_t}}(x_t - \sqrt{1-\bar\alpha_t}\epsilon) + \frac{\sqrt{\alpha_t}(1 - \bar\alpha_{t-1})}{1 - \bar\alpha_t} x_t$$

Combine $x_t$ coefficients:

$$\frac{\sqrt{\bar\alpha_{t-1}} \beta_t}{(1-\bar\alpha_t)\sqrt{\bar\alpha_t}} + \frac{\sqrt{\alpha_t}(1-\bar\alpha_{t-1})}{1-\bar\alpha_t}$$

$\sqrt{\bar\alpha_t} = \sqrt{\alpha_t \bar\alpha_{t-1}} = \sqrt{\alpha_t} \sqrt{\bar\alpha_{t-1}}$:

$$= \frac{\beta_t}{(1-\bar\alpha_t) \sqrt{\alpha_t}} + \frac{\sqrt{\alpha_t}(1-\bar\alpha_{t-1})}{1-\bar\alpha_t}$$

$$= \frac{1}{\sqrt{\alpha_t}(1-\bar\alpha_t)}[\beta_t + \alpha_t(1 - \bar\alpha_{t-1})] = \frac{1}{\sqrt{\alpha_t}(1-\bar\alpha_t)}[\beta_t + \alpha_t - \bar\alpha_t]$$

$\alpha_t = 1 - \beta_t$:

$$= \frac{1}{\sqrt{\alpha_t}(1-\bar\alpha_t)}[(1-\beta_t) + \beta_t - \bar\alpha_t] = \frac{1 - \bar\alpha_t}{\sqrt{\alpha_t}(1-\bar\alpha_t)} = \frac{1}{\sqrt{\alpha_t}}$$

Combine $\epsilon$ coefficient:

$$-\frac{\sqrt{\bar\alpha_{t-1}} \beta_t \sqrt{1-\bar\alpha_t}}{(1-\bar\alpha_t)\sqrt{\bar\alpha_t}} = -\frac{\beta_t}{\sqrt{\alpha_t}\sqrt{1-\bar\alpha_t}}$$

따라서:

$$\tilde\mu = \frac{1}{\sqrt{\alpha_t}}\left(x_t - \frac{\beta_t}{\sqrt{1-\bar\alpha_t}}\epsilon\right) \quad \square$$

이게 NN parameterization $\mu_\theta$ 의 ideal form.

### 유도 2 — KL Loss → $\epsilon$ MSE

$\text{KL}(\mathcal{N}(\tilde\mu, \tilde\beta_t I) \| \mathcal{N}(\mu_\theta, \sigma_t^2 I))$:

Two Gaussians 의 closed-form KL (same dim):

$$\text{KL} = \frac{1}{2}\left[\frac{\tilde\beta_t}{\sigma_t^2} + \frac{\|\tilde\mu - \mu_\theta\|^2}{\sigma_t^2} - 1 - \log \frac{\tilde\beta_t}{\sigma_t^2}\right] \cdot d$$

(d = dimension of $x$). $\sigma_t^2 = \tilde\beta_t$ (Ho 2020 choice) 면 first and last terms cancel:

$$\text{KL} = \frac{1}{2\sigma_t^2}\|\tilde\mu - \mu_\theta\|^2$$

$\tilde\mu - \mu_\theta = \frac{1}{\sqrt{\alpha_t}}\left(\frac{\beta_t}{\sqrt{1-\bar\alpha_t}}(\epsilon - \epsilon_\theta)\right) - 0 = \frac{\beta_t}{\sqrt{\alpha_t}\sqrt{1-\bar\alpha_t}}(\epsilon - \epsilon_\theta)$

$$\|\tilde\mu - \mu_\theta\|^2 = \frac{\beta_t^2}{\alpha_t (1-\bar\alpha_t)} \|\epsilon - \epsilon_\theta\|^2$$

따라서:

$$\text{KL} = \frac{\beta_t^2}{2 \sigma_t^2 \alpha_t (1-\bar\alpha_t)} \|\epsilon - \epsilon_\theta\|^2$$

= $L_{t-1}$. $\square$

### 유도 3 — ELBO weight 의 Behavior

$\sigma_t^2 = \tilde\beta_t = \frac{1 - \bar\alpha_{t-1}}{1 - \bar\alpha_t} \beta_t$:

$$w_t^\text{ELBO} = \frac{\beta_t^2}{2 \cdot \frac{1-\bar\alpha_{t-1}}{1-\bar\alpha_t} \beta_t \cdot \alpha_t (1-\bar\alpha_t)} = \frac{\beta_t}{2 (1-\bar\alpha_{t-1}) \alpha_t}$$

**Small $t$**: $\beta_t$ small, $1 - \bar\alpha_{t-1}$ small (still close to data), $\alpha_t$ near 1. Ratio: $\beta_t / (1-\bar\alpha_{t-1})$ 이 $O(1)$ (둘 다 small).

**Large $t$**: $\beta_t$ moderate, $1 - \bar\alpha_{t-1}$ near 1, $\alpha_t$ near 1. $w \approx \beta_t / 2$ — moderate.

**Empirical**: linear schedule 에서 $w_t^\text{ELBO}$ 가 $t$ 에 따라 uniform 하지 않음 — small $t$ 에 약간 weighted.

### 유도 4 — $L_\text{simple}$ 의 implicit emphasis

$L_\text{simple}$ vs ELBO: 같은 $\|\epsilon - \epsilon_\theta\|^2$ 이지만 weight 다름.

**Average across $t$**: 균등 sampling $t \sim \mathcal{U}(1, T)$.

**ELBO 적용 시**: each $t$ 의 contribution 이 $w_t \cdot \mathbb{E}[\|\epsilon - \epsilon_\theta\|^2]$. Small $t$ 의 $w$ 가 큼 → small $t$ 의 학습 bias.

**$L_\text{simple}$**: equal contribution from all $t$ → high-noise (large $t$) step 도 동등 학습.

**왜 이게 좋은가** (Ho 2020 의 hypothesis):
- High-noise step 의 정확한 denoising 이 perceptual quality (FID) 의 핵심
- Low-noise step 의 fine detail 은 over-fit 위험
- $L_\text{simple}$ 이 balanced training

### 유도 5 — $v$-Prediction 의 Stability

$v = \sqrt{\bar\alpha_t} \epsilon - \sqrt{1-\bar\alpha_t} x_0$.

**Properties**:
- $\|v\|^2 = \bar\alpha_t \|\epsilon\|^2 + (1-\bar\alpha_t) \|x_0\|^2 - 2\sqrt{\bar\alpha_t(1-\bar\alpha_t)} \epsilon \cdot x_0$
- $\mathbb{E}[\|v\|^2] = \bar\alpha_t \cdot d + (1-\bar\alpha_t) \cdot d = d$ (assuming $\|x_0\|^2 = d$, $\|\epsilon\|^2 = d$)

따라서 $v$ 의 norm 이 $t$-independent → uniform difficulty across timesteps.

**$\epsilon$ vs $v$**:
- $\epsilon$: large $t$ 에서 well-defined ($x_t \approx \epsilon$), small $t$ 에서 hard ($\epsilon$ buried in signal)
- $v$: all $t$ 에서 well-balanced

**Modern usage**: Stable Diffusion v2, SD3, Imagen 등에서 $v$-prediction. Distillation (Progressive Distillation, Consistency Model) 에 특히 우월.

---

## 💻 실험 재현 / PyTorch 구현

### 실험 1 — DDPM Training with $L_\text{simple}$

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms

class SimpleEpsNet(nn.Module):
    """U-Net-like for image denoising"""
    def __init__(self, in_ch=1, hidden=64):
        super().__init__()
        self.t_emb = nn.Embedding(1000, hidden)
        self.down1 = nn.Conv2d(in_ch, hidden, 3, padding=1)
        self.down2 = nn.Conv2d(hidden, hidden, 3, stride=2, padding=1)
        self.mid = nn.Conv2d(hidden, hidden, 3, padding=1)
        self.up1 = nn.ConvTranspose2d(hidden, hidden, 4, stride=2, padding=1)
        self.up2 = nn.Conv2d(hidden, in_ch, 3, padding=1)

    def forward(self, x, t):
        t_e = self.t_emb(t)[:, :, None, None]
        h = F.silu(self.down1(x) + t_e)
        h = F.silu(self.down2(h) + t_e)
        h = F.silu(self.mid(h) + t_e)
        h = F.silu(self.up1(h))
        return self.up2(h)

# DDPM forward (closed form)
T = 1000
betas = torch.linspace(1e-4, 0.02, T)
alphas = 1 - betas
alpha_bars = torch.cumprod(alphas, dim=0)

def q_sample(x_0, t, noise=None):
    if noise is None:
        noise = torch.randn_like(x_0)
    sqrt_ab = alpha_bars[t][:, None, None, None].sqrt()
    sqrt_1mab = (1 - alpha_bars[t])[:, None, None, None].sqrt()
    return sqrt_ab * x_0 + sqrt_1mab * noise

# Training loop with L_simple
loader = torch.utils.data.DataLoader(
    datasets.MNIST('~/data', download=True,
                   transform=transforms.ToTensor()),
    batch_size=128, shuffle=True
)
model = SimpleEpsNet().cuda()
opt = torch.optim.Adam(model.parameters(), lr=2e-4)

for epoch in range(20):
    for x, _ in loader:
        x = x.cuda() * 2 - 1   # [-1, 1]
        bsz = x.size(0)
        t = torch.randint(0, T, (bsz,), device=x.device)
        noise = torch.randn_like(x)
        x_t = q_sample(x, t, noise)
        eps_pred = model(x_t, t)
        loss = F.mse_loss(eps_pred, noise)   # L_simple
        opt.zero_grad(); loss.backward(); opt.step()
    print(f"Epoch {epoch}: loss = {loss.item():.4f}")
```

### 실험 2 — Sampling

```python
@torch.no_grad()
def ddpm_sample(model, shape, T_steps=T):
    x = torch.randn(shape).cuda()
    for t in reversed(range(T_steps)):
        t_tensor = torch.tensor([t]).expand(x.size(0)).cuda()
        eps_pred = model(x, t_tensor)
        coef1 = 1 / alphas[t].sqrt().cuda()
        coef2 = (betas[t] / (1 - alpha_bars[t]).sqrt()).cuda()
        mu = coef1 * (x - coef2 * eps_pred)
        if t > 0:
            sigma = betas[t].sqrt().cuda()
            x = mu + sigma * torch.randn_like(x)
        else:
            x = mu
    return x

samples = ddpm_sample(model, (16, 1, 28, 28))
# Visualize: 16 generated MNIST digits
```

### 실험 3 — $L_\text{simple}$ vs $L_\text{ELBO}$ 비교

```python
# Modified loss with ELBO weighting
def compute_elbo_weight(t):
    return betas[t]**2 / (2 * (1 - alpha_bars[t-1] if t > 0 else 1) * alphas[t])

# 같은 모델 두 번 학습: L_simple vs weighted (ELBO)
# Compute final FID 비교
# Empirical (Ho 2020): L_simple 이 약 30% 낮은 FID
```

### 실험 4 — $\epsilon$-prediction vs $x_0$-prediction

```python
class X0Net(SimpleEpsNet):
    """Same arch, predicts x_0 instead"""
    pass   # output 의 의미만 다름

def loss_x0_pred(model, x_0, t):
    noise = torch.randn_like(x_0)
    x_t = q_sample(x_0, t, noise)
    x_0_pred = model(x_t, t)
    return F.mse_loss(x_0_pred, x_0)

# Empirical 비교: 두 parameterization 의 학습 dynamics
# Ho 2020: ε-prediction 이 약간 우월, but very close
```

---

## 🔗 이론과 실전의 간극

### 1. $L_\text{simple}$ 의 Empirical Superiority

$L_\text{simple}$ 가 ELBO 의 lower bound 를 maximize 안 함 (가중치 무시) — 이론적으로 sub-optimal.

**그러나 sample quality 우월**: FID 가 실제로 유의미한 quality measure → $L_\text{simple}$ 이 이를 더 잘 align.

**시사점**: 항상 "이론적 optimum" 이 best practical objective 가 아님.

### 2. $v$-Prediction 의 부상

Modern diffusion 의 표준이 $\epsilon$-prediction → $v$-prediction 으로 이동:
- Stable Diffusion v2: $v$-prediction (이미 transition)
- Imagen, SD3: $v$ 또는 flow matching
- Distillation 에 우월 (Salimans 2022, Song 2023)

이는 NN 의 학습 anchor (norm 이 일정) 의 효과.

### 3. Karras et al. 2022 의 EDM Framework

"Elucidating the Design Space of Diffusion-Based Generative Models" — diffusion 의 모든 design choices (parameterization, noise schedule, sampling) 의 unified analysis.

핵심 통찰: "$\sigma$-based" parameterization (noise std 직접) 가 most natural. $\epsilon$, $v$, $x_0$, score 등이 specific instance.

이로 더 좋은 schedule (Karras schedule) 와 sampling (Heun's method) 도출.

---

## ⚖️ 가정과 한계

| 가정 | 한계 |
|------|------|
| $\epsilon$ prediction 이 best | $v$-prediction 이 distillation 에 우월 |
| $L_\text{simple}$ 이 항상 우월 | NLL critical 응용에서는 ELBO 우월 |
| Linear/Cosine schedule 가 적절 | Karras schedule (Karras 2022) 가 우월 |
| Forward 가 fixed | Flow Matching (Lipman 2023) 등 다른 process |
| $\sigma_t$ fixed | Learned $\Sigma_\theta$ (Improved DDPM) 가 better NLL |

---

## 📌 핵심 정리

$$\boxed{L_\text{simple} = \mathbb{E}_{t, x_0, \epsilon}\left[\|\epsilon - \epsilon_\theta(\sqrt{\bar\alpha_t} x_0 + \sqrt{1-\bar\alpha_t}\epsilon, t)\|^2\right]}$$

$$\boxed{\mu_\theta(x_t, t) = \frac{1}{\sqrt{\alpha_t}}\left(x_t - \frac{\beta_t}{\sqrt{1-\bar\alpha_t}}\epsilon_\theta(x_t, t)\right)}$$

| Parameterization | Output | Pros |
|------------------|--------|------|
| **$\epsilon$ prediction** (Ho 2020) | $\epsilon_\theta(x_t, t)$ | Natural, simple |
| **$x_0$ prediction** | $\hat x_0(x_t, t)$ | Direct target |
| **$v$ prediction** (Salimans 2022) | $v = \sqrt{\bar\alpha_t}\epsilon - \sqrt{1-\bar\alpha_t}x_0$ | $t$-uniform norm, distillation |
| **Score prediction** (NCSN) | $s_\theta = \nabla_x \log p_t$ | Connects to score matching |

| Loss | Weight | Best for |
|------|--------|----------|
| **$L_\text{simple}$** (Ho) | $w_t = 1$ | FID (sample quality) |
| **$L_\text{ELBO}$** | $w_t = \beta_t^2 / (...)$ | NLL (likelihood) |
| **EDM** (Karras 2022) | $\sigma$-aware | Both |

---

## 🤔 생각해볼 문제

**문제 1** (기초): $L_\text{simple}$ 의 weight 가 $w_t = 1$ vs ELBO 의 $w_t^\text{ELBO}$ 가 $\beta_t^2 / (2\sigma_t^2 \alpha_t (1-\bar\alpha_t))$ 인 것의 차이 — small $t$ 에서 어떤 weighting 이 더 큰가?

<details>
<summary>해설</summary>

$\sigma_t^2 = \tilde\beta_t = \frac{1-\bar\alpha_{t-1}}{1-\bar\alpha_t}\beta_t$ 대입:

$$w_t^\text{ELBO} = \frac{\beta_t^2 (1-\bar\alpha_t)}{2 (1-\bar\alpha_{t-1}) \beta_t \alpha_t (1-\bar\alpha_t)} = \frac{\beta_t}{2 (1-\bar\alpha_{t-1}) \alpha_t}$$

**Small $t$**: $\beta_t$ small (e.g., $10^{-4}$), $\alpha_t \approx 1$, $1 - \bar\alpha_{t-1}$ small.

For linear schedule with $T = 1000$:
- $t = 1$: $\beta_1 = 10^{-4}$, $\bar\alpha_0 = 1$, $1 - \bar\alpha_0 = 0$ → $w \to \infty$ 위험!

이건 numerical issue — actually $L_0$ (boundary term) 이 따로 처리.

For $t = 10$: $\beta_{10} \approx 2 \times 10^{-4}$, $1 - \bar\alpha_9 \approx 10^{-3}$ → $w \approx 0.1$.

For $t = 500$: $\beta_{500} \approx 0.01$, $1 - \bar\alpha_{499} \approx 0.999$ → $w \approx 0.005$.

For $t = 1000$: $\beta_{1000} = 0.02$, $1 - \bar\alpha_{999} \approx 1$ → $w \approx 0.01$.

**Pattern**: ELBO weight 가 small $t$ 에서 큼 (numerical limit), large $t$ 에서 작음.

$L_\text{simple}$: $w_t = 1$ uniformly. Small $t$ 의 영향 줄임, large $t$ 의 영향 키움.

**Empirical 의미**: $L_\text{simple}$ 가 large $t$ (high noise) 의 학습에 더 attention — perceptual quality (FID) 의 핵심.

</details>

**문제 2** (심화): $v$-prediction 의 norm 이 $t$-independent 임을 증명하라. $\epsilon$-prediction 과 비교해 학습 dynamics 의 stability advantage 는?

<details>
<summary>해설</summary>

**$v$ definition**: $v = \sqrt{\bar\alpha_t}\epsilon - \sqrt{1-\bar\alpha_t}x_0$.

$\|v\|^2 = \bar\alpha_t \|\epsilon\|^2 + (1-\bar\alpha_t) \|x_0\|^2 - 2\sqrt{\bar\alpha_t(1-\bar\alpha_t)} \epsilon \cdot x_0$

**Expectation under $\epsilon \sim \mathcal{N}(0, I)$ (independent of $x_0$)**:
- $\mathbb{E}[\epsilon \cdot x_0] = 0$
- $\mathbb{E}[\|\epsilon\|^2] = d$
- Assume $\mathbb{E}[\|x_0\|^2] = d$ (data normalized)

$\mathbb{E}[\|v\|^2] = \bar\alpha_t \cdot d + (1-\bar\alpha_t) \cdot d = d$

**$t$-independent**! Uniform norm across all timesteps.

**Comparison with $\epsilon$**: $\mathbb{E}[\|\epsilon\|^2] = d$ — also $t$-independent (since $\epsilon$ is the target).

**Hmm — both seem $t$-uniform**. But the **NN's task difficulty** differs:
- $\epsilon$-prediction: at small $t$, $x_t \approx x_0$, NN must extract small $\epsilon$ component from large signal — **hard**
- $v$-prediction: at small $t$, $v \approx -x_0$ (close to data, easy). At large $t$, $v \approx \epsilon$ (close to noise, also easy).

**$x_0$-prediction**: at small $t$ easy ($\hat x_0 = x_t$), at large $t$ hard ($x_t \approx$ noise).

**Stability**: $v$-prediction 의 task difficulty 가 **uniformly moderate** across all $t$ — most stable training. $\epsilon$ 와 $x_0$ 는 각각 한 쪽에서 어려움.

**Distillation**: Progressive Distillation (Salimans 2022), Consistency Model (Song 2023) 에서 $v$-prediction 이 student-teacher distillation 의 stable target. Modern SOTA models (SD3, Imagen) 이 이를 채택.

**시사점**: parameterization 의 선택이 단순 algebraic equivalence 가 아닌 **학습 dynamics 와 distillation friendliness** 의 결정. Theoretical equivalence ≠ practical equivalence.

</details>

**문제 3** (논문 비평): Ho 2020 이 $L_\text{simple}$ 의 우월성을 발견한 후 후속 작업들이 이를 다양한 방향으로 generalize. EDM (Karras 2022) 의 $\sigma$-aware framework 가 어떻게 모든 design choices 를 통합하는지 설명하라.

<details>
<summary>해설</summary>

**EDM (Elucidating the Design Space)** 의 핵심: diffusion 의 모든 변형이 **$\sigma$-parameterization** 의 instance.

**Unified framework**:
- $\sigma_t$ = noise std at time $t$ (continuous parameterization)
- Forward: $x_t = x_0 + \sigma_t \epsilon$ (additive noise)
- 모든 quantity 가 $\sigma$ 의 함수

**Model parameterization**:

$$D_\theta(x_t; \sigma) = c_\text{skip}(\sigma) x_t + c_\text{out}(\sigma) F_\theta(c_\text{in}(\sigma) x_t; c_\text{noise}(\sigma))$$

$F_\theta$: NN. $c_*$: $\sigma$-dependent scalings.

**Choices** (DDPM, $v$, $x_0$, score) 가 각각 specific choice of $c_*$:
- $\epsilon$-prediction (DDPM): specific $c_\text{out}, c_\text{skip}$
- $v$-prediction: different scaling
- Score: different connection

**EDM 의 권고 ("good defaults")**:
- $c_\text{skip} = \sigma_d^2 / (\sigma^2 + \sigma_d^2)$
- $c_\text{out} = \sigma \sigma_d / \sqrt{\sigma^2 + \sigma_d^2}$
- $c_\text{in} = 1 / \sqrt{\sigma^2 + \sigma_d^2}$
- $c_\text{noise} = \log \sigma / 4$

이는 **$F_\theta$ 의 input/output 이 unit-variance** — 모든 $\sigma$ 에서 NN 의 task 가 normalized.

**Schedule**: log-linear in $\sigma$ — uniform info loss.

**Sampling**: Heun's method — 2nd-order ODE solver, fewer steps for same quality.

**Empirical** (EDM, Karras 2022):
- CIFAR-10 unconditional FID: 1.79 (record at the time)
- ImageNet 64×64: 1.36
- 같은 model arch 에서 design choices 만 변경

**시사점**:
1. **Theoretical unification**: $\sigma$-framework 가 all variants 통합
2. **Empirical superiority**: principled design > ad-hoc 선택
3. **Modular**: parameterization, schedule, sampling 을 separate concerns 로 분석
4. **Modern foundation**: 후속 모든 diffusion model 이 EDM 의 insights 활용 (SD3, Imagen 2, etc.)

**시사점**: $L_\text{simple}$ 같은 empirical 발견이 후에 deeper theoretical framework 으로 연결. **Empirical-first, then theoretical** 의 ML research 의 일반 pattern. 이론과 실전의 dialogue 가 progress 의 driver.

</details>

---

<div align="center">

[◀ 이전 (01. DDPM)](./01-ddpm-forward-reverse.md) | [📚 README](../README.md) | [다음 ▶ (03. Score-Based)](./03-score-based-ncsn.md)

</div>
