# 04. Autoregressive Flow — MAF · IAF

## 🎯 핵심 질문

- MAF 의 $x_i = \mu_i(x_{<i}) + \sigma_i(x_{<i}) \cdot z_i$ 와 IAF 의 $z_i = \mu_i(z_{<i}) + \sigma_i(z_{<i}) \cdot x_i$ 의 정확한 차이는?
- MAF 는 density evaluation 빠르고 sampling 느림, IAF 는 반대 — 왜 이런 비대칭이 생기는가?
- 두 flow 가 같은 family 의 분포를 표현 가능한가? (Yes — 서로의 inverse)
- Masked Autoencoder (MADE) 가 MAF 의 효율적 구현에 어떻게 사용되는가?
- VAE 의 amortized posterior 로 IAF 를 사용하는 motivation 은?

---

## 🔍 왜 MAF/IAF 이중성이 결정적인가

Coupling layer (RealNVP, Glow) 와 다른 flow architecture: **autoregressive flow**. 한 transform 에서 모든 dimension 이 변형 (coupling 의 절반-절반 vs 전체).

두 dual variants:
- **MAF** (Masked Autoregressive Flow, Papamakarios 2017): density evaluation 빠름, sampling 느림
- **IAF** (Inverse Autoregressive Flow, Kingma 2016): sampling 빠름, density 느림

**Duality**: MAF 와 IAF 는 서로의 inverse — 같은 transformation, 다른 방향에서 효율. 이 이중성이 사용 사례를 분리:

- **MAF**: density estimation (likelihood-critical 응용 — anomaly detection)
- **IAF**: variational inference (VAE 의 expressive posterior), Parallel WaveNet 의 student

이 문서에서는 두 flow 의 수학과 architectural 함의, 그리고 MADE-style masking 의 efficient 구현을 다룹니다.

---

## 📐 수학적 선행 조건

- 이전 문서들: 01-change-of-variables, 02-realnvp-coupling
- Ch2: AR factorization (chain rule)
- [Probability Theory Deep Dive](https://github.com/iq-ai-lab/probability-theory-deep-dive): Conditional density

---

## 📖 직관적 이해

### "Coupling 의 일반화 — 모든 dim 이 dependent"

Coupling: 절반 ($x_A$) unchanged, 절반 ($x_B$) 가 $x_A$ 의존.

**Autoregressive**: 각 $x_i$ 가 $x_{<i}$ 의존 — 모든 dim 이 다른 dim 의존, 단 sequential ordering 으로.

이는 chain rule $p(x) = \prod p(x_i | x_{<i})$ 의 직접 구현. PixelCNN, GPT 와 동일한 idea.

### MAF: Density First

MAF 는 **base distribution 으로 가는 방향** 이 autoregressive:

$$z_i = (x_i - \mu_i(x_{<i})) / \sigma_i(x_{<i})$$

또는 동등:

$$x_i = \mu_i(x_{<i}) + \sigma_i(x_{<i}) \cdot z_i$$

**Density evaluation**: $z = f^{-1}(x)$ 를 한 번에 계산. 모든 $\mu_i, \sigma_i$ 가 $x_{<i}$ 의존이지만 **$x$ 가 다 알려져 있으므로 parallel** (mask 사용).

**Sampling**: $z \sim \mathcal{N}$, 각 $x_i$ 를 sequential 로 $\mu_i(x_{<i}), \sigma_i(x_{<i})$ 계산 후 sample. **$O(D)$ sequential**.

### IAF: Sampling First

IAF 는 반대 — **base 에서 데이터로 가는 방향** 이 autoregressive:

$$x_i = \mu_i(z_{<i}) + \sigma_i(z_{<i}) \cdot z_i$$

**Sampling**: $z \sim \mathcal{N}$, 모든 $\mu_i, \sigma_i$ 가 $z_{<i}$ 의존 — **$z$ 가 다 알려져 있으므로 parallel**. 한 번에 $x$ 계산.

**Density evaluation**: $z = f^{-1}(x)$ — $z_i$ 가 $x_{<i}, z_{<i}$ 의존 (sequential). **$O(D)$ sequential**.

### Duality 의 이유

MAF: $x_i = $ ($x_{<i}$ 의존) → density 시 $x$ 다 있음, sampling 시 $x$ sequential
IAF: $x_i = $ ($z_{<i}$ 의존) → sampling 시 $z$ 다 있음, density 시 $x \to z$ sequential

**Duality**: MAF 의 forward $z = f^{-1}(x)$ = IAF 의 forward $x = g(z)$ — 같은 $\mu, \sigma$ 가 다른 입력에 의존. 두 flow 는 **서로의 inverse**.

---

## ✏️ 엄밀한 정의·정리

### 정의 4.1 — Masked Autoregressive Flow (MAF)

NN $\mu_i, \sigma_i: \mathbb{R}^{i-1} \to \mathbb{R}$ ($\sigma_i > 0$, e.g., $\sigma_i = \exp(\hat\sigma_i)$).

**Forward (sampling, $z \to x$)**:
$$x_i = \mu_i(x_{<i}) + \sigma_i(x_{<i}) \cdot z_i, \quad i = 1, \ldots, D$$

(Wait — MAF 의 정의는 $x \to z$ 가 forward 인 경우가 더 자연. 명확화):

**Density direction**: $z_i = (x_i - \mu_i(x_{<i})) / \sigma_i(x_{<i})$.

이를 forward 로 정의하면:
- Forward (density): $z = f^{-1}_{\text{flow}}(x)$ — parallel, all $\mu_i, \sigma_i$ computed at once
- Inverse (sampling): $x = f_{\text{flow}}(z)$ — sequential

### 정의 4.2 — Inverse Autoregressive Flow (IAF)

**Forward (sampling, $z \to x$)**:
$$x_i = \mu_i(z_{<i}) + \sigma_i(z_{<i}) \cdot z_i$$

- Forward (sampling): $x = f_{\text{flow}}(z)$ — parallel, all $\mu_i, \sigma_i$ computed at once
- Inverse (density): sequential, $x \to z$

### 정리 4.3 — Jacobian 의 Triangular 구조

MAF density direction $z = f^{-1}(x)$:

$$\partial z_i / \partial x_j = \begin{cases} 0 & \text{if } j > i \\ -\mu_i'(x_{<i}) / \sigma_i + (x_i - \mu_i)/\sigma_i^2 \cdot \sigma_i'(x_{<i}) & \text{if } j < i \\ 1 / \sigma_i(x_{<i}) & \text{if } j = i \end{cases}$$

**Lower triangular** with diagonal $1/\sigma_i$.

$$\det J_{f^{-1}} = \prod_{i=1}^D \frac{1}{\sigma_i(x_{<i})}$$

$$\log p_X(x) = \log p_Z(z) + \log |\det J_{f^{-1}}| = \log p_Z(z) - \sum_{i=1}^D \log \sigma_i(x_{<i})$$

### 정리 4.4 — MAF 와 IAF 의 Inverse 관계

MAF 의 forward (sampling): $x_i = \mu_i^{\text{MAF}}(x_{<i}) + \sigma_i^{\text{MAF}}(x_{<i}) z_i$
IAF 의 forward (sampling): $x_i = \mu_i^{\text{IAF}}(z_{<i}) + \sigma_i^{\text{IAF}}(z_{<i}) z_i$

**Duality**: MAF 의 inverse direction (sampling) 이 IAF 의 forward direction. 같은 $\mu, \sigma$ family 이지만 conditioning 변수 가 다름 — $x$ vs $z$.

따라서 **표현력** (어떤 분포를 표현할 수 있는가) 은 같지만, **계산 효율** 이 sample/density 방향에 따라 다름.

### 정의 4.5 — MADE (Masked AutoEncoder for Density Estimation)

MAF 의 효율적 구현. NN 의 weight 에 mask 적용하여 각 output $i$ 가 input $1, \ldots, i-1$ 만 보게 함:

- 각 hidden unit $h_k$ 에 random integer $m_k \in [1, D-1]$ 할당
- Layer $l$ to $l+1$ weight: $W_{j,k}^{(l)} = 0$ if $m_k \geq m_j$
- Output $i$ to hidden: $W_{i, k}^{(\text{out})} = 0$ if $m_k \geq i$
- Input $j$ to hidden: $W_{k, j}^{(\text{in})} = 0$ if $m_k < j$

이 mask 패턴이 **autoregressive constraint** 를 자동으로 강제. PyTorch/TF 에서 단일 forward pass 로 모든 $\mu_i, \sigma_i$ 계산.

### 정리 4.6 — IAF in VAE

VAE 의 posterior $q_\phi(z|x) = \mathcal{N}(\mu_\phi(x), \sigma_\phi^2(x) I)$ — factored Gaussian. 한계: factored 의 표현력 부족.

**IAF posterior** (Kingma 2016):
$$z_0 \sim \mathcal{N}(0, I), \quad z_i = \mu_i(z_{<i}) + \sigma_i(z_{<i}) z_i$$

IAF 의 sampling 이 parallel — **VAE forward 시 빠름**. Density evaluation 도 학습 시 알려진 $z$ 위에서 single forward pass 로 가능.

**효과**: VAE 의 posterior 가 더 expressive → ELBO tighter → likelihood 개선.

---

## 🔬 증명 및 수학적 유도

### 유도 1 — MAF 의 Density 가 Parallel 인 이유

$z_i = (x_i - \mu_i(x_{<i})) / \sigma_i(x_{<i})$.

모든 $i$ 의 $\mu_i, \sigma_i$ 가 $x_{<i}$ 의존 — **모든 $x$ 가 알려져 있으면 한 번에 계산 가능** (MADE 의 single forward pass).

각 $i$ 의 conditional 이 다른 NN 일 필요 없음 — 같은 NN 의 different output, mask 로 dependency 강제.

### 유도 2 — MAF 의 Sampling 이 Sequential 인 이유

$x_i = \mu_i(x_{<i}) + \sigma_i(x_{<i}) z_i$.

$x_1$ 계산 위해: $\mu_1, \sigma_1$ (no dependency). OK.
$x_2$ 계산 위해: $\mu_2(x_1), \sigma_2(x_1)$. $x_1$ 이 알려진 후에야 가능.
$x_3$ 계산 위해: $\mu_3(x_1, x_2), \sigma_3(x_1, x_2)$. ...

각 $x_i$ 가 이전 $x_{<i}$ 가 알려져야 — **$D$ 번 sequential**. 비효율.

### 유도 3 — IAF 의 Sampling 이 Parallel 인 이유

$x_i = \mu_i(z_{<i}) + \sigma_i(z_{<i}) z_i$.

$z = (z_1, \ldots, z_D) \sim \mathcal{N}$ — 한 번에 sampling.

모든 $i$ 의 $\mu_i, \sigma_i$ 가 $z_{<i}$ 의존 — **$z$ 가 다 있으면 single MADE forward pass 로 모든 $\mu_i, \sigma_i$**. 그 후 $x_i = \mu_i + \sigma_i z_i$ — element-wise, parallel.

### 유도 4 — IAF 의 Density 가 Sequential 인 이유

$z = f^{-1}_{\text{IAF}}(x)$:

$z_1 = (x_1 - \mu_1) / \sigma_1$ (no dependency).
$z_2 = (x_2 - \mu_2(z_1)) / \sigma_2(z_1)$ — $z_1$ 알아야.
...

각 $z_i$ 가 이전 $z_{<i}$ 알아야 — **$D$ 번 sequential**.

### 유도 5 — Duality 의 정확한 의미

**MAF in density direction**: $f_{\text{MAF}}: x \to z$ (parallel via $\mu_i(x_{<i}), \sigma_i(x_{<i})$).
**IAF in sampling direction**: $g_{\text{IAF}}: z \to x$ (parallel via $\mu_i(z_{<i}), \sigma_i(z_{<i})$).

만약 두 flow 의 NN $\mu, \sigma$ 같으면 (architecture 같음), 그러나 conditioning 이 다름 ($x$ vs $z$):
- MAF density direction: $z_i = (x_i - \mu_i(x_{<i})) / \sigma_i(x_{<i})$
- IAF sampling direction: $x_i = \mu_i(z_{<i}) + \sigma_i(z_{<i}) z_i$

이 둘은 **서로의 inverse 가 아닐 수 있음** — different conditioning. 그러나 distribution family 는 같음 — 어떤 MAF 가 표현하는 $p_X$ 는 어떤 IAF 도 (다른 parameter 로) 표현 가능.

**그러나 실전적 의의**: density 가 important 면 MAF, sampling 이 important 면 IAF. NN parameter 는 each 에 대해 다름.

---

## 💻 실험 재현 / PyTorch 구현

### 실험 1 — MADE 구현

```python
import torch
import torch.nn as nn
import numpy as np

class MaskedLinear(nn.Linear):
    def __init__(self, in_f, out_f, bias=True):
        super().__init__(in_f, out_f, bias)
        self.register_buffer('mask', torch.ones(out_f, in_f))

    def set_mask(self, mask):
        self.mask.data.copy_(mask.float())

    def forward(self, x):
        return torch.nn.functional.linear(x, self.mask * self.weight, self.bias)

class MADE(nn.Module):
    def __init__(self, dim, hidden=512, n_layers=2):
        super().__init__()
        self.dim = dim
        self.layers = nn.ModuleList()
        # First layer
        self.layers.append(MaskedLinear(dim, hidden))
        for _ in range(n_layers - 1):
            self.layers.append(MaskedLinear(hidden, hidden))
        # Output layer (mu, log_sigma): 2 * dim outputs
        self.out = MaskedLinear(hidden, 2 * dim)

        # Initialize masks
        self.reset_masks(hidden, n_layers)

    def reset_masks(self, hidden, n_layers):
        # Assign integer to each hidden unit
        m = {}
        m[-1] = torch.arange(self.dim) + 1   # input degrees
        for l in range(n_layers):
            m[l] = torch.randint(1, self.dim, (hidden,))
        m[n_layers] = torch.cat([torch.arange(self.dim) + 1, torch.arange(self.dim) + 1])  # output: mu_i, log_sigma_i for i=1..D

        # Hidden masks: m[l-1] >= m[l]? hide
        for l, layer in enumerate(self.layers):
            prev = m[l - 1]
            curr = m[l]
            mask = (prev.unsqueeze(0) <= curr.unsqueeze(1)).float()
            layer.set_mask(mask)
        # Output mask: m[L] > m[L-1] for valid AR
        mask_out = (m[n_layers - 1].unsqueeze(0) < m[n_layers].unsqueeze(1)).float()
        self.out.set_mask(mask_out)

    def forward(self, x):
        h = x
        for layer in self.layers:
            h = torch.relu(layer(h))
        params = self.out(h)
        mu, log_sigma = params.chunk(2, dim=-1)
        return mu, log_sigma
```

### 실험 2 — MAF 구현

```python
class MAF(nn.Module):
    def __init__(self, dim, n_layers=5, hidden=512):
        super().__init__()
        self.dim = dim
        self.flows = nn.ModuleList([MADE(dim, hidden) for _ in range(n_layers)])

    def forward(self, x):
        """x → z (density direction, parallel)"""
        log_det_total = 0
        for flow in self.flows:
            mu, log_sigma = flow(x)
            z = (x - mu) * torch.exp(-log_sigma)
            log_det_total += -log_sigma.sum(-1)
            x = z   # for next layer
        return z, log_det_total

    def inverse(self, z):
        """z → x (sampling, sequential)"""
        for flow in reversed(self.flows):
            x = torch.zeros_like(z)
            for i in range(self.dim):
                mu, log_sigma = flow(x)
                x[:, i] = mu[:, i] + torch.exp(log_sigma[:, i]) * z[:, i]
            z = x   # for next layer
        return x

    def log_p(self, x):
        z, log_det = self.forward(x)
        log_p_z = -0.5 * z.pow(2).sum(-1) - 0.5 * z.shape[-1] * np.log(2 * np.pi)
        return log_p_z + log_det

# 학습
model = MAF(dim=2, n_layers=5)
opt = torch.optim.Adam(model.parameters(), lr=1e-3)

# 2-moon 데이터에 학습 — RealNVP 와 비슷한 결과
```

### 실험 3 — Forward vs Inverse Cost 측정

```python
import time

# Density evaluation (forward in MAF) — parallel
t0 = time.time()
for _ in range(100):
    z, _ = model.forward(torch.randn(64, 2))
print(f"Density (parallel): {time.time() - t0:.3f}s")

# Sampling (inverse in MAF) — sequential
t0 = time.time()
for _ in range(100):
    x = model.inverse(torch.randn(64, 2))
print(f"Sampling (sequential): {time.time() - t0:.3f}s")

# 일반 결과: dim=2 에서는 차이 작음 (sequential 이 D=2 만)
# dim=100, 1000 에서 sampling 이 매우 느려짐
```

### 실험 4 — IAF 의 VAE Posterior

```python
class IAFPosterior(nn.Module):
    """Initial Gaussian → IAF transform"""
    def __init__(self, dim=32, n_layers=4, hidden=128):
        super().__init__()
        self.dim = dim
        # Initial μ, σ from encoder
        # IAF transforms
        self.flows = nn.ModuleList([MADE(dim, hidden) for _ in range(n_layers)])

    def sample(self, mu_init, log_sigma_init):
        """Sample z from IAF posterior"""
        eps = torch.randn_like(mu_init)
        z = mu_init + torch.exp(log_sigma_init) * eps
        log_q_z = -0.5 * eps.pow(2).sum(-1) - log_sigma_init.sum(-1) \
                 - 0.5 * self.dim * np.log(2 * np.pi)
        for flow in self.flows:
            mu, log_sigma = flow(z)
            z = mu + torch.exp(log_sigma) * z
            log_q_z -= log_sigma.sum(-1)   # accumulate Jacobian log_det
        return z, log_q_z

# VAE 에서 사용:
# encoder → mu_init, log_sigma_init → IAF → expressive z
# log q(z|x) = original Gaussian log_p_eps - sum(log_sigma_init) - sum(IAF log_sigma)
# ELBO 계산 시 사용
```

---

## 🔗 이론과 실전의 간극

### 1. MAF/IAF 의 사용 사례 분리

**MAF**:
- Density estimation (anomaly detection): density evaluation 빈번 → parallel critical
- 한 번의 sampling, 많은 likelihood query
- UCI dataset benchmark 에서 SOTA NLL 흔함

**IAF**:
- VAE posterior: encoder forward 시 sampling 필요 → IAF parallel sample 활용
- Parallel WaveNet (van den Oord 2017): IAF student 가 AR teacher 를 distillation, real-time TTS
- Sample 빈번, density rare

### 2. Coupling vs Autoregressive 의 Trade-off

| | Coupling (RealNVP, Glow) | MAF | IAF |
|--|---|----|----|
| Density evaluation | Parallel | Parallel | Sequential |
| Sampling | Parallel | Sequential | Parallel |
| Expressiveness per layer | Low (half unchanged) | Higher (all dims) | Higher (all dims) |
| Implementation | Simple | MADE masks | MADE masks |

**결론**: coupling 이 양방향 모두 parallel 의 장점, 그러나 표현력 부족. MAF/IAF 는 한 방향 efficient + expressive.

### 3. Spline Flow 의 일반화

NSF (Neural Spline Flow, Durkan 2019) 가 affine coupling + spline 으로 확장. MAF 도 spline-MAF 로 일반화 (Durkan 2019). 더 적은 layer 로 같은 표현력.

---

## ⚖️ 가정과 한계

| 가정 | 한계 |
|------|------|
| Autoregressive ordering 자연 | Image, graph 등에서는 ordering 선택이 임의적 |
| Same NN 가 모든 conditional 처리 | Limited expressiveness vs separate NNs |
| MADE mask 가 dependency 보존 | Random mask 의 quality 가 model 마다 |
| Sequential cost 가 acceptable | High-dim ($D > 1000$) 에서 prohibitive |
| Inverse 도 NN forward | Gaussian noise 의 1D inversion 만 고려 |

---

## 📌 핵심 정리

$$\boxed{\text{MAF: } x_i = \mu_i(x_{<i}) + \sigma_i(x_{<i}) z_i \Rightarrow \text{density parallel, sampling sequential}}$$

$$\boxed{\text{IAF: } x_i = \mu_i(z_{<i}) + \sigma_i(z_{<i}) z_i \Rightarrow \text{sampling parallel, density sequential}}$$

| Flow | Density | Sampling | 사용 |
|------|---------|----------|------|
| **MAF** | $O(1)$ parallel | $O(D)$ sequential | Density estimation |
| **IAF** | $O(D)$ sequential | $O(1)$ parallel | VAE posterior, Parallel WaveNet |
| **Coupling** | $O(1)$ parallel | $O(1)$ parallel | General (both efficient, less expressive) |

| MADE 구성 | 역할 |
|-----------|------|
| Hidden integer $m_k$ | Each unit 의 "max input dependency" |
| Mask weight | $W_{j,k} = 0$ if $m_k \geq m_j$ — AR violation 막음 |
| Single forward pass | 모든 $\mu_i, \sigma_i$ 동시 산출 |

---

## 🤔 생각해볼 문제

**문제 1** (기초): $D = 3$ MAF 의 sampling sequential 을 단계별로 보여라. 각 단계에서 어떤 변수가 input 으로 사용되는지 명시.

<details>
<summary>해설</summary>

**Step 1**: $z = (z_1, z_2, z_3) \sim \mathcal{N}(0, I)$, sampling.

**Step 2**: $x_1 = \mu_1(\emptyset) + \sigma_1(\emptyset) z_1$. ($\mu_1, \sigma_1$ 은 conditioning 없음 → constant 또는 input 0)

**Step 3**: $x_2 = \mu_2(x_1) + \sigma_2(x_1) z_2$. ($\mu_2, \sigma_2$ 가 NN 으로 $x_1$ 을 input)

**Step 4**: $x_3 = \mu_3(x_1, x_2) + \sigma_3(x_1, x_2) z_3$.

**관찰**: 각 step 이 NN forward pass 1회 (MADE 가 모든 $\mu_i, \sigma_i$ 동시 계산하지만, 사용 가능한 것은 $i$-th 까지). $D$ steps total — sequential.

**대안**: pure Python loop (above) vs MADE forward + iterate. MADE 의 효율은 density 에 있음, sampling 시는 어차피 sequential.

</details>

**문제 2** (심화): IAF 가 VAE 의 posterior 를 더 expressive 하게 만든다는 주장 — Kingma 2016 의 ELBO 향상 측면에서 분석하라.

<details>
<summary>해설</summary>

**Standard VAE posterior**: $q_\phi(z|x) = \mathcal{N}(\mu_\phi(x), \sigma_\phi^2(x) I)$ — factored Gaussian.

**한계**: 진짜 posterior $p_\theta(z|x)$ 가 일반적으로 multimodal, correlated. Factored Gaussian 으로 근사 → KL gap 큼 → ELBO loose.

**IAF posterior**:
$$z_0 \sim \mathcal{N}(\mu_\phi, \sigma_\phi^2 I), \quad z_i^{(l)} = \mu_l(z^{(l-1)}_{<i}) + \sigma_l(z^{(l-1)}_{<i}) z^{(l-1)}_i$$

(several IAF layers)

**효과**:
- $z$ 의 분포가 factored Gaussian 의 invertible transformation — 임의 복잡한 분포 표현 가능
- AR 구조로 dependency 도입
- $\log q(z|x)$ 가 closed-form: $\log q(z_0) - \sum_l \sum_i \log \sigma_l$

**ELBO 향상**:
$$\log p(x) - \text{ELBO} = \text{KL}(q_\phi(z|x) \| p_\theta(z|x))$$

더 expressive $q$ → KL 더 작음 → ELBO tighter → 더 정확한 likelihood 추정.

**Empirical** (Kingma 2016, MNIST):
- Standard VAE: ELBO ≈ -84
- VAE + IAF posterior: ELBO ≈ -82.0 (better)
- VAE + 다중 IAF layers: ELBO ≈ -80.5 (best)

**Cost**:
- IAF posterior: VAE forward 시 IAF sample (parallel — fast) + density (parallel since z is known on forward — fast)
- Wait — IAF density 는 sequential. 하지만 학습 시는 z 가 sample 시 알려졌으므로 parallel evaluable.

**미묘한 점**: training 시 IAF 의 모든 intermediate $z$ 값이 sample 시 cached → density 계산은 single pass.

**현대적 위치**: IAF posterior 는 정교한 VAE 의 핵심. Hierarchical VAE (NVAE, VDVAE) 도 IAF-style transformation 사용. Diffusion 으로 이동하면서 VAE 자체의 importance 감소.

</details>

**문제 3** (논문 비평): Coupling layer (RealNVP/Glow) vs Autoregressive flow (MAF/IAF) 가 같은 expressiveness 를 가진다는 주장 — Papamakarios 2017 의 결과. 어떤 의미에서 그런가? 실전에서는 어떻게 다른가?

<details>
<summary>해설</summary>

**Theoretical equivalence (Papamakarios 2017)**:

각 architecture 가 sufficient depth 와 expressive $\mu, \sigma$ NN 으로:
- 임의 $C^1$ diffeomorphism $f: \mathbb{R}^D \to \mathbb{R}^D$ 를 임의로 잘 근사 가능
- 따라서 임의 분포 표현 가능 (universal flow approximator)

**동등성 증명** (informal): coupling 이 절반-절반 처리 → 충분한 layer 로 모든 dim 변형. AR 도 각 dim 이 결국 변형. 둘 다 표현력 무한 (in limit).

**실전적 차이**:

1. **Layer 수 / Depth 의 차이**:
   - Coupling: shallow per layer (절반 unchanged). 깊은 stack 필요.
   - AR: each layer 가 모든 dim 변형. 더 적은 layer 로 같은 표현력 가능.

2. **Computational cost asymmetry**:
   - Coupling: forward, inverse 모두 parallel
   - MAF: density parallel, sampling sequential
   - IAF: sampling parallel, density sequential

3. **Architecture flexibility**:
   - Coupling: $s, t$ NN 가 임의 (image 에서 CNN, sequence 에서 Transformer 등)
   - AR: MADE-style mask 필요, NN architecture 제약

4. **Empirical 결과**:
   - UCI tabular density: MAF > Glow (NLL)
   - Image generation: Glow ≈ MAF (둘 다 GAN/Diffusion 보다 못 함)
   - VAE posterior: IAF >> MAF (sampling 빠름)

**시사점**:
- Theoretical equivalence ≠ practical equivalence
- Use case 에 따라 architecture 선택
- 일반적 image generation: coupling 의 양방향 parallel 이 유리
- Density estimation: MAF 의 expressiveness per layer 유리
- Posterior approximation: IAF 의 sampling parallel 유리

**현대적 동향**:
- **Spline-based flows** (NSF, B-NAF) 가 coupling 과 AR 모두에 적용 가능
- **Continuous Normalizing Flow** (FFJORD) 가 architecture-free (next 문서)
- 모든 flow 가 GAN/Diffusion 의 image quality 못 미침 — fundamental 한 likelihood vs perceptual gap

</details>

---

<div align="center">

[◀ 이전 (03. Glow)](./03-glow.md) | [📚 README](../README.md) | [다음 ▶ (05. CNF/Neural ODE)](./05-cnf-neural-ode.md)

</div>
