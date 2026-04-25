# 05. Spectral Normalization (Miyato 2018)

## 🎯 핵심 질문

- Spectral norm $\sigma(W) = \max_{\|x\|=1} \|Wx\|$ = largest singular value 의 의미와 NN Lipschitz 와의 관계?
- Power iteration 으로 $\sigma(W)$ 를 어떻게 efficient 하게 추정하는가? Convergence 의 보장?
- $\|f\|_L \leq \prod_l \sigma(W_l)$ 의 product bound 가 1-Lipschitz constraint 에 어떻게 사용되는가?
- WGAN-GP 의 gradient penalty 와 비교한 spectral norm 의 architectural advantage?
- BigGAN, StyleGAN 등이 spectral norm 을 선택한 이유?

---

## 🔍 왜 Spectral Norm 이 SOTA 의 표준이 되었는가

Miyato 2018 "Spectral Normalization for GANs" 의 기여:

1. **Per-layer Lipschitz constraint**: 각 weight matrix 의 spectral norm = 1
2. **Power iteration**: $\sigma(W)$ 추정의 cheap algorithm
3. **No gradient penalty**: implementation 간단
4. **WGAN-GP 보다 효율**: per-batch overhead 적음

이로 인해 BigGAN (Brock 2019), Self-Attention GAN (Zhang 2018), StyleGAN2 등 주요 GAN 의 표준이 됨. 이 문서에서는 spectral norm 의 수학적 정당성, power iteration 의 efficiency, 그리고 architectural 의의를 다룹니다.

---

## 📐 수학적 선행 조건

- 이전 문서들: 03-mode-collapse, 04-wgan
- [Linear Algebra Deep Dive](https://github.com/iq-ai-lab/linear-algebra-deep-dive): SVD, singular value, matrix norm
- [Optimization Theory Deep Dive](https://github.com/iq-ai-lab/optimization-theory-deep-dive): Power iteration

---

## 📖 직관적 이해

### "각 layer 의 expansion factor 제한"

NN $f = \sigma(W_L \sigma(W_{L-1} \cdots W_1 x))$ 의 Lipschitz constant:

$$\|f\|_L \leq \prod_l \sigma(W_l) \cdot \|\sigma\|_L^L$$

(activation $\sigma$ 의 Lipschitz, e.g., ReLU $\|\sigma\|_L = 1$).

각 layer 의 spectral norm $\sigma(W_l) = 1$ 이면 전체 NN 이 1-Lipschitz (ReLU/LeakyReLU 가정).

### Spectral Norm 의 의미

$W \in \mathbb{R}^{m \times n}$ 의 spectral norm:

$$\sigma(W) = \max_{\|x\| = 1} \|Wx\| = \text{largest singular value}$$

기하학적: unit vector 를 $W$ 로 변환했을 때 가장 길어지는 길이. **Maximum stretching factor**.

### Power Iteration 의 Efficiency

$\sigma(W)$ 의 정확한 계산: SVD $O(\min(m, n)^2 \max(m, n))$. 매 forward 마다 prohibitive.

**Power iteration** (간단 algorithm):
$$u^{(t+1)} = \frac{W v^{(t)}}{\|W v^{(t)}\|}, \quad v^{(t+1)} = \frac{W^\top u^{(t+1)}}{\|W^\top u^{(t+1)}\|}$$

수렴: $u, v$ 가 first left/right singular vectors 에. $\sigma(W) \approx u^\top W v$.

**한 forward 당 1 iteration 만**: $u, v$ 를 매 step 에 update — **online 추정**, cost 매우 작음.

### Spectral Norm 의 적용

각 weight $W_l$ 을 $W_l / \sigma(W_l)$ 로 normalize:

$$\bar W_l = W_l / \sigma(W_l), \quad \sigma(\bar W_l) = 1$$

이로 NN 이 1-Lipschitz (with appropriate activation).

**No additional loss term** — architecture 자체가 constraint.

---

## ✏️ 엄밀한 정의·정리

### 정의 5.1 — Spectral Norm

$W \in \mathbb{R}^{m \times n}$ 의 spectral norm:

$$\sigma(W) := \max_{x \neq 0} \frac{\|Wx\|_2}{\|x\|_2} = \sigma_{\max}(W)$$

= largest singular value of $W$.

### 정리 5.2 — Lipschitz Bound for Linear Layer

Linear layer $y = Wx + b$ 의 Lipschitz constant:

$$\|y_1 - y_2\| = \|W(x_1 - x_2)\| \leq \sigma(W) \|x_1 - x_2\|$$

따라서 $\|f\|_L = \sigma(W)$.

### 정리 5.3 — Lipschitz Bound for Composition

$f = f_L \circ \cdots \circ f_1$ 에서:

$$\|f\|_L \leq \prod_{l=1}^L \|f_l\|_L$$

NN $f = \sigma_L(W_L \sigma_{L-1}(W_{L-1} \cdots W_1 x))$ 에 적용 ($\sigma_l$ = activation):

$$\|f\|_L \leq \prod_l \sigma(W_l) \cdot \prod_l \|\sigma_l\|_L$$

ReLU/LeakyReLU: $\|\sigma\|_L = 1$. 따라서:

$$\|f\|_L \leq \prod_l \sigma(W_l)$$

### 정의 5.4 — Power Iteration

Initialize: $v^{(0)} = $ random unit vector.

Repeat:
$$u^{(t+1)} = \frac{W v^{(t)}}{\|W v^{(t)}\|}, \quad v^{(t+1)} = \frac{W^\top u^{(t+1)}}{\|W^\top u^{(t+1)}\|}$$

After convergence: $\sigma(W) \approx u^\top W v$.

### 정리 5.5 — Power Iteration Convergence

Random initialization $v^{(0)}$ 에서, $W$ 의 first singular value 가 second 와 strictly greater 이면 ($\sigma_1 > \sigma_2$):

$$v^{(t)} \to v_1 \quad \text{(first right singular vector)}$$

수렴 속도: linear with rate $\sigma_2 / \sigma_1 < 1$.

**Empirical**: 1-2 iterations per training step 으로 충분 — $u, v$ 가 step 간 약간만 변함.

### 정의 5.6 — Spectral Normalization

각 layer 의 weight $W_l$ 을 $\bar W_l = W_l / \sigma(W_l)$ 로 replace.

**Implementation**: forward pass 에서:

```python
def forward_with_sn(W, x):
    sigma = power_iteration(W, u, v, n_iter=1)
    W_normalized = W / sigma
    return F.linear(x, W_normalized)
```

$u, v$ 를 buffer 로 저장, 매 step update.

### 정리 5.7 — SN-Constrained NN 의 Lipschitz

NN 의 모든 weight 가 spectral normalized: $\sigma(\bar W_l) = 1$ for all $l$.

$$\|f\|_L \leq \prod_l 1 = 1$$

(with 1-Lipschitz activation). NN 이 **exactly** 1-Lipschitz (또는 conservative upper bound).

---

## 🔬 증명 및 수학적 유도

### 유도 1 — Spectral Norm 이 Lipschitz Constant

Linear $y = Wx$ 의 Lipschitz:

$\|Wx - Wx'\|^2 = \|W(x - x')\|^2$.

Maximize over $x - x'$ unit vectors: $\max_{\|d\| = 1} \|Wd\|^2 = \sigma_\max^2(W)$.

따라서 $\|W\|_L = \sigma(W)$. $\square$

### 유도 2 — Power Iteration 의 Convergence

$W = U \Sigma V^\top$ (SVD). $W^\top W = V \Sigma^2 V^\top$ — symmetric, eigenvalues $\sigma_i^2$.

Power iteration on $W^\top W$: $v^{(t+1)} \propto W^\top W v^{(t)}$. Decompose $v^{(0)} = \sum_i c_i v_i$:

$$(W^\top W)^t v^{(0)} = \sum_i c_i \sigma_i^{2t} v_i = \sigma_1^{2t}\left(c_1 v_1 + \sum_{i > 1} c_i (\sigma_i / \sigma_1)^{2t} v_i\right)$$

$\sigma_i / \sigma_1 < 1$ for $i > 1$ → as $t \to \infty$, dominant term $c_1 v_1$.

**Normalized**: $v^{(t)} \to v_1$ (first right singular vector) at rate $(\sigma_2 / \sigma_1)^{2t}$.

**Online**: warm-started $v$ 가 매 step 약간 변형되는 $W$ 에 대해 잘 적응.

### 유도 3 — SN vs WGAN-GP 의 Gradient Comparison

**WGAN-GP**: gradient penalty term $\lambda(\|\nabla f\|-1)^2$ — soft, bias 있음. Critic 의 Lipschitz 가 정확히 1 아님.

**Spectral Norm**: 각 layer 의 spectral norm 정확히 1 → product bound가 1 (conservative). 그러나 **conservative**: actual Lipschitz 가 1 보다 작을 수 있음 (skip connection 등에서).

**Trade-off**:
- WGAN-GP: tight 1-Lipschitz, but soft + costly
- SN: hard but conservative + cheap

**Empirical**: 둘 다 stable training, sample quality 비슷. SN 이 더 일반적 (BigGAN, StyleGAN).

### 유도 4 — SN 의 Inductive Bias

$\sigma(W) = 1$ 이 NN 의 expressiveness 제약. 그러나:

1. **Multiple layers**: $\prod_l \sigma(W_l) = 1$ — total expressiveness 분배
2. **Non-linear activation**: ReLU 가 piece-wise linear, complex function 표현 가능
3. **Skip connections**: residual block 의 identity 가 $\sigma$ 에 영향

따라서 SN 이 "constrained but expressive" — practical NN 에 적합.

### 유도 5 — Implementation Detail: U, V 의 Online Update

Per training step:

```
1. Detach W from graph (no gradient through SN computation)
2. v = (W^T @ u) / ||...||
3. u = (W @ v) / ||...||
4. sigma = u @ W @ v
5. W_normalized = W / sigma
6. Use W_normalized in forward
```

**Subtle**: $\sigma$ 의 gradient 가 $W$ 에 대해 backprop 되어야 하나 안 되어야 하나? 일반적으로 backprop 차단 (detach $u, v$) — $\sigma$ 가 상수처럼 보이게.

이로 학습이 stable.

---

## 💻 실험 재현 / PyTorch 구현

### 실험 1 — Spectral Norm Module 구현

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class SpectralNorm(nn.Module):
    def __init__(self, module, name='weight', n_power_iter=1):
        super().__init__()
        self.module = module
        self.name = name
        self.n_power_iter = n_power_iter
        # Initialize u, v
        weight = getattr(module, name)
        u = nn.Parameter(torch.randn(weight.size(0)).normalize_(dim=0), requires_grad=False)
        v = nn.Parameter(torch.randn(weight.size(1)).normalize_(dim=0), requires_grad=False)
        self.register_buffer('u', u)
        self.register_buffer('v', v)

    def update(self):
        weight = getattr(self.module, self.name)
        weight_2d = weight.view(weight.size(0), -1)
        u, v = self.u, self.v
        with torch.no_grad():
            for _ in range(self.n_power_iter):
                v = F.normalize(weight_2d.T @ u, dim=0)
                u = F.normalize(weight_2d @ v, dim=0)
            self.u.copy_(u); self.v.copy_(v)
        sigma = u @ weight_2d @ v
        return sigma

    def forward(self, x):
        sigma = self.update()
        weight = getattr(self.module, self.name)
        weight_normalized = weight / sigma
        # Replace weight temporarily
        original_weight = weight.data.clone()
        weight.data = weight_normalized.data
        out = self.module(x)
        weight.data = original_weight
        return out

# 더 깔끔하게: torch.nn.utils.spectral_norm
sn_layer = nn.utils.spectral_norm(nn.Linear(128, 64))
```

### 실험 2 — SN-GAN on 8-Gaussians

```python
class D_with_SN(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.utils.spectral_norm(nn.Linear(2, 128)), nn.LeakyReLU(0.2),
            nn.utils.spectral_norm(nn.Linear(128, 128)), nn.LeakyReLU(0.2),
            nn.utils.spectral_norm(nn.Linear(128, 1)),
        )
    def forward(self, x):
        return self.net(x)

D_sn = D_with_SN()
opt_D = optim.Adam(D_sn.parameters(), lr=2e-4, betas=(0.0, 0.9))
opt_G = optim.Adam(G.parameters(), lr=2e-4, betas=(0.0, 0.9))

# Hinge loss (SN-GAN paper recommends)
def hinge_loss_d(d_real, d_fake):
    return F.relu(1 - d_real).mean() + F.relu(1 + d_fake).mean()

def hinge_loss_g(d_fake):
    return -d_fake.mean()

for step in range(20000):
    # Train D
    x_real = sample_8gauss(256)
    z = torch.randn(256, 2)
    x_fake = G(z).detach()
    loss_D = hinge_loss_d(D_sn(x_real), D_sn(x_fake))
    opt_D.zero_grad(); loss_D.backward(); opt_D.step()

    # Train G
    z = torch.randn(256, 2)
    loss_G = hinge_loss_g(D_sn(G(z)))
    opt_G.zero_grad(); loss_G.backward(); opt_G.step()

# Mode coverage: 일반적으로 8 modes 모두 cover
```

### 실험 3 — SN 의 Lipschitz 효과 검증

```python
# 임의 입력에서 NN 의 Lipschitz constant 측정
def empirical_lipschitz(net, n_pairs=100, n_perturb=10):
    """입력 perturbation 에 대한 출력 변화 비율의 max"""
    max_ratio = 0
    for _ in range(n_pairs):
        x = torch.randn(1, 2)
        for _ in range(n_perturb):
            d = torch.randn(1, 2) * 0.1
            ratio = (net(x + d) - net(x)).norm() / d.norm()
            max_ratio = max(max_ratio, ratio.item())
    return max_ratio

D_no_sn = D()    # 표준 D
D_sn = D_with_SN()
print(f"Lipschitz (no SN): {empirical_lipschitz(D_no_sn):.3f}")
print(f"Lipschitz (SN):    {empirical_lipschitz(D_sn):.3f}")
# SN: 1.0 근처 (또는 약간 작음, conservative)
# No SN: 매우 큼 (10+ 가능)
```

### 실험 4 — Power Iteration 의 Convergence

```python
W = torch.randn(64, 128)
u = torch.randn(64)
u = u / u.norm()
v = torch.randn(128)
v = v / v.norm()

# True spectral norm
true_sigma = torch.linalg.svd(W).S[0].item()

estimates = []
for _ in range(20):
    v = (W.T @ u); v = v / v.norm()
    u = (W @ v); u = u / u.norm()
    sigma = (u @ W @ v).item()
    estimates.append(sigma)

import matplotlib.pyplot as plt
plt.plot(estimates, 'o-', label='Power iteration')
plt.axhline(true_sigma, color='r', linestyle='--', label='True σ(W)')
plt.xlabel('Iteration'); plt.ylabel('σ estimate')
plt.legend(); plt.title('Power Iteration Convergence')
plt.show()
# 1-3 iteration 으로 거의 정확한 σ
```

---

## 🔗 이론과 실전의 간극

### 1. SN 의 Conservative Bound

$\prod_l \sigma(W_l) = 1$ 이 **upper bound** of $\|f\|_L$, exact 가 아님. Skip connection 등에서:

$$\|y\|_L = \|x + W \sigma(\hat x)\|_L \leq 1 + \sigma(W) \cdot 1 = 2$$

(if naive). 하지만 actual Lipschitz 는 더 작을 수 있음.

이는 conservative — 실제 expressiveness 손해 가능. 그러나 stability 우위.

### 2. SN + Hinge Loss 의 결합

SN-GAN 원 논문은 **hinge loss** 와 함께 사용:

$$L_D = \mathbb{E}[\max(0, 1 - D(x))] + \mathbb{E}[\max(0, 1 + D(G(z)))]$$

$$L_G = -\mathbb{E}[D(G(z))]$$

이는 logistic 보다 saturation 적고, SN 의 1-Lipschitz 와 잘 호환. BigGAN 등의 표준.

### 3. Modern GAN 의 SN Usage

- **BigGAN** (Brock 2019): SN + orthogonal regularization + hinge loss → ImageNet 256×256 conditional generation SOTA
- **SAGAN** (Zhang 2018): SN + self-attention → long-range dependency
- **StyleGAN**: 부분적 SN (mapping network 등)
- **VQGAN** (Esser 2021): SN in discriminator

SN 이 modern GAN 의 default.

---

## ⚖️ 가정과 한계

| 가정 | 한계 |
|------|------|
| ReLU 가 1-Lipschitz | Sigmoid, tanh 도 1-Lipschitz (different bound) |
| Product bound 가 tight | Skip connection, BN 에서 부정확 |
| Power iteration 1 step 충분 | Rapidly changing $W$ 에서 부정확 가능 |
| Conservative bound 가 OK | Expressiveness 약간 손해 |
| Per-layer $\sigma(W) = 1$ | Layer 마다 다른 capacity 분배 못 함 |

---

## 📌 핵심 정리

$$\boxed{\sigma(W) = \max_{\|x\| = 1} \|Wx\| = \sigma_\max(W)}$$

$$\boxed{\|f\|_L \leq \prod_l \sigma(W_l) \quad \text{(NN with 1-Lip activation)}}$$

| Component | 역할 |
|-----------|------|
| **Power iteration** | $\sigma(W)$ 의 cheap online estimation |
| **SN: $\bar W = W / \sigma$** | Per-layer Lipschitz = 1 강제 |
| **No GP** | Architecture-level enforcement |
| **Hinge loss** | SN 과 잘 호환 |
| **u, v buffers** | Online state, 1-2 iteration per step |

| 비교 | WGAN-GP | Spectral Norm |
|------|---------|---------------|
| Constraint type | Soft (loss) | Hard (architecture) |
| Cost | 2x backward | Per-layer cheap |
| Lipschitz exactness | Approximate | Conservative |
| Modern usage | Less common | Default in BigGAN, SAGAN, etc. |

---

## 🤔 생각해볼 문제

**문제 1** (기초): $W = \begin{pmatrix} 2 & 0 \\ 0 & 3 \end{pmatrix}$ 의 spectral norm 을 계산하고, power iteration 으로 검증하라.

<details>
<summary>해설</summary>

**Direct**: $W$ 가 diagonal, singular values = diagonal entries (positive case). $\sigma(W) = \max(2, 3) = 3$.

**Power iteration**:

$v^{(0)} = (1, 0)^\top / 1 = (1, 0)$.

Iter 1: $u^{(1)} = W v^{(0)} = (2, 0)$, normalize: $u = (1, 0)$. $v^{(1)} = W^\top u = (2, 0)$, normalize: $v = (1, 0)$.

Stuck at $(1, 0)$ which corresponds to $\sigma = 2$. **Bad initial guess** — $v^{(0)}$ has no component along second singular vector.

**Random init**: $v^{(0)} = $ random unit. Has both $v_1, v_2$ components. Iteration:

After many steps, dominant $v_2$ (corresponding to $\sigma = 3$) survives. $\sigma_\text{est} = 3$.

**시사점**: power iteration 의 random initialization 중요. Edge case (degenerate) 에서 stuck 가능. Random init 으로 거의 항상 OK.

</details>

**문제 2** (심화): NN 이 conv layer + ReLU 와 skip connection 가질 때의 spectral norm 적용 — Lipschitz bound 가 어떻게 변하는가?

<details>
<summary>해설</summary>

**Conv layer**: $W$ 가 weight tensor. Spectral norm 정의는 동일하지만, conv 는 toeplitz structure → $\sigma(W_\text{conv})$ 가 **input size** 에 의존 (실제로는 spatial structure 가 효율적으로 계산).

**Implementation**:
- Reshape $W \in \mathbb{R}^{C_\text{out} \times C_\text{in} \times k \times k}$ to $W_2 \in \mathbb{R}^{C_\text{out} \times C_\text{in} k^2}$
- Apply standard SN to $W_2$
- 이는 conservative bound (실제 conv operator 의 spectral norm 보다 큼)

**Skip connection** $y = x + F(x)$:

$$\|y_1 - y_2\| = \|(x_1 - x_2) + (F(x_1) - F(x_2))\| \leq \|x_1 - x_2\| + \|F\|_L \|x_1 - x_2\|$$

= $(1 + \|F\|_L) \|x_1 - x_2\|$.

**Lipschitz**: $\|y\|_L \leq 1 + \|F\|_L$.

**SN with skip**: $F$ 의 spectral norm = 1 만들면 $y$ 의 Lipschitz 가 **2** — conservative bound.

**문제**: residual block 가지면 1-Lipschitz 아님. Solution:
- Scale skip: $y = \alpha x + (1 - \alpha) F(x)$ — convex combination
- Or accept conservative bound (실전에서 큰 영향 없음)

**Batch Norm**: BN 의 affine $\gamma x + \beta$ 의 Lipschitz = $|\gamma|_\infty / \sigma_\text{BN}$. SN 만으로 enforcement 안 됨.

**Modern practice**: SN + careful architecture (no BN in critic) — BigGAN 의 선택.

**시사점**: SN 은 ideal architecture (linear + ReLU) 에서 정확하게 1-Lipschitz, 실전 architecture 에서는 conservative upper bound. Stability 의 source 이지만 expressiveness 의 약간 손해.

</details>

**문제 3** (논문 비평): Spectral Norm 이 BigGAN 의 SOTA 결과의 핵심 component 인 이유 — Lipschitz 외에 어떤 effect 가 있는가?

<details>
<summary>해설</summary>

**SN 의 직접적 효과 (Lipschitz)**: WGAN-style stability — vanishing gradient 회피.

**간접적 효과**:

**1. Implicit Regularization**:
- $\sigma(W) = 1$ 이 weight 의 magnitude 제한 → overfitting 감소
- Discriminator 가 너무 정확해지지 않음 → generator 가 학습 가능

**2. Gradient Flow**:
- 깊은 network 에서 $\sigma(W) > 1$ 이면 gradient explosion, $< 1$ 이면 vanishing
- $\sigma(W) = 1$ 이 sweet spot — gradient 가 일정한 magnitude 유지

**3. Mode Coverage**:
- D 가 너무 강해지지 않음 → G 가 다양한 mode 학습
- 강한 D 가 일부 region 만 reject → mode collapse 유도

**4. Conditional Generation 의 Stability**:
- BigGAN 의 class-conditional generation: 1000 ImageNet classes
- Class-specific feature 가 SN 으로 stable 하게 학습

**BigGAN 의 다른 ingredient**:
- **Truncation trick**: latent $z$ 의 truncated normal — quality vs diversity trade-off
- **Orthogonal regularization**: weight 의 orthogonality
- **Large batch**: 2048+ batch size
- **Class-conditional BN**: per-class BN parameters

이들 + SN 의 결합이 ImageNet 256×256 SOTA.

**SN 의 잔존 영향**:
- StyleGAN2: SN 일부 사용
- VQGAN: SN in discriminator
- Diffusion 의 일부 component (e.g., adversarial loss for refinement)

**현대적 관점**:
- Diffusion 시대로 GAN 의 사용 감소
- 그러나 GAN-based vocoder, super-resolution 에서는 여전히 SN 표준
- Transformer 기반 GAN 에서도 SN 활용

**Trade-off 의 본질**:
- SN: stability + simplicity, conservative
- WGAN-GP: tight Lipschitz, costly
- 둘 다 standard GAN 의 fundamental issues 해결, 다른 방식

**시사점**: SN 이 GAN 의 architectural 표준 — Lipschitz 의 "free + stable" enforcement. BigGAN 의 SOTA 가 SN 만으로 가능한 게 아니지만, SN 이 essential ingredient. GAN 의 maturity 의 mark.

</details>

---

<div align="center">

[◀ 이전 (04. WGAN)](./04-wgan.md) | [📚 README](../README.md) | [다음 ▶ (06. StyleGAN)](./06-stylegan.md)

</div>
