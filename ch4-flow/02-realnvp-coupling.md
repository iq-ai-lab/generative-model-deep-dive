# 02. RealNVP — Coupling Layer (Dinh 2017)

## 🎯 핵심 질문

- RealNVP 의 coupling layer $y_{1:d} = x_{1:d}, y_{d+1:D} = x_{d+1:D} \odot \exp(s(x_{1:d})) + t(x_{1:d})$ 가 어떻게 invertible 인가?
- Jacobian 이 lower triangular 가 되는 정확한 이유는? $|\det J| = \exp(\sum_i s_i)$ 의 유도?
- Coupling 의 forward 와 inverse 모두 NN forward pass 1회 — 다른 flow architecture 와의 차이는?
- Alternating split (한 층은 [first half | second half], 다음 층은 reversed) 이 왜 필요한가?
- Squeezing, multi-scale 같은 image-specific 기법이 RealNVP 의 expressiveness 를 어떻게 향상시키는가?

---

## 🔍 왜 RealNVP 가 결정적 발견인가

Dinh 2017 "Density Estimation using Real NVP" 의 coupling layer 는 **invertible NN architecture 의 첫 실용적 design**:

1. **Forward 와 inverse 모두 1 forward pass** — Flow 의 핵심 효율성
2. **Triangular Jacobian → $O(d)$ determinant** — change of variables 의 효율 계산
3. **무한히 stack 가능** — 각 layer 가 invertible, 체인 적용
4. **Architecture 의 free part** — $s, t$ NN 은 자유 (deep CNN 등 가능)

이전의 NICE (Dinh 2015) 는 $\det J = 1$ (volume preserving) 만 가능. RealNVP 의 affine coupling 이 일반화하여 expressive 함을 크게 증가. Glow, MAF, FFJORD 등 모든 후속 flow 의 기반.

이 문서에서는 RealNVP 의 coupling layer 를 수학적으로 정확히 분석하고, 왜 그 design 이 작동하는지를 다룹니다.

---

## 📐 수학적 선행 조건

- 이전 문서: 01-change-of-variables.md
- [Linear Algebra Deep Dive](https://github.com/iq-ai-lab/linear-algebra-deep-dive): Triangular matrix, determinant
- [CNN Deep Dive](https://github.com/iq-ai-lab/cnn-deep-dive): Conv, image processing

---

## 📖 직관적 이해

### "절반은 그대로, 절반만 변환"

Coupling layer 의 핵심 아이디어: 입력 $x \in \mathbb{R}^D$ 을 두 half $x_A, x_B$ 로 나누고:

- $x_A$ 는 그대로 통과: $y_A = x_A$
- $x_B$ 는 $x_A$ 에 의존하는 transformation 적용: $y_B = T(x_B; x_A)$

$T$ 가 invertible (in $x_B$, given $x_A$) 이면 전체 layer invertible:
- Forward: $y_A = x_A, y_B = T(x_B; x_A)$
- Inverse: $x_A = y_A, x_B = T^{-1}(y_B; y_A)$ — 같은 NN forward pass

이 구조의 마법: **Jacobian 이 자동으로 triangular** — $\partial y_A / \partial x_B = 0$ (block).

### "Affine Coupling" 의 표현력

RealNVP 의 specific choice: affine transformation

$$y_B = x_B \odot \exp(s(x_A)) + t(x_A)$$

여기서 $s, t: \mathbb{R}^d \to \mathbb{R}^{D-d}$ 는 임의 NN.

**왜 affine?** 두 가지 이유:
1. **Element-wise invertible**: $x_B = (y_B - t(x_A)) \odot \exp(-s(x_A))$
2. **Diagonal Jacobian wrt $x_B$**: $\partial y_{B,i} / \partial x_{B,i} = \exp(s_i)$

NICE 는 $y_B = x_B + t(x_A)$ (additive only) — $|\det| = 1$. RealNVP 는 scale 추가 → 더 expressive.

### "Alternating Mask" 의 필요성

만약 layer 마다 같은 split ($x_A, x_B$) 사용 → $x_A$ 는 영원히 변형 안 됨 (each layer leaves it identity).

**해결**: 매 layer 마다 split 교체.
- Layer 1: $A = $ first half, $B = $ second half
- Layer 2: $A = $ second half, $B = $ first half
- Layer 3: $A = $ checkerboard pattern A
- ...

이로 모든 dimension 이 결국 변형됨. Glow 의 1×1 conv 가 일반화.

---

## ✏️ 엄밀한 정의·정리

### 정의 2.1 — Affine Coupling Layer

입력 $x \in \mathbb{R}^D$, 분할 indices $A \subset \{1, \ldots, D\}$, $B = \{1, \ldots, D\} \setminus A$.

NN $s: \mathbb{R}^{|A|} \to \mathbb{R}^{|B|}$, $t: \mathbb{R}^{|A|} \to \mathbb{R}^{|B|}$ 가 주어졌을 때:

**Forward**:
$$y_A = x_A$$
$$y_B = x_B \odot \exp(s(x_A)) + t(x_A)$$

**Inverse**:
$$x_A = y_A$$
$$x_B = (y_B - t(y_A)) \odot \exp(-s(y_A))$$

### 정리 2.2 — Coupling 의 Invertibility

위 forward 와 inverse 가 well-defined (모든 $x, y$ 에 대해).

**증명**: forward 후 inverse 적용:

$$x_B = (y_B - t(y_A)) \odot \exp(-s(y_A))$$
$$= (x_B \odot \exp(s(x_A)) + t(x_A) - t(x_A)) \odot \exp(-s(x_A))$$
$$= x_B \odot \exp(s(x_A)) \odot \exp(-s(x_A)) = x_B \quad \square$$

**핵심**: $s, t$ 가 임의의 NN 이라도 invertibility 가 architecture 에 의해 보장 — coupling 의 magic.

### 정리 2.3 — Coupling Jacobian Determinant

위 affine coupling 의 Jacobian $J = \partial y / \partial x \in \mathbb{R}^{D \times D}$:

$$J = \begin{pmatrix} I_{|A| \times |A|} & 0 \\ \partial y_B / \partial x_A & \text{diag}(\exp(s(x_A))) \end{pmatrix}$$

**Lower triangular block** form. Determinant:

$$\det J = \det(I) \cdot \det(\text{diag}(\exp(s))) = 1 \cdot \prod_i \exp(s_i) = \exp\left(\sum_i s_i(x_A)\right)$$

따라서:

$$\log |\det J| = \sum_{i=1}^{|B|} s_i(x_A)$$

**$O(|B|)$ 계산** — coupling 의 결정적 효율 이점.

### 정리 2.4 — Coupling Stack 의 Likelihood

$L$ coupling layers $f_1, \ldots, f_L$ 의 stack:

$$\log p_X(x) = \log p_Z(z_0) - \sum_{l=1}^L \sum_{i \in B_l} s^{(l)}_i(z_{l-1, A_l})$$

각 layer 가 alternating split $A_l$ 사용하면 모든 dimension 이 결국 변형.

### 정의 2.5 — Image RealNVP: Squeezing + Multi-Scale

이미지 $H \times W \times C$ 에서 coupling 을 효과적으로 적용:

**Squeezing**: $(H \times W \times C) \to (H/2 \times W/2 \times 4C)$ — spatial 을 channel 로 reshape, RF 효과적 확장.

**Checkerboard mask** vs **channel mask**: spatial 또는 channel 차원으로 split.

**Multi-scale**: 일부 dim 을 일찍 "factor out" (Gaussian 으로 modeling), 나머지로 deeper layer.

### 정리 2.6 — Multi-Scale 의 효과

$L$-step multi-scale 에서, 각 scale 의 latent $z^{(l)}$ 가 prior $\mathcal{N}(0, I)$. Total log-likelihood:

$$\log p(x) = \sum_l \log p(z^{(l)}) + \sum_l \log |\det J_{f_l}|$$

**효과**: low-resolution scale 이 global structure, high-resolution 이 local detail. Convergence 빠름, sample quality 개선.

---

## 🔬 증명 및 수학적 유도

### 유도 1 — Coupling Jacobian 의 Block 구조

$y = f(x)$ 에서:
- $y_A = x_A$ → $\partial y_A / \partial x_A = I, \partial y_A / \partial x_B = 0$
- $y_B = x_B \cdot \exp(s) + t$, where $s = s(x_A), t = t(x_A)$

$$\partial y_{B,i} / \partial x_{A,j} = x_{B,i} \cdot \exp(s_i) \cdot \partial s_i / \partial x_{A,j} + \partial t_i / \partial x_{A,j}$$

(non-zero in general)

$$\partial y_{B,i} / \partial x_{B,i} = \exp(s_i)$$

$$\partial y_{B,i} / \partial x_{B,j} = 0 \text{ for } j \neq i$$

따라서 $\partial y_B / \partial x_B = \text{diag}(\exp(s))$. Block 구조:

$$J = \begin{pmatrix} I & 0 \\ * & \text{diag}(\exp(s)) \end{pmatrix}$$

### 유도 2 — Block Triangular Determinant

$$\det \begin{pmatrix} A & 0 \\ C & D \end{pmatrix} = \det(A) \cdot \det(D)$$

(block triangular formula). $A = I, D = \text{diag}(\exp(s))$:

$$\det J = 1 \cdot \prod_i \exp(s_i) = \exp\left(\sum_i s_i\right) > 0$$

**Sign**: 항상 양수, 절댓값 그대로. Invertibility 보장 ($\det \neq 0$).

### 유도 3 — Why $\exp(s)$ 가 다른 $\sigma(s)$ 보다 좋은가

선택지:
- **$\exp(s)$**: $(0, \infty)$, multiplicative 가능, no upper bound
- **$\text{softplus}(s)$**: $(0, \infty)$, smoother
- **$\sigma(s)$**: $(0, 1)$, 항상 contraction → expressiveness 부족

RealNVP 가 $\exp$ 선택: **expressiveness** 와 **gradient flow** 둘 다 좋음. 단, $s$ 가 너무 크면 numerical 폭발 위험 → 실전에서 $\tanh(s) \cdot c$ 또는 clamp.

### 유도 4 — Inversion Cost

Coupling 의 forward:
1. Compute $s = s(x_A), t = t(x_A)$ — NN forward pass
2. Compute $y_B = x_B \exp(s) + t$ — element-wise

Coupling 의 inverse:
1. Compute $s = s(y_A), t = t(y_A)$ — same NN, **forward pass** (NN 자체는 invertible 일 필요 없음)
2. Compute $x_B = (y_B - t) \exp(-s)$

따라서 forward 와 inverse **같은 비용**. 이것이 **MAF/IAF 와의 결정적 차이** (Ch4-04 참조 — MAF 는 forward fast, inverse slow).

### 유도 5 — Multi-Scale 의 Variational Argument

모든 dim 에 같은 prior $\mathcal{N}(0, I)$ 적용 시: 모든 dim 이 같은 "complexity" 표현 — wasteful.

Multi-scale: 일부 dim 을 일찍 "factor out" (separate Gaussian), 그 dim 들은 simple structure 모델링. 나머지 dim 이 deeper layer 거쳐 complex structure 표현.

**효과**: hierarchical representation — coarse-to-fine, NLL 개선 + sample 다양성.

---

## 💻 실험 재현 / PyTorch 구현

### 실험 1 — RealNVP Coupling Layer 구현

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class CouplingLayer(nn.Module):
    def __init__(self, dim, hidden=128, mask_type='alternating', layer_idx=0):
        super().__init__()
        self.dim = dim
        # alternating mask
        if mask_type == 'alternating':
            mask = torch.zeros(dim)
            if layer_idx % 2 == 0:
                mask[:dim // 2] = 1
            else:
                mask[dim // 2:] = 1
        self.register_buffer('mask', mask)
        # s, t networks
        self.s_t = nn.Sequential(
            nn.Linear(dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, 2 * dim),
        )

    def forward(self, x):
        x_a = x * self.mask
        s_t = self.s_t(x_a)
        s, t = s_t.chunk(2, dim=-1)
        s = torch.tanh(s) * (1 - self.mask)   # only modify dim_b
        t = t * (1 - self.mask)
        y = x * torch.exp(s) + t
        log_det = s.sum(-1)
        return y, log_det

    def inverse(self, y):
        y_a = y * self.mask
        s_t = self.s_t(y_a)
        s, t = s_t.chunk(2, dim=-1)
        s = torch.tanh(s) * (1 - self.mask)
        t = t * (1 - self.mask)
        x = (y - t) * torch.exp(-s)
        log_det = -s.sum(-1)
        return x, log_det

class RealNVP(nn.Module):
    def __init__(self, dim, n_layers=8, hidden=128):
        super().__init__()
        self.layers = nn.ModuleList([
            CouplingLayer(dim, hidden, layer_idx=i) for i in range(n_layers)
        ])

    def forward(self, x):
        log_det_total = torch.zeros(x.shape[0], device=x.device)
        for layer in self.layers:
            x, log_det = layer(x)
            log_det_total += log_det
        return x, log_det_total

    def inverse(self, z):
        log_det_total = torch.zeros(z.shape[0], device=z.device)
        for layer in reversed(self.layers):
            z, log_det = layer.inverse(z)
            log_det_total += log_det
        return z, log_det_total

    def log_p(self, x):
        z, log_det = self.forward(x)
        log_p_z = -0.5 * z.pow(2).sum(-1) - 0.5 * z.shape[-1] * torch.log(torch.tensor(2*torch.pi))
        return log_p_z + log_det
```

### 실험 2 — 2-Moon Density Estimation

```python
import numpy as np
from sklearn.datasets import make_moons
import matplotlib.pyplot as plt

# Data
data, _ = make_moons(n_samples=10000, noise=0.05)
data = torch.tensor(data, dtype=torch.float32)

# Train
flow = RealNVP(dim=2, n_layers=10, hidden=64)
opt = torch.optim.Adam(flow.parameters(), lr=1e-3)

for step in range(5000):
    idx = torch.randperm(len(data))[:256]
    x = data[idx]
    log_p = flow.log_p(x)
    loss = -log_p.mean()
    opt.zero_grad(); loss.backward(); opt.step()
    if step % 500 == 0:
        print(f"Step {step}: NLL = {-log_p.mean().item():.3f}")

# Visualize learned density
xx, yy = torch.meshgrid(torch.linspace(-2, 3, 100), torch.linspace(-1, 1.5, 100), indexing='ij')
grid = torch.stack([xx.flatten(), yy.flatten()], dim=-1)
with torch.no_grad():
    log_p_grid = flow.log_p(grid).view(100, 100)
plt.contourf(xx, yy, log_p_grid.exp(), levels=30)
plt.scatter(data[:500, 0], data[:500, 1], s=2, c='red')
plt.title('RealNVP learned density on 2-moons')
plt.show()
```

### 실험 3 — Sampling 검증

```python
# Sample from base
with torch.no_grad():
    z = torch.randn(1000, 2)
    x_samples, _ = flow.inverse(z)

plt.scatter(x_samples[:, 0], x_samples[:, 1], s=2, alpha=0.5, label='Generated')
plt.scatter(data[:500, 0], data[:500, 1], s=2, c='red', alpha=0.5, label='Real')
plt.legend(); plt.title('Real vs Generated 2-moons')
plt.show()
```

### 실험 4 — Forward/Inverse Consistency

```python
# 검증: f(f^{-1}(x)) ≈ x
x_test = torch.randn(100, 2)
z, _ = flow.forward(x_test)
x_recon, _ = flow.inverse(z)
error = (x_recon - x_test).abs().max()
print(f"Forward-inverse consistency error: {error.item():.6f}")
# 예상: < 1e-5 (numerical precision)
```

### 실험 5 — Image RealNVP (MNIST simplified)

```python
class ImageCoupling(nn.Module):
    def __init__(self, channels=1, hidden=64, mask_type='checkerboard'):
        super().__init__()
        self.mask_type = mask_type
        self.s_t_net = nn.Sequential(
            nn.Conv2d(channels, hidden, 3, padding=1), nn.ReLU(),
            nn.Conv2d(hidden, hidden, 3, padding=1), nn.ReLU(),
            nn.Conv2d(hidden, 2 * channels, 3, padding=1),
        )

    def get_mask(self, shape):
        mask = torch.zeros(shape)
        if self.mask_type == 'checkerboard':
            for i in range(shape[-2]):
                for j in range(shape[-1]):
                    mask[..., i, j] = (i + j) % 2
        return mask.to(self.s_t_net[0].weight.device)

    def forward(self, x):
        mask = self.get_mask(x.shape)
        x_masked = x * mask
        s_t = self.s_t_net(x_masked)
        s, t = s_t.chunk(2, dim=1)
        s = torch.tanh(s) * (1 - mask)
        t = t * (1 - mask)
        y = x * torch.exp(s) + t
        log_det = s.flatten(1).sum(-1)
        return y, log_det

# MNIST 에서 학습 → bits per dim 측정
```

---

## 🔗 이론과 실전의 간극

### 1. Numerical Stability

$\exp(s)$ 가 large $s$ 에서 폭발. 실전 stabilization:
- **Tanh clamping**: $s = \tanh(\hat s) \cdot c$ for $c \in [2, 5]$
- **Gradient clipping**
- **Activation Normalization** (Glow 의 ActNorm)

### 2. Image-Specific Architectural Choices

표준 coupling (split half) 는 image 에 부적절 (spatial structure 무시):
- **Checkerboard mask**: 픽셀 단위 alternating
- **Channel mask**: channel 단위 split (after squeezing)
- **1×1 conv** (Glow): channel permutation 의 학습된 generalization

이들이 RealNVP 의 표현력을 image 에 적합하게 확장.

### 3. Coupling 의 Information Bottleneck

각 coupling 에서 절반 입력은 unchanged → information 이 layer 마다 절반 영향. 따라서:
- 깊은 stack 필요 (수십 layer)
- Same dim throughout (compression 불가)
- 효과적 표현력이 layer 수에 strong dependence

이것이 Flow 가 GAN/Diffusion 보다 sample quality 약한 이유 중 하나.

---

## ⚖️ 가정과 한계

| 가정 | 한계 |
|------|------|
| Affine coupling 만 | 더 expressive transformation (NSF, Spline) 으로 확장 가능 |
| Same dim 전체 | Multi-scale 로 부분 완화 |
| Alternating mask | Image 의 경우 checkerboard/channel 필요 |
| Tanh clamping | Gradient saturation 위험 |
| Continuous data | Discrete 는 dequantization 필요 |
| Architecture-dependent expressiveness | Universal approximation 보장 없음 (RealNVP 자체로는) |

---

## 📌 핵심 정리

$$\boxed{y_A = x_A, \quad y_B = x_B \odot \exp(s(x_A)) + t(x_A)}$$

$$\boxed{\log |\det J| = \sum_{i \in B} s_i(x_A) \quad — O(|B|) \text{ 계산}}$$

| 특성 | 값 |
|------|------|
| **Invertibility** | Architecturally guaranteed |
| **Forward cost** | NN $s, t$ 1 forward pass |
| **Inverse cost** | NN $s, t$ 1 forward pass (same!) |
| **Jacobian det** | Triangular, $O(d)$ |
| **Volume preserving** | No (additive coupling 만 yes) |
| **Information bottleneck** | Half input unchanged per layer |

| 변형 | 차이 |
|------|------|
| **NICE** (Dinh 2015) | Additive only ($s = 0$), $|\det| = 1$ |
| **RealNVP** | Affine, expressive |
| **Glow** | + 1×1 conv, ActNorm |
| **NSF** (Durkan 2019) | Spline-based (more expressive coupling) |

---

## 🤔 생각해볼 문제

**문제 1** (기초): 2D coupling layer $f(x_1, x_2) = (x_1, x_2 \cdot \exp(s(x_1)) + t(x_1))$ 의 inverse 를 명시적으로 쓰고, $|\det J_f|$ 와 $|\det J_{f^{-1}}|$ 의 관계를 확인하라.

<details>
<summary>해설</summary>

**Inverse**:
$$f^{-1}(y_1, y_2) = (y_1, (y_2 - t(y_1)) \exp(-s(y_1)))$$

**$\det J_f$**:
$$J_f = \begin{pmatrix} 1 & 0 \\ \partial / \partial x_1 & \exp(s(x_1)) \end{pmatrix}$$

$\det J_f = \exp(s(x_1))$.

**$\det J_{f^{-1}}$**:
$$J_{f^{-1}} = \begin{pmatrix} 1 & 0 \\ \partial / \partial y_1 & \exp(-s(y_1)) \end{pmatrix}$$

$\det J_{f^{-1}} = \exp(-s(y_1)) = 1 / \exp(s(x_1)) = 1 / \det J_f$. ✓

**일반 정리**: $\det J_{f^{-1}}(f(z)) = 1 / \det J_f(z)$ — chain rule consequence.

</details>

**문제 2** (심화): Alternating mask 가 모든 dimension 을 결국 변형시킴을 보여라. 단일 mask 만 사용하면 무엇이 잘못되는가?

<details>
<summary>해설</summary>

**Single mask (e.g., 항상 first half = $A$)**:
- Layer 1: $y_A = x_A, y_B = T_1(x_B; x_A)$ — $x_A$ unchanged
- Layer 2: $z_A = y_A = x_A, z_B = T_2(y_B; y_A)$ — 여전히 $x_A$ unchanged
- ...
- 결국 first half 가 영원히 변형 안 됨 → $z_A = x_A$ — distribution 이 input 에 directly 의존

**Issue**: $p(z_A) = p(x_A)$ — base distribution 이 first half 를 그대로 받아야 함. 만약 $p_Z = \mathcal{N}(0, I)$ 라면 $x_A \sim \mathcal{N}(0, I)$ 이어야 함 — 데이터 분포의 first half 가 표준 정규여야 한다는 비현실적 제약.

**Alternating mask**:
- Layer 1: $A_1 = $ first half, $B_1 = $ second half — first half 가 conditioning, second half 변형
- Layer 2: $A_2 = $ second half, $B_2 = $ first half — second half 가 conditioning, first half 변형
- Layer 3: $A_3 = $ first half — second half 가 (이전 layer 에서 변형된 것을) conditioning, first half 추가 변형

**효과**: 모든 dim 이 결국 다른 dim 에 의존하는 transformation 통해 변형 → expressive enough to model arbitrary distributions (with sufficient depth).

**시사점**: alternating mask 는 architectural necessity, NICE/RealNVP 의 핵심 design.

</details>

**문제 3** (논문 비평): RealNVP 의 affine coupling 은 element-wise (diagonal Jacobian on $x_B$). NSF (Neural Spline Flow, Durkan 2019) 는 element-wise rational quadratic spline 으로 이를 확장. (i) NSF 가 affine 보다 더 expressive 한 이유, (ii) computational cost 의 변화를 논하라.

<details>
<summary>해설</summary>

**NSF (Rational Quadratic Spline)**:
- 각 element $x_{B, i}$ 를 monotonic spline $T_i(x_{B,i})$ 으로 변환
- Spline parameters $\{(\theta^k_i, \theta^k_{i+1}, \cdots)\}$ 가 $x_A$ 의 NN function

**(i) Expressiveness**:
- Affine: $y_{B,i} = x_{B,i} \cdot a_i + b_i$ — 각 dim 의 1D mapping 이 linear
- NSF: spline 으로 임의 monotonic 1D function — non-linear marginal modeling 가능

예: $x \sim \mathcal{N}(0, 1)$ 을 bimodal 분포로 변환:
- Affine 으로는 한 layer 에서 불가능 (linear)
- NSF spline 으로 한 layer 에서 가능 (non-linear monotone mapping)

따라서 **같은 layer 수로 더 표현력**, **더 적은 layer** 로 같은 표현력.

**(ii) Cost**:
- Affine: $s, t \in \mathbb{R}^{|B|}$ output, $|B|$ exp + $|B|$ add
- NSF: $K$ knots × $|B|$ output (for $K$-piece spline), spline forward/inverse 가 $O(\log K)$ binary search

추가 cost: spline parameter 수 $K$ 배, computation $O(\log K)$ 배. $K = 8$ 같은 작은 값에서도 큰 표현력 — overhead vs gain 좋음.

**Forward = Inverse cost**: spline 의 monotonicity 가 보장되면 element-wise inverse 도 binary search → $O(\log K)$.

**결과**:
- NSF 가 같은 NLL 을 더 적은 layer 로 (e.g., RealNVP 32 layer ≈ NSF 8 layer)
- Total compute 은 비슷하거나 작음
- Image NLL SOTA at the time of publication

**현대적 위치**: NSF 가 flow 의 표현력 frontier 를 증명 — "coupling layer 는 필연적으로 expressive 하지 못한 게 아님". 그러나 GAN/Diffusion 의 sample quality 에는 여전히 못 미침 — likelihood ≠ perceptual quality.

</details>

---

<div align="center">

[◀ 이전 (01. Change of Variables)](./01-change-of-variables.md) | [📚 README](../README.md) | [다음 ▶ (03. Glow)](./03-glow.md)

</div>
