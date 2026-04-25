# 01. Change of Variables 와 Flow 의 정의

## 🎯 핵심 질문

- Change of variables 공식 $p_X(x) = p_Z(f^{-1}(x)) |\det J_{f^{-1}}(x)|$ 는 어떤 조건 하에서 성립하는가? (Invertibility, $C^1$, 등)
- 왜 Jacobian determinant 의 절댓값이 등장하는가? Sign 은 어떻게 처리되는가?
- 체인 $f = f_L \circ \cdots \circ f_1$ 의 log-density 가 $\log p_Z(z) - \sum_l \log |\det J_{f_l}|$ 가 되는 이유는?
- "Invertibility + tractable Jacobian determinant" 의 이중 제약이 왜 architecture 에 strong 한 inductive bias 가 되는가?
- Flow 가 exact likelihood 를 제공하는데도 왜 GAN/Diffusion 보다 sample quality 가 떨어지는가?

---

## 🔍 왜 Change of Variables 가 결정적인가

Normalizing Flow 의 모든 것이 **change of variables 공식** 위에 구축됩니다:

$$p_X(x) = p_Z(f^{-1}(x)) \cdot |\det J_{f^{-1}}(x)|$$

이 공식이 우리에게 주는 것:

1. **Exact likelihood** — VAE 의 ELBO (lower bound) 와 다름, 정확한 $\log p(x)$
2. **Sampling trivial** — base distribution $p_Z$ 에서 sampling, $f$ 적용
3. **Latent inference exact** — $z = f^{-1}(x)$ deterministic
4. **Invertibility 의 architectural constraint** — coupling layer, 1×1 conv 등의 design

이 공식의 **조건** 과 **체인 적용** 이 flow 의 핵심 수학. 이 문서에서는:

- Change of variables 의 엄밀한 statement 와 증명
- 다양한 $f$ family 의 Jacobian 분석
- 체인 적용으로 deep flow 구축
- "왜 invertibility + tractable det 이 어려운가" 의 architectural 함의

를 다룹니다.

---

## 📐 수학적 선행 조건

- [Probability Theory Deep Dive](https://github.com/iq-ai-lab/probability-theory-deep-dive): Probability density, distribution
- [Linear Algebra Deep Dive](https://github.com/iq-ai-lab/linear-algebra-deep-dive): Determinant, Jacobian, eigendecomposition
- [Calculus Deep Dive](https://github.com/iq-ai-lab/calculus-deep-dive): Multivariate change of variables in integrals

---

## 📖 직관적 이해

### "확률 질량 보존"

연속 확률 변수의 변환 $x = f(z)$ 에서 핵심 직관:

**전체 확률 = 1** 이 보존되어야 함. $f$ 가 공간을 stretching/compressing → density 가 inversely scale.

1D 예: $z \sim \mathcal{N}(0, 1)$, $x = 2z$. $x$ 의 분포는 $\mathcal{N}(0, 4)$ — variance 가 4배, density 가 1/2 배 (peak 절반). Density 의 스케일링 factor = $|dz/dx| = 1/2$.

2D 예: $f$ 가 면적을 2배 expand → 같은 영역의 density 가 1/2. Factor = $1 / |\det J_f|$.

### Jacobian Determinant 의 기하학적 의미

$f: \mathbb{R}^d \to \mathbb{R}^d$ 의 Jacobian $J_f \in \mathbb{R}^{d \times d}$. $|\det J_f|$ 는:

- **점 $z$ 근방의 작은 cube 가 $f$ 에 의해 변환되면, 그 부피 변화의 비율**
- $|\det J_f| = 1$: volume preserving (예: rotation)
- $|\det J_f| > 1$: expansion
- $|\det J_f| < 1$: contraction

따라서 $p_X(x) = p_Z(z) / |\det J_f(z)|$ — base density 를 volume change rate 로 나눔.

### 왜 "Tractable" Jacobian 이 어려운가

$|\det J_f|$ 의 직접 계산: $O(d^3)$ — Gaussian elimination 또는 LU decomposition. $d = 1000$ 이면 $10^9$ ops, 가능하지만 매 forward pass 마다 prohibitive.

해결책 (모두 architecture 에 제약):
- **Triangular Jacobian** (RealNVP coupling): $\det = \prod_i J_{ii}$, $O(d)$
- **Autoregressive** (MAF, IAF): triangular 와 유사
- **Restricted form**: orthogonal, low-rank correction

→ Flow 의 design 은 항상 "**invertible + efficient Jacobian computation**" 의 trade-off.

---

## ✏️ 엄밀한 정의·정리

### 정리 1.1 — Change of Variables (Multivariate)

$f: U \to V$ ($U, V \subseteq \mathbb{R}^d$ open) 가 $C^1$ diffeomorphism (i.e., $f$ 가 bijective, $f$ 와 $f^{-1}$ 모두 $C^1$). $z \in U$ 가 density $p_Z(z)$ 의 random variable, $x = f(z)$. 그러면 $x$ 의 density:

$$p_X(x) = p_Z(f^{-1}(x)) \cdot |\det J_{f^{-1}}(x)|$$

또는 $z = f^{-1}(x)$ 로 표현:

$$p_X(x) = \frac{p_Z(z)}{|\det J_f(z)|}$$

**조건**:
- $f$ bijective (invertible)
- $f, f^{-1}$ 모두 $C^1$ (smooth)
- $\det J_f \neq 0$ everywhere (locally invertible)

### 증명 (정리 1.1)

확률 질량 보존: $V \subseteq U$ 의 measurable subset $A$ 에 대해

$$\Pr(z \in A) = \Pr(f(z) \in f(A)) = \Pr(x \in f(A))$$

밀도로:

$$\int_A p_Z(z) dz = \int_{f(A)} p_X(x) dx$$

좌변에 변수 변환 $x = f(z)$:

$$\int_A p_Z(z) dz = \int_{f(A)} p_Z(f^{-1}(x)) \cdot |\det J_{f^{-1}}(x)| dx$$

(multivariate calculus 의 change of variables in integrals — Jacobian 의 역수 또는 inverse 의 Jacobian)

따라서:

$$\int_{f(A)} p_X(x) dx = \int_{f(A)} p_Z(f^{-1}(x)) |\det J_{f^{-1}}(x)| dx$$

$f(A)$ 가 임의이므로 integrand 가 일치:

$$p_X(x) = p_Z(f^{-1}(x)) |\det J_{f^{-1}}(x)| \quad \square$$

### 정리 1.2 — Jacobian Inverse

$$J_{f^{-1}}(x) = (J_f(z))^{-1}, \quad z = f^{-1}(x)$$

따라서:

$$|\det J_{f^{-1}}(x)| = \frac{1}{|\det J_f(z)|}$$

대입하면:

$$p_X(x) = \frac{p_Z(z)}{|\det J_f(z)|}$$

두 형태가 동등.

### 정리 1.3 — 체인 적용 (Composition)

$f = f_L \circ f_{L-1} \circ \cdots \circ f_1$ ($f_l: \mathbb{R}^d \to \mathbb{R}^d$ 모두 diffeomorphism). 중간 변수 $z_0 = z, z_l = f_l(z_{l-1}), z_L = x$. 그러면:

$$\log p_X(x) = \log p_Z(z_0) - \sum_{l=1}^L \log |\det J_{f_l}(z_{l-1})|$$

**증명**: 단일 $f$ 에 대해 정리 1.1. 체인은 $J_f = J_{f_L} \cdots J_{f_1}$ (chain rule of Jacobians), $\det J_f = \prod_l \det J_{f_l}$. Log: $\log |\det J_f| = \sum_l \log |\det J_{f_l}|$. 대입. $\square$

### 정의 1.4 — Normalizing Flow

Base distribution $p_Z$ (보통 $\mathcal{N}(0, I)$) 와 invertible parameterized maps $\{f_{\theta_l}\}_{l=1}^L$ 의 composition $f_\theta = f_{\theta_L} \circ \cdots \circ f_{\theta_1}$ 로:

$$x = f_\theta(z), \quad z \sim p_Z$$

학습:

$$\hat\theta = \arg\max_\theta \mathbb{E}_{x \sim p_d}\left[\log p_Z(f^{-1}_\theta(x)) - \sum_{l=1}^L \log |\det J_{f_{\theta_l}}|\right]$$

즉 **MLE 직접** — VAE 의 ELBO 와 다름, exact likelihood.

### 정리 1.5 — Architecture 제약

Flow 의 architectural choice 는 다음을 모두 만족해야:

1. **Invertibility**: $f^{-1}$ closed-form 또는 numerical efficient
2. **Tractable Jacobian determinant**: $\log |\det J_f|$ 가 $O(d^k)$ for small $k$
3. **Expressive**: 충분히 다양한 $f$ family

이 셋의 양립 가능한 architectures:
- **Triangular**: coupling, autoregressive — $\det = \prod J_{ii}$
- **Orthogonal**: rotation 등 — $|\det| = 1$
- **Low-rank corrections**: $f(z) = z + UV^\top z$ — Sherman-Morrison

Compromise 가 항상 존재 — 이것이 Flow 가 GAN/Diffusion 만큼 expressive 하지 못한 이유.

---

## 🔬 증명 및 수학적 유도

### 유도 1 — 1D Change of Variables 의 Sanity Check

$z \sim \mathcal{N}(0, 1)$, $f(z) = az + b$. 그러면 $x = az + b \sim \mathcal{N}(b, a^2)$ (linear Gaussian).

Verify:
- $f^{-1}(x) = (x - b)/a$
- $J_f(z) = a$, $|J_f| = |a|$

$$p_X(x) = \frac{p_Z((x-b)/a)}{|a|} = \frac{1}{|a| \sqrt{2\pi}} e^{-(x-b)^2 / 2a^2}$$

= $\mathcal{N}(b, a^2)$ density. ✓

### 유도 2 — 2D 회전이 Density 변화 안 시킴

$f: \mathbb{R}^2 \to \mathbb{R}^2$, $f(z) = R z$, $R$ rotation matrix ($R^\top R = I, \det R = 1$).

$J_f = R$, $|\det R| = 1$. $p_X(x) = p_Z(R^\top x) / 1 = p_Z(R^\top x)$ — base distribution 의 회전된 버전.

**시사점**: orthogonal $f$ 는 volume-preserving — density 가 단순히 변형, scaling 없음. Glow 의 1×1 conv 가 이 성질 활용.

### 유도 3 — Triangular Jacobian 의 Determinant

$J \in \mathbb{R}^{d \times d}$ 가 lower triangular ($J_{ij} = 0$ for $i < j$):

$$\det J = \prod_{i=1}^d J_{ii}$$

**증명**: triangular 행렬의 determinant 가 diagonal product 임은 LU decomposition 의 직접적 결과. $\square$

이 사실이 **coupling layer** (Ch4-02) 와 **autoregressive flow** (Ch4-04) 의 핵심.

### 유도 4 — Volume 변화의 직접 측정

$f: \mathbb{R}^d \to \mathbb{R}^d$, $z$ 근방의 작은 cube $[z, z + dz_1] \times \cdots \times [z, z + dz_d]$. $f$ 적용 후 변환된 region 의 부피:

$$\text{vol}(f(\text{cube})) \approx |\det J_f(z)| \cdot \text{vol}(\text{cube})$$

이는 multivariate calculus 의 기본 — Jacobian 의 기하학적 의미.

따라서 density 보존 = 부피 변화 보상:

$$p_X(x) \cdot \text{vol}(f(\text{cube})) = p_Z(z) \cdot \text{vol}(\text{cube})$$

$$p_X(x) = p_Z(z) / |\det J_f(z)|$$

### 유도 5 — Composition 의 Jacobian Chain Rule

$f = g \circ h$, $w = h(z), x = g(w) = f(z)$. Chain rule:

$$J_f(z) = J_g(w) \cdot J_h(z)$$

Determinant:

$$\det J_f(z) = \det J_g(w) \cdot \det J_h(z)$$

Log (절댓값):

$$\log |\det J_f| = \log |\det J_g| + \log |\det J_h|$$

귀납적으로 $f = f_L \circ \cdots \circ f_1$:

$$\log |\det J_f(z_0)| = \sum_l \log |\det J_{f_l}(z_{l-1})|$$

이로 정리 1.3 의 체인 공식 $\square$.

---

## 💻 실험 재현 / PyTorch 구현

### 실험 1 — 1D Linear Flow

```python
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

class LinearFlow1D(nn.Module):
    """x = a * z + b, learnable a, b"""
    def __init__(self):
        super().__init__()
        self.log_a = nn.Parameter(torch.zeros(1))
        self.b = nn.Parameter(torch.zeros(1))

    def forward(self, z):
        a = self.log_a.exp()
        x = a * z + self.b
        log_det = self.log_a   # log |a|
        return x, log_det

    def inverse(self, x):
        a = self.log_a.exp()
        z = (x - self.b) / a
        return z, -self.log_a   # log |a^{-1}|

def log_p_normal(x):
    return -0.5 * x.pow(2) - 0.5 * np.log(2 * np.pi)

# Target: bimodal mixture (1D)
def sample_target(n):
    return torch.where(torch.rand(n) > 0.5,
                       torch.randn(n) * 0.5 + 2,
                       torch.randn(n) * 0.5 - 2)

# 단일 linear flow 는 unimodal 만 표현 가능 — 다음 실험에서 확장
flow = LinearFlow1D()
opt = torch.optim.Adam(flow.parameters(), lr=1e-2)

for step in range(2000):
    x = sample_target(256)
    z, log_det = flow.inverse(x)
    log_p_x = log_p_normal(z) + log_det
    loss = -log_p_x.mean()
    opt.zero_grad(); loss.backward(); opt.step()

print(f"Learned a = {flow.log_a.exp().item():.2f}, b = {flow.b.item():.2f}")
# Linear flow 는 mean=0, var=4.25 (=2.06^2, includes mode separation) 정도로 fit
# Bimodal 은 표현 못 함 — 더 expressive flow 필요
```

### 실험 2 — Composition 의 효과

```python
class CompositionFlow1D(nn.Module):
    """다중 piecewise linear (또는 다른 simple) flows 의 composition"""
    def __init__(self, n_layers=4):
        super().__init__()
        # 각 layer: x = (z + a_l) * exp(s_l)  ← scale + shift
        self.scales = nn.Parameter(torch.zeros(n_layers))
        self.shifts = nn.Parameter(torch.zeros(n_layers))

    def forward(self, z):
        log_det_total = torch.zeros_like(z)
        for s, t in zip(self.scales, self.shifts):
            z = (z + t) * s.exp()
            log_det_total += s
        return z, log_det_total

    def inverse(self, x):
        log_det_total = torch.zeros_like(x)
        for s, t in zip(reversed(self.scales), reversed(self.shifts)):
            x = x * (-s).exp() - t
            log_det_total += -s
        return x, log_det_total

# Composition of linear maps 는 여전히 linear → bimodal 표현 불가
# Non-linear flow (next chapters) 필요
```

### 실험 3 — Jacobian Determinant 의 수치 검증

```python
# Triangular Jacobian: det = product of diagonals
def random_triangular_jacobian(d=10):
    J = torch.tril(torch.randn(d, d))
    # diagonal = exp(s) for stability
    J.diagonal().fill_(1.0)
    s = torch.randn(d)
    J.diagonal().copy_(s.exp())
    return J, s

J, s = random_triangular_jacobian(d=20)
det_direct = torch.linalg.det(J)
det_diag = s.exp().prod()
print(f"Direct det:  {det_direct.item():.4f}")
print(f"Prod diag:   {det_diag.item():.4f}")
# 두 값이 일치 — triangular 의 핵심 성질

# log |det| computation
log_det_direct = torch.linalg.slogdet(J)[1]
log_det_sum = s.sum()
print(f"log|det| direct: {log_det_direct.item():.4f}")
print(f"sum s:           {log_det_sum.item():.4f}")
```

### 실험 4 — 2D Flow 시각화

```python
class Affine2D(nn.Module):
    """x = A z + b, learnable 2x2 A, 2-vec b"""
    def __init__(self):
        super().__init__()
        self.A = nn.Parameter(torch.eye(2))
        self.b = nn.Parameter(torch.zeros(2))

    def forward(self, z):
        x = z @ self.A.T + self.b
        log_det = torch.linalg.slogdet(self.A)[1]
        return x, log_det.expand(z.shape[0])

# 두 Gaussian mixture 학습 시도 (여전히 표현력 제한)
# 진짜 expressive flow 는 coupling layer, ch4-02
```

---

## 🔗 이론과 실전의 간극

### 1. Flow 의 Architectural Constraint

Universal approximation 이론상 — bijective NN 으로 임의 분포 근사 가능하지만 (Huang 2018 NAF 의 결과), **practical** 한 architecture 는 모두 제약:

- Coupling: 절반 입력은 그대로 (information bottleneck)
- Autoregressive: sequential, parallelization 어려움
- Continuous (Neural ODE): training cost 큼

따라서 Flow 의 expressiveness 가 GAN/Diffusion 보다 부족 → image quality 손해.

### 2. "Latent" 의 의미 차이

VAE 의 latent: stochastic, low-dim, semantic.
Flow 의 latent: **deterministic** ($z = f^{-1}(x)$ exact), **same dim as $x$**, less semantic.

따라서 Flow 의 representation learning 은 제한적 — 대부분 density estimation 에 사용.

### 3. Numerical Stability

$\log |\det J|$ 가 $-\infty$ 또는 $+\infty$ 로 발산하는 region: numerical 어려움. Stable parameterization (예: $\sigma > 0$ via exp, sigmoid bounded outputs) 이 design 의 큰 부분.

---

## ⚖️ 가정과 한계

| 가정 | 한계 |
|------|------|
| $f$ bijective + $C^1$ | NN 으로 만족시키기 어려움 — special architecture |
| Tractable $\log \|\det J\|$ | $O(d)$ for triangular, $O(d^3)$ general |
| Same dim $z$ and $x$ | Compression 불가능 — VAE/AE 의 dimensionality reduction 못 함 |
| Continuous data | Discrete data 에는 dequantization 필요 |
| Architecture-dependent expressiveness | GAN/Diffusion 만큼 sharp samples 어려움 |

---

## 📌 핵심 정리

$$\boxed{p_X(x) = p_Z(f^{-1}(x)) \cdot |\det J_{f^{-1}}(x)| = \frac{p_Z(z)}{|\det J_f(z)|}}$$

$$\boxed{f = f_L \circ \cdots \circ f_1 \Rightarrow \log p_X(x) = \log p_Z(z_0) - \sum_l \log |\det J_{f_l}(z_{l-1})|}$$

| 조건 | 의미 |
|------|------|
| **Bijective** | $f$ invertible — exact $z = f^{-1}(x)$ |
| **$C^1$** | Smooth — Jacobian well-defined |
| **$\det J \neq 0$** | Locally invertible everywhere |
| **Tractable det** | $O(d)$ 이상이면 prohibitive — architecture 제약 |

| Architecture | $\det J$ 계산 | 제약 |
|-------------|--------------|------|
| **Coupling (RealNVP)** | Triangular, $\prod_i J_{ii}$, $O(d)$ | 절반 입력 변경 안 됨 |
| **Autoregressive (MAF, IAF)** | Triangular, $O(d)$ | Sequential |
| **Orthogonal (Glow 1×1)** | $|\det| = 1$ | Volume preserving |
| **Continuous (FFJORD)** | Trace integral, Hutchinson $O(d)$ | ODE solve cost |
| **General linear** | $O(d^3)$ | Prohibitive |

---

## 🤔 생각해볼 문제

**문제 1** (기초): $z \sim \mathcal{N}(0, 1)$, $f(z) = z^2$. $x = f(z)$ 의 분포는?

<details>
<summary>해설</summary>

$f(z) = z^2$ 는 **bijective 가 아님** ($f(-1) = f(1) = 1$). 따라서 standard change of variables 직접 적용 불가.

**올바른 방법**: $z$ 의 두 branch 합산.

$F_X(x) = \Pr(Z^2 \leq x) = \Pr(-\sqrt x \leq Z \leq \sqrt x) = \Phi(\sqrt x) - \Phi(-\sqrt x) = 2\Phi(\sqrt x) - 1$ (for $x \geq 0$)

$p_X(x) = F_X'(x) = \phi(\sqrt x) / \sqrt x$ — **Chi-squared with 1 dof**.

**시사점**: change of variables 는 invertible $f$ 만 직접 적용. Non-invertible 은 sum over preimages:

$$p_X(x) = \sum_{z: f(z) = x} \frac{p_Z(z)}{|f'(z)|}$$

Flow 가 invertible 한 architectural restriction 의 이유.

</details>

**문제 2** (심화): $f: \mathbb{R}^2 \to \mathbb{R}^2$, $f(z_1, z_2) = (z_1, z_2 + g(z_1))$ ($g$ 임의 NN). 이 $f$ 가 bijective 임을 보이고, $|\det J_f|$ 를 계산하라. (이것이 RealNVP coupling 의 단순 형태)

<details>
<summary>해설</summary>

**Bijective 증명**: 역함수 명시.

$f^{-1}(x_1, x_2) = (x_1, x_2 - g(x_1))$. 확인: $f(f^{-1}(x)) = (x_1, x_2 - g(x_1) + g(x_1)) = x$. ✓

**Jacobian**:
$$J_f = \begin{pmatrix} \partial x_1 / \partial z_1 & \partial x_1 / \partial z_2 \\ \partial x_2 / \partial z_1 & \partial x_2 / \partial z_2 \end{pmatrix} = \begin{pmatrix} 1 & 0 \\ g'(z_1) & 1 \end{pmatrix}$$

**Lower triangular** with diagonal (1, 1).

$\det J_f = 1 \cdot 1 = 1$. $|\det J_f| = 1$ — **volume preserving**.

**시사점**:
- 이 simple coupling 은 $z_1$ 그대로, $z_2$ 만 변형
- Jacobian 이 항상 1 — log det = 0
- 그러나 표현력 제한 — $x_1 = z_1$ 이므로
- RealNVP 의 affine coupling (Ch4-02) 가 이를 확장: $x_2 = z_2 \cdot \exp(s(z_1)) + t(z_1)$, $|\det| = \prod \exp(s_i)$

</details>

**문제 3** (논문 비평): Flow 가 exact likelihood 를 제공함에도 ImageNet 등에서 sample quality 가 GAN/Diffusion 에 못 미치는 이유를 architectural 측면에서 논하라.

<details>
<summary>해설</summary>

**1. Architectural Restriction**:
- Coupling layer: 절반 입력은 unchanged, 표현력 손실
- Same dim $z, x$: bottleneck (compression) 불가능 → semantic abstraction 못 함
- Invertibility 강제: 자유로운 architecture 선택 불가

**2. Likelihood vs Sample Quality 의 불일치 (Theis 2016)**:
- NLL 은 mode coverage 측정
- Sample quality 는 perceptual sharpness 측정
- Flow 의 mass-covering training 이 mode 평균화 → blurry

**3. Inductive Bias 부족**:
- GAN: discriminator 가 "real-like" 의 inductive bias 제공 (perceptual)
- Diffusion: forward/reverse process 의 score-based inductive bias
- Flow: change of variables formula 만 — generic

**4. Computational Cost**:
- 같은 capacity 의 GAN/Diffusion 보다 Flow 가 더 깊은 모델 필요
- 100+ coupling layers 흔함 — training/inference cost
- 같은 시간 budget 에서 GAN/Diffusion 이 더 큰 효과

**5. Continuous Normalizing Flow (FFJORD) 의 시도**:
- Free-form Jacobian (no coupling restriction)
- 이론상 더 expressive
- 하지만 ODE solving cost, training instability
- Diffusion 이 동일한 SDE framework 에서 더 잘 작동

**현대적 위치**:
- Pure Flow: density estimation (anomaly detection, scientific) 의 niche
- Latent Flow (Stable Diffusion 의 일부 component): 보조 역할
- Rectified Flow (Liu 2022): diffusion + flow hybrid 가 SD3 의 기반

**시사점**: "Exact likelihood" 가 항상 "best generative model" 을 의미하지 않음. Architecture 의 inductive bias 와 perceptual training signal 이 sample quality 의 결정적 요인. Flow 는 likelihood-critical 응용에서만 우월.

</details>

---

<div align="center">

[◀ 이전 (Ch3-05. VQ-VAE)](../ch3-vae/05-vq-vae.md) | [📚 README](../README.md) | [다음 ▶ (02. RealNVP)](./02-realnvp-coupling.md)

</div>
