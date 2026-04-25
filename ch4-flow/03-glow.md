# 03. Glow (Kingma & Dhariwal 2018)

## 🎯 핵심 질문

- $1 \times 1$ invertible convolution 이 무엇이고, 왜 RealNVP 의 fixed permutation 을 일반화하는가?
- $\log |\det J|$ 가 $H \cdot W \cdot \log |\det W|$ 인 이유는? Spatial 차원과 channel 차원의 분리?
- LU decomposition 으로 $1 \times 1$ conv 의 cost 를 $O(C^3)$ → $O(C)$ 로 줄이는 trick?
- ActNorm (Activation Normalization) 의 역할 — Batch Norm 과 어떻게 다른가?
- Glow block (ActNorm + 1×1 conv + Affine Coupling) 의 stack 이 어떻게 256×256 high-res 이미지 생성을 가능케 했는가?

---

## 🔍 왜 Glow 가 결정적 발전인가

Kingma & Dhariwal 2018 "Glow: Generative Flow with Invertible 1×1 Convolutions" 의 기여:

1. **$1 \times 1$ invertible convolution** — channel permutation 의 학습 가능 일반화
2. **Activation Normalization** — batch norm 의 invertible 대체
3. **고해상도 이미지 생성** — CelebA 256×256 의 첫 photorealistic flow generation
4. **Latent interpolation** — semantic manipulation 이 가능한 latent space

이전 RealNVP 는 **고정된** alternating mask. Glow 의 1×1 conv 는 매 layer 마다 channel ordering 을 학습 — 더 flexible, deeper model 에서 더 효과. 이는 후속 모든 image flow (Flow++, MaCow 등) 의 기반.

이 문서에서는 Glow 의 세 핵심 요소 (ActNorm, 1×1 conv, Affine Coupling) 의 수학과 architectural 의의를 다룹니다.

---

## 📐 수학적 선행 조건

- 이전 문서: 02-realnvp-coupling.md
- [Linear Algebra Deep Dive](https://github.com/iq-ai-lab/linear-algebra-deep-dive): LU decomposition, matrix determinant
- [CNN Deep Dive](https://github.com/iq-ai-lab/cnn-deep-dive): Convolution, batch normalization

---

## 📖 직관적 이해

### "RealNVP 의 한계: 고정된 permutation"

RealNVP coupling 은 alternating mask 사용 → channel 들이 항상 같은 순서로 split. 만약 channel 1 과 channel 5 의 dependency 가 강하면, 그 정보가 efficiently propagate 되지 못함 (mask 가 두 channel 을 분리).

**해결**: 매 layer 마다 channel ordering 변경 — permutation. 단순 permutation matrix 는 fixed, 학습 안 됨.

**Glow 의 통찰**: permutation 을 **학습 가능한 invertible $1 \times 1$ conv** 로 일반화. $W \in \mathbb{R}^{C \times C}$ matrix, 각 spatial 위치에 동일하게 적용. Permutation matrix 는 $W$ 의 특수한 경우.

### $1 \times 1$ Conv 의 Jacobian

$x \in \mathbb{R}^{H \times W \times C}$ 에 $W \in \mathbb{R}^{C \times C}$ 를 channel 차원으로 적용:

$$y_{i,j,:} = W \cdot x_{i,j,:} \quad \text{for each spatial } (i, j)$$

Jacobian $J \in \mathbb{R}^{HWC \times HWC}$ 가 block diagonal — 각 block 이 $W$:

$$J = \text{block-diag}(\underbrace{W, W, \ldots, W}_{HW \text{ blocks}})$$

$\det J = (\det W)^{HW}$, $\log |\det J| = HW \cdot \log |\det W|$.

따라서 $1 \times 1$ conv 의 contribution 은 spatial 크기 에 비례. $\log |\det W|$ 만 계산하면 됨 — $O(C^3)$.

### LU Decomposition Trick

$O(C^3)$ 도 매 forward pass 마다 reasonable 하지만, $W$ 를 LU decomposition 으로 parameterize:

$$W = P L (U + \text{diag}(s))$$

$P$: fixed permutation matrix.
$L$: lower triangular with 1's on diagonal — learnable lower entries.
$U$: upper triangular with 0's on diagonal — learnable upper entries.
$s \in \mathbb{R}^C$: log-diagonal of $U$ + diag.

그러면 $\det W = \prod s_i \cdot 1 \cdot \det P = \pm \prod \exp(s_i)$ — $O(C)$ 계산. Inversion 도 $O(C^2)$ via triangular solve.

### ActNorm 의 역할

Batch Norm 은 batch 의 통계 사용 → batch size 1 에서 부정확. Flow 의 invertibility 와 호환 어려움.

**ActNorm**: batch 의 첫 batch 통계로 초기화한 후, **per-channel scale + shift** parameter 로 만들어 학습 가능:

$$y_{:, :, c} = s_c \cdot x_{:, :, c} + b_c$$

$\log |\det| = HW \sum_c \log |s_c|$. 단순한 element-wise affine 이지만 image 의 각 channel 의 scale 정규화에 효과적.

---

## ✏️ 엄밀한 정의·정리

### 정의 3.1 — ActNorm

$x \in \mathbb{R}^{B \times C \times H \times W}$ 에 channel-wise affine:

$$y_{b, c, h, w} = s_c \cdot x_{b, c, h, w} + b_c$$

learnable $s, b \in \mathbb{R}^C$. Initialization: 첫 batch 의 per-channel mean/std 으로 $b = -\mu, s = 1/\sigma$ → 초기 output zero-mean unit-variance.

**Jacobian**: per spatial 위치마다 $\text{diag}(s)$. 전체:

$$\log |\det J| = HW \sum_{c=1}^C \log |s_c|$$

### 정의 3.2 — Invertible $1 \times 1$ Convolution

$x \in \mathbb{R}^{B \times C \times H \times W}$ 에 channel-mix matrix $W \in \mathbb{R}^{C \times C}$ ($\det W \neq 0$):

$$y_{b, :, h, w} = W \cdot x_{b, :, h, w}$$

**Jacobian**: 각 spatial 위치마다 $W$. 전체:

$$\log |\det J| = HW \cdot \log |\det W|$$

**Inverse**: $W^{-1}$ 적용 — 같은 형태.

### 정리 3.3 — LU Parameterization 의 Cost

$W = P L (U + \text{diag}(s))$ 에서:
- $P$ fixed (permutation)
- $L$: lower triangular, diagonal = 1 (learnable lower entries: $C(C-1)/2$ params)
- $U$: upper triangular, diagonal = 0 (learnable upper entries: $C(C-1)/2$ params)
- $s \in \mathbb{R}^C$: log of $U$'s diagonal

$$\det W = \det P \cdot \det L \cdot \det(U + \text{diag}(s)) = \pm 1 \cdot 1 \cdot \prod_i \exp(s_i)$$

$$\log |\det W| = \sum_i s_i$$

**Total params**: $C^2$ (same as full $W$). Computation:
- Forward $y = W x$: standard matmul, $O(C^2)$
- $\log |\det W|$: $O(C)$
- Inverse: $W^{-1} = (\text{diag}(s) + U)^{-1} L^{-1} P^\top$, 두 triangular solve, $O(C^2)$

### 정의 3.4 — Glow Block

각 block:
1. **ActNorm**
2. **Invertible 1×1 Conv** (LU parameterization)
3. **Affine Coupling** (RealNVP-style)

$L$ blocks stack. Squeezing + multi-scale 도 사용.

### 정리 3.5 — Glow 의 Total Log-Likelihood

$L$ Glow blocks, multi-scale level $S$:

$$\log p(x) = \sum_s \log p(z^{(s)}) + \sum_{l, s} (\log |\det J_{\text{ActNorm}}| + \log |\det J_{1 \times 1}| + \log |\det J_{\text{Coupling}}|)$$

= 합산 형태로 efficient computation.

### 정리 3.6 — 1×1 Conv 의 Permutation 일반화

Permutation matrix $P$: 각 row/column 에 단 하나의 1, 나머지 0. $\det P = \pm 1$.

Glow 의 학습 가능 $W$: 임의 invertible matrix. Permutation 의 superset.

**시사점**: RealNVP 의 fixed alternating mask 는 Glow 의 특수한 경우 (specific $W$). 학습된 $W$ 가 더 좋은 channel mixing 발견 → expressive.

---

## 🔬 증명 및 수학적 유도

### 유도 1 — $1 \times 1$ Conv 의 Block Diagonal Jacobian

$x \in \mathbb{R}^{C \times H \times W}$, vectorize 해서 $\tilde x \in \mathbb{R}^{CHW}$ (with stride: spatial $\times$ channel ordering).

$y = W x$ in channel dimension means:

$$\tilde y_{c, h, w} = \sum_{c'} W_{c, c'} \tilde x_{c', h, w}$$

Jacobian $\partial \tilde y_{c, h, w} / \partial \tilde x_{c', h', w'}$:

- $h = h', w = w'$: $W_{c, c'}$
- 다른 spatial: 0

따라서 spatial 차원으로 indexing 하면 block-diagonal — 각 block 이 $W \in \mathbb{R}^{C \times C}$.

$$\det J = \prod_{(h, w)} \det(W) = (\det W)^{HW}$$

$$\log |\det J| = HW \cdot \log |\det W|$$

### 유도 2 — LU Decomposition 의 정당성

임의 invertible $W \in \mathbb{R}^{C \times C}$ 는 $W = PLU'$ 로 분해 가능 ($P$ permutation, $L$ unit lower triangular, $U'$ upper triangular).

Glow 의 parameterization: $W = P L (U + \text{diag}(s))$ — $U$ 가 strict upper triangular (diagonal = 0), $\text{diag}(s) = $ exp 로 양수.

**왜 이 specific form?**:
- $P$ fixed → reparameterization 모호 제거
- $\det L = 1$ (unit diagonal) → $\det$ 의 $L$ 영향 없음
- $\det(U + \text{diag}(s)) = \prod_i \exp(s_i)$ — closed-form

따라서 학습 시 $L, U, s$ 만 update — 효율적.

### 유도 3 — ActNorm 이 Batch Norm 과 다른 점

**Batch Norm** (training):
$$y = (x - \mu_\text{batch}) / \sigma_\text{batch} \cdot \gamma + \beta$$

Batch dependent → invertibility 어려움 (다른 input batch 가 다른 transformation).

**ActNorm**:
$$y = s \cdot x + b$$

$s, b$ 가 batch-independent learnable parameter. **초기화** 만 batch-dependent — 첫 batch 로 mean=0, std=1 만들어 안정적 시작.

**장점**:
- Batch size 1 에서도 작동 (test-time, fine-tuning)
- Invertible (per element)
- Jacobian closed-form

**단점**: BN 의 normalization 효과 (data-adaptive) 부재 — 점차 학습으로 적응.

### 유도 4 — Multi-Scale 의 Image-Specific Effect

Multi-scale: 일부 channel 을 일찍 factor out (Gaussian latent), 나머지로 deeper layer.

**효과**:
- Coarse-to-fine: 일찍 factor out 된 channel 은 high-level structure (e.g., overall shape, color)
- Late layer: fine details (texture, edges)
- Hierarchical interpolation: 다른 scale 에서 다른 변화

$x \in \mathbb{R}^{H \times W \times C}$ 의 multi-scale flow:

```
Squeeze (H, W, C) → (H/2, W/2, 4C)
Glow blocks
Split: half → z₁ ~ N, half → continue
Squeeze
Glow blocks
Split: ...
```

각 scale 에서 latent $z^{(s)}$ 가 그 scale 의 information 표현.

### 유도 5 — High-Res Generation 의 Compute Requirements

256×256 image 에서:
- Multi-scale 4 levels: $256, 128, 64, 32$ resolution
- 각 level: 32 Glow blocks
- 총 32 × 4 = **128 invertible layers**

Compute: forward pass 가 매우 무거움 (deep model + per-layer NN computations). Training cost 가 GAN 대비 5-10 배.

**Trade-off**: exact likelihood + invertibility vs computational efficiency. Diffusion 이 같은 compute 으로 더 sharp samples — 이것이 Flow 가 image generation 의 mainstream 이 되지 못한 이유.

---

## 💻 실험 재현 / PyTorch 구현

### 실험 1 — ActNorm 구현

```python
import torch
import torch.nn as nn

class ActNorm(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.s = nn.Parameter(torch.zeros(1, channels, 1, 1))   # log scale
        self.b = nn.Parameter(torch.zeros(1, channels, 1, 1))
        self.initialized = False

    def forward(self, x):
        if not self.initialized:
            with torch.no_grad():
                # data-dependent initialization
                self.b.copy_(-x.mean([0, 2, 3], keepdim=True))
                self.s.copy_((-x.std([0, 2, 3], keepdim=True) + 1e-6).log())
            self.initialized = True
        s = self.s.exp()
        y = s * (x + self.b)
        # log_det per sample
        H, W = x.shape[-2:]
        log_det = self.s.sum() * H * W
        log_det = log_det.expand(x.shape[0])
        return y, log_det

    def inverse(self, y):
        s = self.s.exp()
        x = y / s - self.b
        H, W = y.shape[-2:]
        log_det = -self.s.sum() * H * W
        return x, log_det.expand(y.shape[0])
```

### 실험 2 — Invertible 1×1 Conv 구현 (LU)

```python
class Inv1x1Conv(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.C = channels
        # Random orthogonal initialization
        W = torch.linalg.qr(torch.randn(channels, channels))[0]
        # LU decomposition
        P, L, U = torch.linalg.lu(W)
        s = U.diagonal()
        U_strict = U.triu(1)   # remove diagonal
        L_strict = L.tril(-1)  # remove diagonal (which is 1)

        self.register_buffer('P', P)
        self.register_buffer('sign_s', torch.sign(s))
        self.register_buffer('mask_l', torch.tril(torch.ones_like(L), -1))
        self.register_buffer('mask_u', torch.triu(torch.ones_like(U), 1))
        self.L = nn.Parameter(L_strict)
        self.U = nn.Parameter(U_strict)
        self.log_s = nn.Parameter(s.abs().log())

    def get_W(self):
        L = self.L * self.mask_l + torch.eye(self.C, device=self.L.device)
        U = self.U * self.mask_u + torch.diag(self.sign_s * self.log_s.exp())
        return self.P @ L @ U

    def forward(self, x):
        W = self.get_W()
        # Apply per spatial position via 1x1 conv
        y = torch.nn.functional.conv2d(x, W.unsqueeze(-1).unsqueeze(-1))
        H, W_size = x.shape[-2:]
        log_det = self.log_s.sum() * H * W_size
        return y, log_det.expand(x.shape[0])

    def inverse(self, y):
        W = self.get_W()
        W_inv = torch.linalg.inv(W)
        x = torch.nn.functional.conv2d(y, W_inv.unsqueeze(-1).unsqueeze(-1))
        H, W_size = y.shape[-2:]
        log_det = -self.log_s.sum() * H * W_size
        return x, log_det.expand(y.shape[0])
```

### 실험 3 — Glow Block 통합

```python
class GlowBlock(nn.Module):
    def __init__(self, channels, hidden=128):
        super().__init__()
        self.actnorm = ActNorm(channels)
        self.inv1x1 = Inv1x1Conv(channels)
        # Affine coupling (channel half split after squeeze)
        self.coupling = ImageCoupling(channels, hidden)

    def forward(self, x):
        x, ld1 = self.actnorm(x)
        x, ld2 = self.inv1x1(x)
        x, ld3 = self.coupling(x)
        return x, ld1 + ld2 + ld3

    def inverse(self, y):
        y, ld3 = self.coupling.inverse(y)
        y, ld2 = self.inv1x1.inverse(y)
        y, ld1 = self.actnorm.inverse(y)
        return y, ld1 + ld2 + ld3
```

### 실험 4 — CIFAR-10 NLL 측정

```python
class Glow(nn.Module):
    def __init__(self, channels=3, n_blocks=32, hidden=512, n_levels=3):
        super().__init__()
        self.n_levels = n_levels
        # ... squeeze + Glow blocks + split, repeat ...

    def forward(self, x):
        log_det_total = 0
        z_list = []
        for level in range(self.n_levels):
            x = squeeze(x)
            for block in self.blocks[level]:
                x, ld = block(x)
                log_det_total += ld
            if level < self.n_levels - 1:
                x, z_split = split(x)
                z_list.append(z_split)
        z_list.append(x)
        return z_list, log_det_total

    def log_p(self, x):
        z_list, log_det = self.forward(x)
        log_p_z = sum(-0.5 * z.flatten(1).pow(2).sum(-1) for z in z_list)
        return log_p_z + log_det

# 학습 후 CIFAR-10 BPD 측정
# Glow 원 논문: 3.35 bpd, RealNVP 3.49 bpd 대비 개선
```

---

## 🔗 이론과 실전의 간극

### 1. Glow 의 Sample Quality

CelebA 256×256: photorealistic faces, latent interpolation 가능. 단, sharper 한 GAN (StyleGAN) 대비 약간 덜 sharp.

**왜**: Forward KL 의 mass-covering 성향 + coupling 의 architectural restriction.

### 2. Latent Space Interpolation

Glow 의 latent 가 의미 있는 manifold:
- $z_1 = f^{-1}(x_1), z_2 = f^{-1}(x_2)$
- $z(\alpha) = (1 - \alpha) z_1 + \alpha z_2$
- $x(\alpha) = f(z(\alpha))$ — semantic interpolation (smile, age, gender 등)

**Attribute manipulation**: smiling vs not-smiling 의 평균 latent 차이가 "smile direction". 이를 다른 face 에 더하면 smile.

이 latent manipulation 의 quality 가 Glow 의 큰 장점 — VAE 보다 sharp, GAN 보다 controllable.

### 3. 후속 발전

- **Flow++** (Ho 2019): Mixture of logistic CDF coupling, attention-augmented
- **MaCow** (Ma 2019): Masked convolutional flow
- **WaveGlow** (Prenger 2019): Audio Glow
- **Score-based** (Diffusion) 으로 image generation 의 mainstream 이동

---

## ⚖️ 가정과 한계

| 가정 | 한계 |
|------|------|
| 1×1 conv 가 channel mixing 충분 | Spatial mixing 은 coupling 에 의존 |
| LU parameterization 가 expressive | Permutation $P$ 가 fixed — 일부 mixing 제약 |
| ActNorm 이 BN 대체 | Data-adaptive normalization 효과 약함 |
| 깊은 stack 으로 expressiveness | 128+ layer 필요 — compute heavy |
| Continuous data | Image 에 dequantization 필요 |
| Same dim throughout | Compression 불가, large memory |

---

## 📌 핵심 정리

$$\boxed{\text{Glow Block} = \text{ActNorm} \to 1 \times 1 \text{ Conv} \to \text{Affine Coupling}}$$

$$\boxed{\log |\det J_{1 \times 1}| = HW \cdot \log |\det W|, \quad W = PL(U + \text{diag}(s))}$$

| 구성요소 | $\log |\det J|$ | Cost |
|---------|----------------|------|
| **ActNorm** | $HW \sum_c \log s_c$ | $O(C)$ |
| **1×1 Conv (LU)** | $HW \sum_c s_c$ | $O(C)$ |
| **Affine Coupling** | $\sum_i s_i$ | $O(\dim/2)$ |

| 비교 | RealNVP | Glow |
|------|---------|------|
| Channel mixing | Fixed alternating | Learned $1 \times 1$ conv |
| Normalization | None (or BN) | ActNorm |
| Image high-res | 64×64 어려움 | 256×256 가능 |
| Latent interpolation | OK | Better |

---

## 🤔 생각해볼 문제

**문제 1** (기초): $1 \times 1$ conv 가 단순 fully-connected layer 의 spatial-tiling 임을 보이고, 이것이 invertible 인 이유를 설명하라.

<details>
<summary>해설</summary>

**Spatial tiling**: $W \in \mathbb{R}^{C \times C}$ 가 channel 차원에 적용. 각 spatial 위치 $(h, w)$ 에서 $y_{:, h, w} = W x_{:, h, w}$ — same $W$ 가 모든 위치에 동일하게.

이는 Conv2d (kernel size 1, stride 1, padding 0) 와 동일. 따라서 PyTorch 에서 `nn.Conv2d(C, C, 1)` 로 구현.

**Invertibility**: 단일 $W$ 가 invertible ($\det W \neq 0$) 이면, **모든 spatial 위치의 transformation 이 invertible** — global invertibility 보장.

**Invertibility 보장 방법** (Glow): $W = P L (U + \text{diag}(s))$ with $s = e^{\hat s}$ (정수성 enforce). $\det W = \pm \prod \exp(\hat s_i) > 0$ — 항상 양수, 항상 invertible.

**시사점**: 1×1 conv 의 invertibility 가 spatial 차원과 분리되어 channel 차원의 단순 matrix invertibility 만 보장하면 됨. Architectural simplicity.

</details>

**문제 2** (심화): LU parameterization 에서 $P$ 가 학습 가능하면 어떤 문제가 생기는가? Glow 가 $P$ 를 fixed 로 한 이유는?

<details>
<summary>해설</summary>

**$P$ 가 학습 가능하면**:
1. **Discrete optimization**: permutation matrix 는 discrete (각 row/column 에 정확히 하나의 1) — gradient descent 직접 적용 불가
2. **Soft relaxation 필요**: doubly stochastic matrix 또는 Sinkhorn-Knopp 등으로 continuous approximation
3. **Multiple optima**: 다른 $P$ 가 다른 LU decomposition 줌, 학습 과정에서 jumping 가능

**Glow 의 선택 (fixed $P$)**:
- Random orthogonal initialization 으로 $W$ 결정
- $W$ 의 LU decomposition 한 번 계산 → $P, L, U, s$ 추출
- $P$ freeze, $L, U, s$ 만 학습

**이유**:
1. **Reparameterization 문제 회피**: $W = P L U'$ 와 $W = P' L' U''$ 가 같은 $W$ 줄 수 있음 — fixed $P$ 가 unique parameterization
2. **Optimization 안정**: continuous parameter 만 학습
3. **충분한 expressiveness**: $L, U, s$ 만으로도 $C^2$ degrees of freedom — full $W$ 의 표현력

**Trade-off**:
- ✅ Simple, stable
- ❌ $P$ 가 random 으로 fixed — initialization 에 의존, optimal 이 아닐 수 있음

**현대적 대안**:
- **Householder reflections**: orthogonal matrix 의 product 로 $W$ parameterize, $\det = \pm 1$
- **Cayley transform**: skew-symmetric → orthogonal mapping
- **Sinkhorn parameterization**: doubly stochastic → soft permutation

</details>

**문제 3** (논문 비평): Glow 가 256×256 photorealistic 얼굴 생성을 처음 달성한 flow 인데, StyleGAN 등 GAN 이 곧 더 sharp 한 결과 도달. Flow 가 sample quality 에서 GAN 에 못 미친 이유를 architectural + objective 측면에서 설명하라.

<details>
<summary>해설</summary>

**Architectural 측면**:

1. **Coupling 의 information bottleneck**: 절반 입력은 unchanged per layer. 깊은 stack 필요지만 non-linearity 가 layer 마다 절반만 영향 → effective depth 제한.

2. **Invertibility 강제**: GAN 의 generator 는 free-form (any NN), Flow 는 strict invertibility. 표현력 손실.

3. **Same dim throughout**: GAN 은 low-dim noise → high-dim image (compression 가능). Flow 는 same dim — 직접적 spatial 복잡도 모델링 어려움.

**Objective 측면**:

1. **Forward KL (mass-covering)**: 모든 mode cover 하려고 평균화. GAN 은 mode-seeking — sharp.

2. **Likelihood ≠ Perceptual quality** (Theis 2016): NLL 좋다고 sample quality 좋은 게 아님. Flow 는 NLL optimization, GAN 은 perceptual (discriminator).

3. **Gradient signal**: GAN 의 discriminator 가 perceptually-aligned gradient 제공. Flow 의 gradient 는 likelihood 의 average.

**Compute 측면**:

- GAN: 같은 capacity 로 더 효율 (no invertibility constraint)
- Flow: 깊이 + invertibility 로 더 많은 params 필요

**Glow 의 의의**:
- Flow 가 high-res generation 가능함을 증명 (StyleGAN 이전)
- Latent interpolation 의 controllability 강점
- Likelihood-critical 응용 (anomaly detection, scientific) 에서 우월

**현대적 위치**:
- Image generation main stream 은 GAN/Diffusion 으로 이동
- Flow 의 잔존 응용: density estimation, normalizing flow as prior (variational inference, Bayesian NN), molecular/protein generation
- Rectified Flow (Liu 2022): diffusion + flow hybrid 가 SD3 의 기반 — flow 의 부활

**시사점**: Architectural constraints 가 expressiveness 의 ceiling. Flow 의 elegant theory 가 sample quality 의 frontier 가 되지 못함. Diffusion 이 score-based + likelihood-bounded 로 두 패러다임의 장점 결합.

</details>

---

<div align="center">

[◀ 이전 (02. RealNVP)](./02-realnvp-coupling.md) | [📚 README](../README.md) | [다음 ▶ (04. MAF/IAF)](./04-maf-iaf.md)

</div>
