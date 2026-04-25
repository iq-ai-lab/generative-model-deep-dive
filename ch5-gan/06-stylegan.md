# 06. Progressive GAN · StyleGAN · StyleGAN2 (Karras)

## 🎯 핵심 질문

- Progressive Growing of GANs 가 어떻게 1024×1024 face 의 첫 photorealistic generation 을 가능케 했는가?
- StyleGAN 의 **mapping network** $z \to w$ 와 **AdaIN** style injection 이 disentanglement 를 어떻게 만드는가?
- $\text{AdaIN}(x, y) = \sigma(y) \cdot (x - \mu(x))/\sigma(x) + \mu(y)$ 의 정확한 의미와 style transfer 와의 관계?
- StyleGAN2 의 **weight demodulation** 이 droplet artifact 를 어떻게 제거하는가?
- StyleGAN3 의 alias-free design 이 motion 의 "텍스처 sticking" 문제를 어떻게 해결하는가?

---

## 🔍 왜 StyleGAN family 가 결정적인가

Karras et al. 의 NVIDIA 작업 — Progressive GAN (2018), StyleGAN (2019), StyleGAN2 (2020), StyleGAN3 (2021) — 가 GAN 의 photorealism 의 SOTA 를 정의:

1. **Progressive Growing**: 단계적 학습으로 stability + high-res
2. **Style-based Generator**: mapping network + AdaIN, semantic disentanglement
3. **Stochastic Variation**: per-pixel noise 로 fine detail
4. **Weight Demodulation** (StyleGAN2): artifact 제거
5. **Alias-Free Design** (StyleGAN3): 일관성 있는 motion

이 모델들의 face generation quality 가 거의 사진 수준 — FFHQ, AFHQ datasets. 후속 face editing, GAN inversion, image-to-image translation 의 기반. 이 문서에서는 이 family 의 architectural innovations 를 다룹니다.

---

## 📐 수학적 선행 조건

- 이전 문서들 (Ch5 전체)
- [CNN Deep Dive](https://github.com/iq-ai-lab/cnn-deep-dive): Conv, transposed conv, normalization
- Style transfer 의 기본 (AdaIN, Huang & Belongie 2017)

---

## 📖 직관적 이해

### "낮은 해상도부터 점진적 학습"

Progressive Growing (Karras 2018) 의 idea:

1. **4×4 image generation 학습**: small generator + small discriminator
2. **8×8 으로 grow**: 새 layer 추가 (fade in 으로 smooth transition)
3. **16×16, ..., 1024×1024**: 점진적 확장

**효과**:
- Low-res 에서 stable 하게 mode coverage 학습
- High-res 에서는 detail 만 추가
- Training time + GPU memory 효율

### "Style-based Generator: 노이즈가 아닌 style"

표준 GAN: $z \in \mathbb{R}^k$ 가 직접 generator 에 input, layer 마다 conv 통과.

StyleGAN: 
1. **Mapping network** $f: z \to w$ — 8-layer MLP, $z$ 를 disentangled $w$ 로 변환
2. **Synthesis network**: 고정된 learned constant $c \in \mathbb{R}^{4 \times 4 \times 512}$ 로 시작
3. **AdaIN style injection**: 각 layer 의 feature 를 $w$ 의 style 로 modulate
4. **Noise injection**: 각 layer 에 per-pixel noise 추가 (stochastic detail)

**Disentanglement**:
- Coarse $w$ (low-res layers): pose, identity
- Middle $w$: facial expression, hair style
- Fine $w$: micro-detail (skin, hair texture)

각 level 의 $w$ 를 다른 face 의 것과 swap → **style mixing**.

### AdaIN 의 의미

**Adaptive Instance Normalization** (Huang & Belongie 2017):

$$\text{AdaIN}(x, y) = \sigma(y) \cdot \frac{x - \mu(x)}{\sigma(x)} + \mu(y)$$

- $x$: content feature (synthesis network 의 layer output)
- $y$: style feature (mapping 된 $w$ 의 transformation)
- 효과: $x$ 의 statistics 를 $y$ 의 statistics (mean, std) 로 replace

이는 **style transfer** 의 핵심 — feature space 의 1st, 2nd order moment 가 style 을 capture.

---

## ✏️ 엄밀한 정의·정리

### 정의 6.1 — Progressive Growing

Generator $G$ 와 discriminator $D$ 가 점진적으로 growing:

- Iteration $T_1$: $G_1$ outputs $4 \times 4$ image, $D_1$ accepts $4 \times 4$
- Iteration $T_2$: $G_2 = G_1 + $ new layer (8×8 output), $D_2 = D_1 + $ new layer
- ... up to 1024×1024

**Fade-in**: 새 layer 가 add 될 때 $\alpha \in [0, 1]$ 로 점진적 weighting:

$$\text{output} = \alpha \cdot \text{new\_layer\_output} + (1 - \alpha) \cdot \text{upsampled\_old\_output}$$

$\alpha$ 가 0 → 1 점진적 증가.

### 정의 6.2 — Mapping Network (StyleGAN)

8-layer MLP $f: \mathbb{R}^{512} \to \mathbb{R}^{512}$:

$$w = f(z), \quad z \sim \mathcal{N}(0, I_{512})$$

$w$ 의 분포 $p_w$ 는 더 disentangled — manifold 의 axis 가 의미 있는 attribute 와 align.

### 정의 6.3 — AdaIN Style Injection

각 conv layer 의 feature $x \in \mathbb{R}^{H \times W \times C}$ 에 대해:

$$\text{AdaIN}(x_c, y_s, y_b) = y_s \cdot \frac{x_c - \mu_c(x)}{\sigma_c(x)} + y_b$$

여기서:
- $\mu_c(x), \sigma_c(x)$: $x$ 의 channel-wise mean, std (per spatial position)
- $y_s, y_b \in \mathbb{R}^C$: style scale and bias from $w$ via affine transformation

각 channel 별로 normalize 후 style 의 statistics 로 re-modulate.

### 정의 6.4 — Per-Pixel Noise Injection

각 conv layer output 에 spatial noise:

$$\text{out}_{i, j, c} = \text{conv\_out}_{i, j, c} + B_c \cdot N_{i, j}$$

여기서 $N_{i, j} \sim \mathcal{N}(0, 1)$ — per-pixel, $B_c$ — learned per-channel scale.

이로 stochastic detail (pore, hair strands, etc.) 이 latent space 와 분리되어 표현.

### 정의 6.5 — Weight Demodulation (StyleGAN2)

AdaIN 이 generate "droplet" artifact (blob in feature map). StyleGAN2 의 해결: AdaIN 을 weight modulation/demodulation 으로 대체.

**Modulation**: conv weight 에 style 적용:

$$w'_{ijk} = s_i \cdot w_{ijk}$$

(style scale $s_i$ 를 input channel 에 곱)

**Demodulation**: output statistics 정규화:

$$w''_{ijk} = w'_{ijk} / \sqrt{\sum_{i, k} (w'_{ijk})^2 + \epsilon}$$

이로 output 의 statistics 가 일정 — droplet 없음.

### 정의 6.6 — Alias-Free Design (StyleGAN3)

Conv + nonlinearity + downsample/upsample 의 chain 이 aliasing 유발:
- Activation 이 sharp transitions
- 인접 frequency 의 mixing
- Motion 시 "texture sticking" — pixel 위치에 attached features

**Solution**: 모든 operations 가 continuous (alias-free):
- Up/downsampling: ideal lowpass filter
- Activation: in continuous domain
- Translation equivariance 보장

이로 video/animation 에서 일관된 motion.

---

## 🔬 증명 및 수학적 유도

### 유도 1 — Progressive Growing 의 Stability Argument

저해상도 ($4 \times 4$): 16 pixels, easy distribution. $G$ 가 빠르게 mode coverage 달성.

고해상도로 grow: 새 detail 만 추가, low-res 의 structure 유지. **Local optimization** — mode coverage 안 깨짐.

**경험적**:
- 1024×1024 의 from-scratch GAN 학습: 거의 항상 실패
- Progressive: 첫 photorealistic 1024×1024 (Karras 2018)

### 유도 2 — Mapping Network 의 Disentanglement Effect

표준 generator: $z \sim \mathcal{N}(0, I)$ direct input → 데이터 manifold 가 $z$-space 에서 entangled.

**Mapping**: $w = f(z)$ — NN 이 데이터 distribution 에 맞게 $z$ 를 reshape.

**효과**:
- $w$ 의 manifold 가 데이터의 의미 있는 axis 와 align
- Latent traversal in $w$ 가 단일 attribute 변화 (e.g., age, smile)
- Linear interpolation in $w$ 가 자연스러운 transition

**Empirical** (Karras 2019): $w$-space 의 perceptual path length (PPL) 이 $z$-space 보다 작음 — smoother manifold.

### 유도 3 — AdaIN 의 Style-Content 분리

**Style transfer** (Gatys 2015): style = Gram matrix of features. AdaIN 은 더 단순:

$$\text{Style of } y = \{(\mu_c(y), \sigma_c(y))\}_c$$

(per-channel statistics — first 2 moments).

이를 다른 content $x$ 에 적용 = AdaIN.

**StyleGAN 의 활용**:
- Each layer 의 style = $w$ 의 affine transformation
- Coarse layer (low-res): structural style (pose)
- Fine layer (high-res): texture style (skin)
- 다른 layer 의 style 조합 = "style mixing"

### 유도 4 — Weight Demodulation 의 Math

AdaIN: feature $x$ 를 normalize → modulate. Issue: training 시 feature statistics 가 batch-dependent → instability.

**Weight modulation 등가**: 

$$x' = \text{AdaIN}(x, y_s, y_b) = y_s \cdot \frac{x - \mu(x)}{\sigma(x)} + y_b$$

다음 conv: $x' \to w_{ijk} * x'$. Distribute style:

$$\text{conv}(x', w) = w * (y_s \cdot x_\text{normalized} + y_b)$$

$$= (y_s \cdot w) * x_\text{normalized} + w * y_b$$

= conv with **modulated weight** $w'_{ijk} = y_{s, i} \cdot w_{ijk}$.

**Demodulation**: output 의 expected variance 가 일정 — modulation 의 amplification 을 normalize:

$$w''_{ijk} = w'_{ijk} / \sqrt{\text{expected output variance}}$$

이로 droplet artifact 의 root cause (instance normalization 의 spatial coupling) 제거.

### 유도 5 — Alias-Free Equivariance

Translation equivariance: $G(z, t = \text{shift}) = \text{shift}(G(z, t = 0))$.

표준 conv: integer-pixel translation 에 equivariant. 그러나 **subpixel translation** + activation/upsampling 이 violation.

**Alias-free** (StyleGAN3):
- Continuous representation 으로 all operations
- Lowpass filter before downsample (no aliasing)
- Activation in lifted (higher-frequency) domain

**효과**: motion video 에서 features 가 character 에 attached (not pixel).

---

## 💻 실험 재현 / PyTorch 구현

### 실험 1 — Mapping Network + AdaIN Block

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class MappingNetwork(nn.Module):
    def __init__(self, z_dim=512, w_dim=512, n_layers=8):
        super().__init__()
        self.net = nn.Sequential(*[
            nn.Sequential(nn.Linear(z_dim if i == 0 else w_dim, w_dim), nn.LeakyReLU(0.2))
            for i in range(n_layers)
        ])

    def forward(self, z):
        return self.net(z)

class AdaIN(nn.Module):
    def __init__(self, channels, w_dim=512):
        super().__init__()
        self.norm = nn.InstanceNorm2d(channels, affine=False)
        self.style_scale = nn.Linear(w_dim, channels)
        self.style_bias = nn.Linear(w_dim, channels)

    def forward(self, x, w):
        x_norm = self.norm(x)
        scale = self.style_scale(w).unsqueeze(-1).unsqueeze(-1)
        bias = self.style_bias(w).unsqueeze(-1).unsqueeze(-1)
        return scale * x_norm + bias

class NoiseInjection(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.weight = nn.Parameter(torch.zeros(1, channels, 1, 1))

    def forward(self, x, noise=None):
        if noise is None:
            noise = torch.randn(x.size(0), 1, x.size(2), x.size(3), device=x.device)
        return x + self.weight * noise

class StyleGANBlock(nn.Module):
    def __init__(self, in_ch, out_ch, w_dim=512):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, 3, padding=1)
        self.noise = NoiseInjection(out_ch)
        self.adain = AdaIN(out_ch, w_dim)
        self.act = nn.LeakyReLU(0.2)

    def forward(self, x, w):
        x = self.conv(x)
        x = self.noise(x)
        x = self.act(x)
        x = self.adain(x, w)
        return x

# Mini StyleGAN-like generator
class StyleGAN(nn.Module):
    def __init__(self, z_dim=512, w_dim=512, n_layers=8):
        super().__init__()
        self.mapping = MappingNetwork(z_dim, w_dim, n_layers)
        # Constant input
        self.const = nn.Parameter(torch.randn(1, 512, 4, 4))
        # Synthesis blocks (4x4 → ... → 256x256, simplified)
        self.blocks = nn.ModuleList([
            StyleGANBlock(512, 512),
            StyleGANBlock(512, 256),
            StyleGANBlock(256, 128),
            StyleGANBlock(128, 64),
        ])
        self.to_rgb = nn.Conv2d(64, 3, 1)

    def forward(self, z):
        w = self.mapping(z)
        x = self.const.expand(z.size(0), -1, -1, -1)
        for block in self.blocks:
            x = F.interpolate(x, scale_factor=2, mode='bilinear')
            x = block(x, w)
        return torch.tanh(self.to_rgb(x))
```

### 실험 2 — Style Mixing Demonstration

```python
@torch.no_grad()
def style_mixing(model, z1, z2, mix_layer):
    """Layer < mix_layer 에서 z1 의 style, layer >= mix_layer 에서 z2 의 style"""
    w1 = model.mapping(z1)
    w2 = model.mapping(z2)
    x = model.const.expand(z1.size(0), -1, -1, -1)
    for i, block in enumerate(model.blocks):
        w = w1 if i < mix_layer else w2
        x = F.interpolate(x, scale_factor=2, mode='bilinear')
        x = block(x, w)
    return torch.tanh(model.to_rgb(x))

# Coarse style mixing (mix_layer = 1): pose 만 z1, 나머지 z2 → 다른 사람의 pose
# Fine style mixing (mix_layer = 3): texture 만 z2, 나머지 z1 → 같은 사람 다른 texture
```

### 실험 3 — Truncation Trick (Sample Quality vs Diversity)

```python
@torch.no_grad()
def truncated_sample(model, n=10, psi=0.7):
    """w 를 mean toward truncate"""
    z_avg = torch.randn(10000, 512)
    w_avg = model.mapping(z_avg).mean(0, keepdim=True)
    z = torch.randn(n, 512)
    w = model.mapping(z)
    w_truncated = w_avg + psi * (w - w_avg)   # Move towards mean
    # ... rest of forward with w_truncated ...

# psi = 1.0: full diversity
# psi = 0.5: high quality, low diversity
# psi = 0: average face
```

### 실험 4 — Latent Space Interpolation

```python
@torch.no_grad()
def latent_interp(model, z1, z2, n=10):
    alphas = torch.linspace(0, 1, n)
    images = []
    for a in alphas:
        z = (1 - a) * z1 + a * z2
        # Interpolate in z (or w) — w is smoother
        w = (1 - a) * model.mapping(z1) + a * model.mapping(z2)
        # ... synthesis with w ...
        images.append(...)
    return torch.cat(images)

# z-interpolation: 일부 jumpy
# w-interpolation: smoother (mapping network 의 효과)
```

---

## 🔗 이론과 실전의 간극

### 1. StyleGAN 의 Face Generation Dominance

FFHQ (Flickr-Faces-HQ) 1024×1024 에서 StyleGAN2 의 FID < 2.0. Diffusion 기반 face generation (이후 등장) 도 비슷한 quality 이지만 sampling 더 느림. **Real-time face editing, identity preservation** 에서 StyleGAN family 가 여전히 dominant.

### 2. GAN Inversion

학습된 StyleGAN 의 latent space 에 real image 를 invert: $x_\text{real} \to z$ such that $G(z) \approx x_\text{real}$.

**방법**:
- Optimization-based: $\arg\min_z \|G(z) - x_\text{real}\|$
- Encoder-based: $E_\phi(x) = z$ 학습
- Hybrid: encoder + per-image fine-tuning

이를 통해 real face 의 attribute editing (age, smile, glasses, etc.) 가능.

### 3. Diffusion 시대의 StyleGAN

Diffusion 이 image generation 의 mainstream 이지만 StyleGAN family 가 여전히 강세인 영역:
- **Real-time generation**: 단일 forward (vs diffusion 의 50+ steps)
- **Controllable editing**: latent space 의 명확한 structure
- **GAN inversion**: image-to-latent 의 mature methods

**Hybrid**: StyleGAN-Diffusion 의 결합 — diffusion 의 sample quality + GAN 의 controllability.

---

## ⚖️ 가정과 한계

| 가정 | 한계 |
|------|------|
| Progressive growing 이 stability | High-res 에서 detail 추가 어려움 |
| Mapping network 가 disentanglement | $w$ 가 의미 있는 axis 와 align 보장 없음 |
| AdaIN 이 style-content 분리 | First 2 moments 만 — limited "style" 정의 |
| Per-pixel noise 가 stochastic | Spatial structure 약함 |
| Face dataset 의 specialization | 다른 domain (animal, scene) 에서는 일반화 한계 |
| Single-image generation | Video, 3D 에는 다른 architecture 필요 (StyleGAN-V) |

---

## 📌 핵심 정리

$$\boxed{\text{AdaIN}(x, y) = \sigma(y) \cdot \frac{x - \mu(x)}{\sigma(x)} + \mu(y)}$$

$$\boxed{\text{StyleGAN: } z \to^\text{Mapping} w \to^\text{AdaIN} \text{synthesis}}$$

| 모델 | 핵심 기여 |
|------|----------|
| **Progressive GAN** (2018) | 점진적 학습, 1024×1024 first |
| **StyleGAN** (2019) | Mapping network + AdaIN, disentanglement |
| **StyleGAN2** (2020) | Weight demodulation, no droplet |
| **StyleGAN3** (2021) | Alias-free, motion equivariance |

| Component | Function |
|-----------|----------|
| **Mapping** $z \to w$ | Disentangled latent |
| **Constant input** | Synthesis 의 시작점 |
| **AdaIN** | Style injection per layer |
| **Noise injection** | Stochastic fine detail |
| **Style mixing** | 다른 face 의 layer 별 style 조합 |
| **Truncation** | Quality vs diversity trade-off |

---

## 🤔 생각해볼 문제

**문제 1** (기초): AdaIN 이 single-channel feature $x \in \mathbb{R}^{H \times W}$ 에 어떻게 적용되는지 단계별로 설명하라. Spatial structure 가 유지되는가?

<details>
<summary>해설</summary>

**Step 1**: $\mu(x), \sigma(x)$ 계산 — single channel 의 spatial mean, std.

**Step 2**: $x_\text{norm} = (x - \mu(x)) / \sigma(x)$ — zero mean, unit variance.

**Step 3**: Style $y_s, y_b$ from $w$ — scalars (single channel).

**Step 4**: $\text{AdaIN}(x) = y_s \cdot x_\text{norm} + y_b$.

**Spatial structure**:
- Normalize 단계: $x$ 의 spatial pattern 유지 (subtraction + scaling 은 spatial)
- $x_\text{norm}$ 이 같은 spatial pattern, 다른 statistics
- $y_s, y_b$ 가 spatial uniform (단일 scalar)
- 따라서 AdaIN output 이 spatial pattern 유지, statistics 만 style 의 것

**시사점**: AdaIN 의 "style" = first 2 moments (mean, std). Spatial **content** (어디에 무엇이 있는지) 는 $x$ 에서 유지. **Style transfer** 의 핵심 idea — content 와 style 의 분리.

</details>

**문제 2** (심화): StyleGAN 의 mapping network 가 8 layer MLP 인 이유 — 왜 그렇게 깊은가? 깊이가 disentanglement 에 어떤 역할?

<details>
<summary>해설</summary>

**$z \to w$ 의 목적**: $z \sim \mathcal{N}(0, I)$ (factored Gaussian) 의 distribution 을 데이터 분포에 적합한 $w$-space 로 변환.

**얕은 mapping (e.g., 1 layer)**: 거의 linear, $w$ 의 manifold 가 $z$ 의 transformation. 데이터의 nonlinear factor 표현 못 함.

**깊은 mapping (8 layer)**:
1. **Nonlinear distribution shaping**: 데이터 distribution 의 axis 와 align
2. **Disentanglement**: layer 가 깊을수록 의미 있는 disentangled axis 학습 가능
3. **Smooth manifold**: $w$-space 의 perceptual path length 가 $z$ 의 1/3 정도

**Empirical** (Karras 2019):
- 1 layer: PPL = 230 (jagged manifold)
- 8 layer: PPL = 80 (smooth)
- 16 layer: PPL = 75 (marginal improvement)

**Cost**:
- Mapping 의 8 layer × 512 dim 은 작은 cost (synthesis 가 main)
- 학습 시간 약간 증가, inference 무시 가능

**왜 정확히 8?**: 실험적 — 4-16 사이가 모두 비슷 우월. 8 이 sweet spot.

**시사점**: depth 가 representation 의 quality. 단순 linear projection 으로는 disentanglement 부족. 학습된 deep nonlinear mapping 이 useful semantic structure 발견.

</details>

**문제 3** (논문 비평): StyleGAN3 가 motion equivariance 위해 alias-free design 을 도입했다. 이것이 video generation 에서 어떤 problem 을 해결하는가? 표준 StyleGAN2 video 의 limitation 은?

<details>
<summary>해설</summary>

**StyleGAN2 의 video 문제 — "Texture Sticking"**:
- Frame 간 latent 변화 (e.g., $w_t \to w_{t+1}$) 시 character 가 움직임
- 그러나 fine textures (skin pore, hair, fabric) 가 pixel 위치에 stuck
- 결과: character 가 움직여도 pore 는 같은 pixel 좌표에 남음 — unnatural

**원인**:
- Conv + activation + upsampling chain 이 **aliasing** 유발
- High-frequency texture 가 spatial grid 에 attached
- Latent 변화가 character 의 좌표 이동을 유도하지만 texture 는 grid 에 종속

**Alias-Free Design** (StyleGAN3):
1. **Continuous-time representation**: 모든 features 가 continuous function (sampled at grid)
2. **Ideal lowpass filter** 로 down/upsample: aliasing 차단
3. **Lifted activation**: nonlinearity 가 continuous domain 에서

**효과**: 모든 transformation 이 translation-equivariant. Latent 변화가 character 의 움직임 유발 → texture 가 character 와 함께 이동.

**Result**:
- Smooth video (interpolation between frames)
- Animatable face (latent 의 expression axis)
- 3D-aware generation 의 foundation

**Cost**:
- 학습 시간 증가
- Architecture 복잡 (continuous filtering)
- Slightly worse FID for static images (lower expressiveness due to constraints)

**Modern Status**:
- StyleGAN3 가 face animation, video editing 의 foundation
- Combined with Neural Radiance Fields (NeRF) for 3D-aware
- Diffusion video models (Sora, etc.) 가 다른 접근

**시사점**: theoretical principle (translation equivariance) 의 architectural enforcement 가 practical generation quality 향상. "Aliasing" 같은 signal processing concept 가 deep learning 의 detail. CNN Deep Dive (Chapter 1) 의 equivariance 논의와 연결.

</details>

---

<div align="center">

[◀ 이전 (05. Spectral Norm)](./05-spectral-norm.md) | [📚 README](../README.md) | [다음 ▶ (Ch6-01. DDPM)](../ch6-diffusion/01-ddpm-forward-reverse.md)

</div>
