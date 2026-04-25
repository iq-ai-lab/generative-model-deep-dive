# 01. 5대 계보 통합 비교

## 🎯 핵심 질문

- AR · VAE · Flow · GAN · Diffusion 의 핵심 trade-off (likelihood, sampling speed, quality, stability) 를 어떻게 통합 비교하는가?
- 각 family 의 inductive bias 와 strengths/weaknesses 가 어떤 use case 에 fit 한가?
- "어떤 generative model 을 선택해야 하는가" 의 의사결정 트리 — 데이터 종류, 응용, 제약에 따른 분류?
- Hybrid approaches (VAE+Flow, Diffusion+GAN, AR+VQ-VAE) 가 왜 등장하고 어떤 advantages 를 갖는가?
- Diffusion 이 2020 이후 dominant 가 된 이유와, 그 이전 GAN-dominated era 의 차이?

---

## 🔍 왜 통합 비교가 결정적인가

각 family 를 깊이 본 후, **선택 기준** 이 명확해야 함:

1. **응용 별 맞춤** — image, audio, text, scientific 마다 다른 family 우월
2. **제약 별 맞춤** — compute, memory, latency, training data 에 따른 선택
3. **Hybrid 의 합리성** — 두 family 의 강점 결합으로 최적
4. **Frontier 위치** — 어떤 family 가 future-proof?

이 문서에서는 5 family 의 **systematic comparison**, **decision tree**, 그리고 **hybrid 접근** 을 다룹니다.

---

## 📐 수학적 선행 조건

- Ch1 ~ Ch6 모든 챕터
- 각 family 의 핵심 mechanism

---

## 📖 직관적 이해

### "도서관의 5가지 책 정리 방법"

각 family 를 "데이터 분포를 학습하는 5가지 방법" 으로:

**AR (Autoregressive)**: 한 단어씩 순서대로 읽기. 길지만 정확.

**VAE**: 책의 "요약" (latent) 만 외우고 다시 쓰기. 빠르지만 detail 손실.

**Flow**: 책을 invertible 변환으로 압축. 정확하지만 architecture 제약.

**GAN**: 모방 vs 감정의 게임으로 학습. Sharp 하지만 unstable.

**Diffusion**: 책에 점진적 noise 주입 → 점진적 제거로 학습. SOTA 이지만 slow.

### 5 Family 의 Trade-Off 표

| Family | Likelihood | Sampling Speed | Sample Quality | Training |
|--------|------------|----------------|----------------|----------|
| **AR** | Exact | Slow ($O(n)$ sequential) | Good | Stable |
| **VAE** | Lower bound (ELBO) | Fast | Blurry | Stable |
| **Flow** | Exact (change-of-vars) | Medium | Medium | Stable |
| **GAN** | None (implicit) | Fast | Sharp | Unstable |
| **Diffusion** | Lower (ELBO) + PF-ODE exact | Slow ($T$ steps) | SOTA | Stable |

각 family 가 **다른 측면 우월** — single "best" 없음.

### Decision Tree

"어떤 generative model 을 사용해야 하는가?":

1. **Modality?**
   - Text → AR (GPT-style)
   - Image → Diffusion or GAN
   - Audio → AR (WaveNet) or Flow
   - Tabular → Flow

2. **Likelihood needed (anomaly detection, density)?**
   - Yes → AR or Flow
   - No → GAN or Diffusion (Diffusion 이 NLL 도 reasonable)

3. **Sample quality critical?**
   - SOTA quality → Diffusion
   - Real-time + sharp → GAN
   - Reasonable + density → AR/Flow

4. **Compute constrained?**
   - Heavy → Diffusion 어려움 → smaller GAN
   - Free → Diffusion 의 SOTA

5. **Training stability?**
   - Stable training 필수 → Diffusion, AR, Flow, VAE
   - GAN 위험

---

## ✏️ 엄밀한 정의·정리 (Comparison Table)

### 정리 1.1 — 각 Family 의 Loss

| Family | Training Loss |
|--------|---------------|
| **AR** | $-\sum_i \log p_\theta(x_i \| x_{<i})$ |
| **VAE** | $-\text{ELBO} = -\mathbb{E}_q[\log p_\theta(x\|z)] + \text{KL}(q\|p(z))$ |
| **Flow** | $-\log p_Z(z) - \sum_l \log\|\det J_{f_l}\|$ |
| **GAN** | $\min_G \max_D \mathbb{E}_{p_d}[\log D] + \mathbb{E}_{p_g}[\log(1-D)]$ |
| **Diffusion** | $\mathbb{E}_{t, x_0, \epsilon}[\|\epsilon - \epsilon_\theta(x_t, t)\|^2]$ |

### 정리 1.2 — 각 Family 의 Sampling Cost

| Family | Cost | Note |
|--------|------|------|
| **AR** | $O(n)$ NN forward pass | $n$ = sequence length |
| **VAE** | $O(1)$ (single decoder pass) | Sample $z$, decode |
| **Flow** | $O(L)$ (layer count) | $L$ = flow depth |
| **GAN** | $O(1)$ (single generator pass) | Fastest |
| **Diffusion** | $O(T)$ NN passes | $T$ = denoising steps |

### 정의 1.3 — Quality Metric Comparison

CIFAR-10 benchmark (대표 결과, 2022 기준):

| Model | FID | NLL (bpd) | Notes |
|-------|-----|-----------|-------|
| PixelCNN++ (AR) | 50+ | 2.92 | Good NLL, poor FID |
| Glow (Flow) | 45 | 3.35 | Decent NLL, mediocre FID |
| VAE (best) | 30 | 3.5 | Both medium |
| StyleGAN2 | 3.0 | N/A | SOTA FID, no NLL |
| DDPM | 3.17 | 3.17 | Both excellent |
| EDM (Karras 2022) | 1.79 | ~3.0 | SOTA FID + good NLL |

### 정의 1.4 — Hybrid Architectures

**VAE + AR**: VQ-VAE (Ch3-05) — discrete latent + AR over latents (DALL-E)

**Flow + VAE**: VAE with Flow posterior (better $q$) — IAF posterior (Kingma 2016)

**Diffusion + AR**: Latent diffusion 위 AR 또는 Hybrid Diffusion-LM (Lin 2022)

**Diffusion + GAN**: Adversarial diffusion (Wang 2022) — diffusion 의 supervision + GAN's sharpness

**AR + Diffusion**: Autoregressive Diffusion (Hoogeboom 2022) — best of both

---

## 🔬 증명 및 수학적 유도

### 유도 1 — KL Minimization 의 통합 관점

모든 family 가 사실 $\min_\theta \text{KL}(p_d \| p_\theta)$ 의 다른 approximation:

| Family | KL Minimization Strategy |
|--------|--------------------------|
| **AR** | Direct: $\log p$ exact |
| **Flow** | Direct via change-of-variables |
| **VAE** | Lower bound (ELBO) |
| **Diffusion** | Weighted KL across noise levels |
| **GAN** | JSD (different metric, not KL) |

이것이 Ch1-03 의 unified view. WGAN 등은 Wasserstein metric — KL 이 아님.

### 유도 2 — Sample Quality 의 Source 분석

**Why GAN is sharp**:
- Adversarial training pushes mode-seeking
- Discriminator 가 perceptual gradient (semantic features) 제공
- Reverse KL 성향

**Why VAE is blurry**:
- Forward KL (mass-covering) → mode 평균
- Gaussian decoder $p(x|z)$ — 단일 Gaussian 의 한계
- Posterior collapse 가능

**Why Diffusion is SOTA**:
- Score-based training: 데이터 manifold 로의 강한 attraction
- Iterative refinement: cascading errors 회피
- Forward KL across noise levels: balanced

### 유도 3 — 왜 Diffusion 이 Dominant 가 되었는가

2020 이전: GAN dominant.
2020 이후: Diffusion takes over.

**Diffusion 의 advantages**:
1. **Stable training**: GAN 의 instability/mode collapse 회피
2. **Mode coverage**: forward KL 으로 모든 mode capture
3. **Likelihood-bounded**: ELBO 로 reasonable NLL
4. **Architectural freedom**: U-Net 자유 design
5. **Scaling**: 큰 모델 + 큰 데이터에서 잘 scale (DALL-E, Imagen)

**GAN 의 잔존 응용**:
- Real-time generation (single forward)
- Face editing (StyleGAN 의 latent space 의 controllability)
- Vocoder (HiFi-GAN)

### 유도 4 — Hybrid 의 정당성

**VQ-VAE + AR** (DALL-E):
- VAE: high-dim image → low-dim discrete tokens (compression)
- AR: token sequence 의 distribution learning (modeling)
- 둘 다의 strengths — compression + tractable likelihood

**Latent Diffusion** (Stable Diffusion):
- VAE: pixel → latent (compression, perceptual)
- Diffusion: latent space modeling
- Compute efficient + SOTA quality

**Score-SDE 의 unification**: Forward = Markov chain, Reverse = score-based ODE/SDE — diffusion 자체가 hybrid (chain + score).

---

## 💻 실험 재현 / PyTorch 구현

### 실험 1 — 5 Family 동시 비교 (Toy 1D)

```python
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

# Target: bimodal mixture
def sample_target(n):
    return torch.where(torch.rand(n) > 0.5,
                       torch.randn(n) * 0.5 + 2,
                       torch.randn(n) * 0.5 - 2)

# 5 family 의 simplified 1D implementations
# (full implementations in respective chapter files)

def evaluate_models():
    target = sample_target(2000).numpy()
    results = {}

    # AR (autoregressive 1D — trivial in 1D, just MLP density)
    # ... train AR1D ...

    # VAE
    # ... train VAE ...

    # Flow (RealNVP 1D)
    # ... train Flow ...

    # GAN
    # ... train GAN ...

    # Diffusion
    # ... train DDPM 1D (Ch6-02 의 example) ...

    # 각 family 의 sample 1000 개 생성, compare with target via:
    # - KS distance (distribution similarity)
    # - Wasserstein distance
    # - NLL (where applicable)

    return results

# 결과 (typical):
# AR: KS=0.05, NLL=-1.2, sample quality good
# VAE: KS=0.10, ELBO=-1.5, blurry
# Flow: KS=0.06, NLL=-1.3, decent
# GAN: KS=0.07, no NLL, sharp
# Diffusion: KS=0.04, ELBO=-1.25, best
```

### 실험 2 — Sampling Speed 비교

```python
import time

models = {
    'AR': ar_model,
    'VAE': vae_model,
    'Flow': flow_model,
    'GAN': gan_model,
    'Diffusion (DDPM)': ddpm_model,
    'Diffusion (DDIM 50)': ddim_model,
}

for name, model in models.items():
    t0 = time.time()
    samples = model.sample(1000)
    elapsed = time.time() - t0
    print(f"{name}: {1000/elapsed:.1f} samples/sec")

# 일반적 결과 (relative):
# GAN:           1000+ samples/sec
# VAE:           500+ samples/sec
# Flow:          200 samples/sec
# AR (sequential): 50 samples/sec
# DDPM (1000 steps): 1 sample/sec
# DDIM (50 steps):  20 samples/sec
```

### 실험 3 — Mode Coverage 비교

```python
# 8-Gaussian benchmark (Ch5-03)
def measure_coverage(samples, centers, threshold=0.3):
    coverage = 0
    for c in centers:
        d = np.linalg.norm(samples - c, axis=-1)
        if (d < threshold).any():
            coverage += 1
    return coverage / len(centers)

# 동일 데이터 (8-Gaussians) 에 5 family 학습 후 coverage 측정
# 일반:
# AR: 8/8 (perfect — chain rule guarantees)
# VAE: 7/8
# Flow: 7/8
# GAN (vanilla): 3-5/8 (mode collapse common)
# WGAN: 6-7/8
# Diffusion: 8/8
```

### 실험 4 — Hybrid Approaches

```python
# VQ-VAE + AR Transformer (mini DALL-E)
# 1. Train VQ-VAE on MNIST → discrete tokens
# 2. Train mini-GPT on token sequences
# 3. Sample: GPT generates tokens → VAE decoder produces image

# Latent Diffusion (mini Stable Diffusion)
# 1. Train autoencoder on MNIST
# 2. Train DDPM on latent space
# 3. Sample: DDPM in latent → decoder produces image

# Compare with pure approaches:
# - Computational cost
# - Final quality
# - Implementation complexity
```

---

## 🔗 이론과 실전의 간극

### 1. 응용 별 Family 선택

**Image Generation (high-res, photorealistic)**:
- Best: Diffusion (Stable Diffusion, DALL-E 3)
- Real-time: GAN (StyleGAN2)
- Open source: Stable Diffusion + community

**Audio Generation**:
- Speech: WaveNet (AR), HiFi-GAN (vocoder)
- Music: AudioLM, MusicLM (AR + tokens), Diffusion (Stable Audio)

**Text Generation**:
- LLM: AR (GPT, Claude, Gemini) — 압도적
- Diffusion-LM: experimental

**Scientific (molecules, proteins)**:
- AlphaFold 3: Flow Matching (related to Diffusion)
- Material discovery: Diffusion (MatterGen)

**Tabular Data**:
- Flow (CTGAN, TabDDPM)
- GAN (CTGAN)

### 2. Compute Constraints 의 영향

**Limited compute (smartphone, edge)**:
- GAN-based vocoder (real-time TTS)
- Distilled diffusion (Consistency Model, Ch7-02)
- Smaller AR models

**Cloud-scale compute**:
- Large diffusion (DALL-E 3, Imagen)
- Large LLMs (GPT-4)
- Cascaded diffusion (Imagen)

### 3. Diffusion 의 Speed Bottleneck

Diffusion 의 main weakness: $T = 50-1000$ steps. 해결책:
- **DDIM** (Song 2020): 50 steps
- **DPM-Solver** (Lu 2022): 10-20 steps
- **Consistency Models** (Song 2023): 1-4 steps
- **Rectified Flow** (Liu 2022): 1-step capable
- **Distillation** (Salimans 2022): 4 steps from teacher

이 trend 가 diffusion 의 sampling 을 GAN 수준 (single step) 으로 가속.

---

## ⚖️ 가정과 한계

| 가정 | 한계 |
|------|------|
| 5 family 가 mutually exclusive | Hybrid 가 흔함 |
| Single best per task | Domain-specific tuning 필요 |
| Diffusion 이 항상 best | Real-time 응용에서는 GAN 우월 |
| Likelihood = quality | NLL 과 FID 의 weak correlation (Theis 2016) |
| Architecture 가 family 결정 | Modern: 같은 arch 가 multiple loss 로 학습 가능 |

---

## 📌 핵심 정리

5 family 의 통합 비교:

| Aspect | AR | VAE | Flow | GAN | Diffusion |
|--------|-----|------|------|-----|-----------|
| **Likelihood** | Exact | Lower (ELBO) | Exact | None | Lower + PF-ODE exact |
| **Training** | Stable | Stable | Stable | Unstable | Stable |
| **Sampling Speed** | Slow | Fast | Medium | Fastest | Slow → Fast (Distill) |
| **Sample Quality** | Good | Blurry | Medium | Sharp | SOTA |
| **Architecture** | AR mask | Encoder-decoder | Invertible | Free | Free (U-Net) |
| **Mode Coverage** | Perfect | Good | Good | Poor | Excellent |
| **Compute** | Medium | Low | Medium | Low | High |
| **Use Cases** | Text, audio | Representation | Density estimation | Real-time, face | Image, video |

| Hybrid | Components | Advantage |
|--------|-----------|-----------|
| **VQ-VAE + AR** | VAE compress + AR model | Tractable + scalable |
| **Latent Diffusion** | AE compress + diffusion | Compute + quality |
| **Diffusion + GAN** | Diffusion supervised + GAN sharpen | Quality + speed |
| **Flow + VAE** | Flow posterior in VAE | Tighter ELBO |
| **AR + Diffusion** | AR for tokens + diffusion for refine | Multi-scale |

---

## 🤔 생각해볼 문제

**문제 1** (기초): 다음 응용에 가장 적합한 family 를 선택하고 이유를 설명하라:
- (a) Real-time face video editing
- (b) Drug molecule generation with property targeting
- (c) Code completion in IDE
- (d) Anomaly detection in network traffic

<details>
<summary>해설</summary>

**(a) Real-time face video editing**: **GAN (StyleGAN3)**.
- Real-time → single forward pass 필요
- Face quality → StyleGAN family 의 strength
- Latent space editing → StyleGAN 의 controllable manifold
- StyleGAN3 의 alias-free → motion equivariance

**(b) Drug molecule generation**: **Diffusion (e.g., RFdiffusion variant) + property guidance**.
- Quality + diversity 모두 중요 (drug variety)
- Property targeting: classifier guidance with target property
- Mode coverage 중요 (다양한 molecule types)
- Scientific 응용에서 Diffusion + flow matching 이 SOTA

**(c) Code completion**: **AR (LLM, GPT-style)**.
- Sequential text generation
- Causal mask for left-to-right
- Long-range context 필요
- Token-by-token sampling 자연스러움
- GitHub Copilot, Codex 가 이 형태

**(d) Anomaly detection**: **AR or Flow (likelihood-based)**.
- Likelihood evaluation 필수: $p_\theta(x_\text{test}) < \tau$ 로 anomaly 판단
- Diffusion (ELBO) 도 가능
- GAN 은 부적합 (no likelihood)
- Flow 가 exact NLL 으로 이상적

**일반 원칙**: task 의 핵심 requirement → family 의 strength 와 매칭. "Always Diffusion" 은 잘못된 default.

</details>

**문제 2** (심화): Diffusion 이 SOTA image generation 에서 GAN 을 대체했지만, real-time TTS 의 vocoder 에서는 GAN-based (HiFi-GAN) 이 여전히 표준이다. 이 분야 별 dominance 의 차이를 architectural 측면에서 분석하라.

<details>
<summary>해설</summary>

**Image Generation**:
- 응용: artistic creation, design, content (real-time 강제 없음)
- Quality > Speed
- Diffusion 의 50-step sampling 이 acceptable (GPU 1-5 sec)
- SOTA quality 가 결정적 — Diffusion 우월

**Real-time TTS Vocoder**:
- 응용: 음성 합성 (실시간 응답)
- Speed >> Quality (slight 차이는 acceptable)
- 16kHz audio = 16000 samples/sec 처리 필요
- Diffusion 50 step = 16000 × 50 = 800k NN evals/sec — prohibitive
- GAN single forward = 16000 evals/sec — possible

**HiFi-GAN 의 architectural choices**:
- Generator: dilated convolution (efficient)
- Discriminator: multi-scale (different time resolutions)
- Adversarial + perceptual + STFT loss
- Single forward → real-time

**Why not Diffusion for vocoder**:
- 16k samples × 50 steps = excessive
- DDIM/Distillation 으로 reduce 했지만 여전히 못 따름
- Audio 는 이미 sequence, no spatial complexity

**Trade-off**:
- Image: complex spatial patterns → diffusion 의 iterative refinement 가 quality
- Audio: temporal sequence + real-time → GAN 의 single-pass + WaveNet-style architecture

**Modern Direction**:
- Audio: Consistency Model 로 distilled diffusion (ConsistencyTTA)
- TTS: Stable Audio (latent diffusion) for non-realtime quality
- Hybrid: AR LLM + GAN vocoder (Tacotron 2 style 진화)

**시사점**: dominance 가 절대적 quality 가 아닌 **task-specific compute requirement** 에 의해 결정. Hardware constraint + latency requirement 가 architecture 선택의 driver. Generative model 의 "state of the art" 는 task dependent.

</details>

**문제 3** (논문 비평): Hybrid (VQ-VAE + AR, Latent Diffusion 등) 가 "best of both worlds" 라는 주장을 비판적으로 분석하라. 어떤 trade-off 가 있는가?

<details>
<summary>해설</summary>

**Pros of Hybrid**:

1. **VQ-VAE + AR (DALL-E)**:
   - VAE: spatial compression (224×224 → 32×32 tokens)
   - AR: long-range modeling on tokens
   - Result: text-to-image 가능, scale 가능

2. **Latent Diffusion (Stable Diffusion)**:
   - VAE: 48× compression
   - Diffusion: latent modeling
   - Result: 50× faster diffusion, GPU-friendly

**Cons / Trade-offs**:

1. **Two-stage Training Complexity**:
   - VAE 학습 → freeze → second stage 학습
   - VAE 의 quality 가 final quality 의 ceiling
   - End-to-end 학습 어려움 (gradient flow 제약)

2. **VAE Bottleneck**:
   - Compression 이 lossy
   - Fine details 가 reconstruction 시 잃을 수 있음
   - Stable Diffusion 의 face/text artifacts 의 source

3. **Latent Space Mismatch**:
   - VAE 의 latent distribution 이 diffusion 가정 (Gaussian) 과 mismatch 가능
   - VAE-VAE-Diffusion (LDM 의 KL regularization) 으로 mitigate

4. **Implementation Complexity**:
   - Two networks + interface
   - Pretrained VAE 의존
   - Custom debugging 어려움

5. **Inheritance of Each Family's Issues**:
   - VQ-VAE: codebook collapse 위험
   - AR: sequential sampling cost
   - Latent Diffusion: 여전히 50 step

**Specific Examples**:

**DALL-E (VQ-VAE + AR)**:
- ✅ First scalable text-to-image
- ❌ Token granularity 가 visual detail 의 ceiling
- ❌ AR sampling 느림

**Stable Diffusion (LDM)**:
- ✅ Open source feasibility
- ✅ Reasonable speed
- ❌ VAE-induced artifacts (eyes, hands, text in image)
- ❌ Two-stage 의 maintenance complexity

**Pure Approaches 의 Counter-arguments**:

- **Pixel Diffusion (Imagen, DALL-E 2)**: 더 많은 compute 으로 fine details 보존
- **Pure AR (Parti)**: tokenizer 의 quality 가 최고면 hybrid 와 비슷한 quality

**현대적 Trend**:
- **Diffusion Transformer (DiT)**: latent diffusion + Transformer — maintains hybrid
- **Flow Matching with continuous latents**: SD3, MovieGen — refined latent diffusion
- **Pure End-to-End**: Imagen 같은 cascaded 가 여전히 valid

**시사점**: hybrid 가 "free lunch" 아님 — 각 part 의 limitation 누적. Specific use case (compute, quality, latency) 에서 best, but not universally optimal. "Best of both worlds" 는 marketing — 실제는 "compromise of both, often acceptable".

**대안적 view**: hybrid 가 modern architecture 의 default 인 이유는 **각 part 의 specialization** — VAE 가 perception, AR/Diffusion 이 distribution modeling. 두 task 의 different inductive bias 를 분리하는 것이 효율적.

</details>

---

<div align="center">

[◀ 이전 (Ch6-05. CFG)](../ch6-diffusion/05-classifier-free-guidance.md) | [📚 README](../README.md) | [다음 ▶ (02. Consistency / Rectified Flow)](./02-consistency-rectified-flow.md)

</div>
