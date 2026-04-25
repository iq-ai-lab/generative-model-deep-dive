# 04. Frontier — Multimodal · Video · 3D · 과학

## 🎯 핵심 질문

- DALL-E 3, Imagen, Stable Diffusion 3 의 architectural choices — text encoder, diffusion structure, sampling 의 차이?
- Sora 의 Diffusion Transformer (DiT) 가 video 의 시공간을 어떻게 모델링하는가?
- NeRF + DreamFusion (SDS loss) 가 3D generation 을 가능케 하는 mechanism?
- AlphaFold 3 의 Flow Matching 이 단백질 구조 예측에 어떻게 적용되는가?
- Material discovery, drug design 등 과학적 응용에서 generative model 의 역할?

---

## 🔍 Frontier 의 의의

5 family 의 통합 후, **현재 활발한 연구 영역**:

1. **Multimodal**: text + image + audio + video 의 unified generation
2. **Video**: temporal coherence + spatial detail
3. **3D**: NeRF-based + diffusion 의 결합
4. **Science**: protein structure, molecules, materials
5. **Embodied**: robotics, simulation

각 영역의 generative model architecture 가 fundamentally 다른 inductive bias. 이 문서에서는 주요 frontier 영역의 개요를 다룹니다.

---

## 📐 수학적 선행 조건

- 모든 이전 챕터
- [Transformer Deep Dive](https://github.com/iq-ai-lab/transformer-deep-dive): Attention, scaling
- [SDE Deep Dive](https://github.com/iq-ai-lab/sde-deep-dive): Continuous-time process

---

## 📖 직관적 이해

### "한 모델이 모든 modality"

Text, image, audio, video 가 **통합된 representation** 으로 처리:

1. **Tokenization**: 각 modality 를 token sequence 로 (text: BPE, image: VQ-VAE, audio: EnCodec)
2. **Unified Transformer**: 모든 token 을 한 sequence 로
3. **Cross-modal generation**: text → image, audio → video 등

### 4가지 Frontier 영역

| 영역 | 대표 모델 | Architecture |
|------|----------|-------------|
| **Image** | DALL-E 3, SD3, Imagen | Latent diffusion + CFG |
| **Video** | Sora, Pika, Runway | DiT on video latent |
| **3D** | DreamFusion, Gaussian Splatting | NeRF + SDS, point clouds |
| **Science** | AlphaFold 3, RFdiffusion | Flow matching, SE(3) equivariance |

---

## ✏️ Architectures (Modern)

### 정의 4.1 — Stable Diffusion 3 (Esser 2024)

**Components**:
- **VAE**: 1024×1024 image → 128×128 latent (8× compression)
- **MMDiT** (Multi-Modal Diffusion Transformer): text + image tokens 함께
- **Rectified Flow**: $v_\theta(x_t, t, y)$ 학습
- **Three text encoders**: CLIP-G, CLIP-L, T5-XXL

**Training**:
$$\mathcal{L} = \mathbb{E}_{t, x_0, \epsilon, y}[\|v_\theta(x_t, t, y) - (\epsilon - x_0)\|^2]$$

**Sampling**: 28-step Rectified Flow + CFG.

### 정의 4.2 — Sora (OpenAI 2024)

**Architecture** (publicly known):
- **3D Patch Tokenization**: video → space-time patches (VAE 의 video version)
- **Diffusion Transformer**: large-scale Transformer on patches
- **Long-context**: minute-long video (수십 second)

**Key innovations**:
- Spacetime patches (2D image patches + temporal extension)
- Variable resolution / aspect ratio handling
- Long-form coherence (consistent character, scene)

### 정의 4.3 — DreamFusion (Poole 2023, SDS Loss)

3D generation from text:

1. **NeRF**: 3D scene representation $\sigma(x, y, z), c(x, y, z)$ — radiance field
2. **Score Distillation Sampling (SDS)**:

$$\nabla_\theta \mathcal{L}_\text{SDS} = \mathbb{E}_{t, \epsilon, c}[w(t)(\epsilon_\phi(x_t; y, t) - \epsilon) \frac{\partial x}{\partial \theta}]$$

**Idea**: pretrained 2D diffusion 의 score 를 NeRF 의 rendered image 에 적용 → 3D structure 가 2D consistency 를 만족하도록.

3. **Optimization**: NeRF 의 parameter $\theta$ 를 SDS gradient 로 update.

### 정의 4.4 — AlphaFold 3 (Abramson 2024)

**Architecture**:
- **MSA + Templates**: evolutionary information
- **Diffusion (Flow Matching) on atom positions**: structure generation
- **SE(3) Equivariance**: rotation-translation invariance

**Loss**: Flow Matching on protein atom coordinates with biochemical constraints.

**Result**: 단백질 + 단백질 complex, RNA, ligand 등 통합 prediction.

### 정의 4.5 — Gaussian Splatting (Kerbl 2023)

3D representation 의 alternative to NeRF:
- Scene = collection of 3D Gaussians (position, scale, color, opacity)
- Differentiable rasterization
- Real-time rendering

**Generative version**: text-to-3D 가 빠르고 high-quality.

---

## 🔬 주요 통찰

### 통찰 1 — Multimodal Tokenization

Modality 별 tokenizer:
- Text: BPE (50K vocab)
- Image: VQ-VAE (8K vocab) 또는 continuous latent (Stable Diffusion)
- Audio: EnCodec (1K-16K)
- Video: 2D image tokens 의 temporal extension

**Unified token space**: cross-modal model 이 single vocabulary 로 처리.

### 통찰 2 — Diffusion Transformer (DiT, Peebles 2023)

**기존 U-Net** (Stable Diffusion): convolutional, image-specific.

**DiT**: Transformer on image patches. Advantages:
- **Scaling**: Transformer 가 잘 scale (model + data)
- **Modality-agnostic**: video (3D patches), audio 에 직접 확장 가능
- **Long-range**: attention 이 global dependencies 처리

**Sora** 가 DiT-based — 이 architectural choice 가 large-scale video 를 가능케 함.

### 통찰 3 — SDS (Score Distillation Sampling)

DreamFusion 의 핵심 trick: 2D diffusion 의 score 를 3D 로 distill.

**기존 problem**: 3D dataset 이 부족 (NeRF 등은 single scene).

**SDS solution**: 풍부한 2D 데이터로 학습된 diffusion 을 multi-view consistency 의 score 로.

NeRF 의 rendered image 가 "real image-like" 가 되도록 → 3D 가 자연스러움.

**한계**: 종종 "Janus problem" (multi-face 등 view inconsistency), 느린 optimization.

### 통찰 4 — Equivariance in Science

**Protein**: 분자가 rotation/translation 에 invariant — physical 법칙.

**SE(3)-Equivariant Networks**: input 의 rotation 이 output 의 rotation 으로 → physical consistency.

**EquiformerV2, AlphaFold 3 의 IPA module**: SE(3) equivariant attention.

이로 분자 generation 이 inherent symmetry 보존.

### 통찰 5 — Flow Matching for Molecules

**Flow Matching** (Lipman 2023): Rectified Flow 의 generalization.

$x_t = \alpha_t x_0 + \beta_t x_1$ (general interpolation, $x_0 = $ noise, $x_1 = $ data).

NN 이 vector field 학습. 다양한 path 가능 (linear, optimal transport, etc.).

**AlphaFold 3** 가 이를 채택한 이유:
- Flexible path design (biochemical 적절)
- Stable training
- Fast sampling

---

## 💻 실험 / 응용

### 실험 1 — Stable Diffusion 사용

```python
from diffusers import StableDiffusion3Pipeline
import torch

pipe = StableDiffusion3Pipeline.from_pretrained(
    "stabilityai/stable-diffusion-3-medium-diffusers",
    torch_dtype=torch.float16
).to("cuda")

image = pipe(
    "A photo of a cat playing chess",
    num_inference_steps=28,
    guidance_scale=7.0
).images[0]
image.save("cat_chess.png")
```

### 실험 2 — DiT Toy Implementation

```python
import torch
import torch.nn as nn

class DiTBlock(nn.Module):
    def __init__(self, hidden=384, n_heads=6):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden)
        self.attn = nn.MultiheadAttention(hidden, n_heads, batch_first=True)
        self.norm2 = nn.LayerNorm(hidden)
        self.mlp = nn.Sequential(
            nn.Linear(hidden, 4 * hidden), nn.GELU(),
            nn.Linear(4 * hidden, hidden),
        )
        # Adaptive Layer Norm with timestep + class condition
        self.ada_ln = nn.Linear(hidden, 6 * hidden)

    def forward(self, x, c):
        # c: condition (timestep + class) embedding
        ada = self.ada_ln(c)
        scale_msa, shift_msa, gate_msa, scale_mlp, shift_mlp, gate_mlp = ada.chunk(6, -1)
        # AdaLN modulation
        x = x + gate_msa.unsqueeze(1) * self.attn(
            modulate(self.norm1(x), scale_msa, shift_msa),
            modulate(self.norm1(x), scale_msa, shift_msa),
            modulate(self.norm1(x), scale_msa, shift_msa)
        )[0]
        x = x + gate_mlp.unsqueeze(1) * self.mlp(modulate(self.norm2(x), scale_mlp, shift_mlp))
        return x

def modulate(x, scale, shift):
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)
```

### 실험 3 — DreamFusion-style SDS Loss

```python
def sds_loss(diffusion_model, nerf_image, prompt_emb, t):
    """SDS gradient 계산"""
    eps = torch.randn_like(nerf_image)
    x_t = diffusion_q_sample(nerf_image, t, eps)
    eps_pred = diffusion_model(x_t, t, prompt_emb)
    # SDS gradient
    grad = (eps_pred - eps) * w(t)
    # Apply to NeRF parameters via chain rule
    nerf_image.backward(grad)
```

### 실험 4 — Protein Flow Matching

```python
# Simplified flow matching for atom coordinates
def flow_matching_loss(model, x_data, x_noise, t):
    """x_data: protein coordinates, x_noise: random Gaussian"""
    x_t = (1 - t) * x_noise + t * x_data
    target = x_data - x_noise   # vector field
    pred = model(x_t, t)
    # SE(3)-equivariant loss
    return ((pred - target) ** 2).mean()
```

---

## 🔗 응용 영역의 현황 (2024)

### 1. Image Generation

**SOTA**: Stable Diffusion 3, DALL-E 3, Midjourney 6, Imagen 3.

**Trends**:
- Photorealism 거의 saturate
- Controllability 향상 (ControlNet, SDEdit, IP-Adapter)
- Real-time generation (LCM, Turbo)
- Personalization (DreamBooth, LoRA)

### 2. Video Generation

**SOTA**: Sora, Runway Gen-3, Pika 1.5, Kling, Luma Dream Machine.

**Trends**:
- 5-60 sec coherent video
- Text + image conditioning
- Camera motion control
- Style consistency across clips

**Challenges**:
- Long-form (minute+) coherence
- Physics consistency
- Identity preservation
- Compute (1 sec video = 100s of frames)

### 3. 3D Generation

**Methods**:
- **NeRF + SDS**: DreamFusion, Magic3D
- **Gaussian Splatting**: 빠르고 high-quality
- **Direct 3D diffusion**: Shap-E, Point-E
- **Multi-view 2D + reconstruction**: Zero-1-to-3, MVDream

**Use**: gaming, AR/VR, product visualization, architecture.

### 4. Science

**Protein**:
- **AlphaFold 3** (DeepMind 2024): Flow Matching, multi-modal (protein, RNA, ligands)
- **RFdiffusion** (Watson 2023): protein design
- **ESMFold**: language model + folding

**Molecules**:
- **MolDiff, GeoDiff**: molecular generation
- **Property-targeted**: classifier guidance for desired property

**Materials**:
- **MatterGen** (Microsoft 2024): material discovery via diffusion
- **CDVAE**: crystal structure generation

**Drug Design**:
- 단백질 binding 의 ligand 생성
- Combinatorial chemistry 자동화

### 5. Audio / Music

**SOTA**: MusicLM, Suno, Stable Audio.

**Architecture**: AR Transformer on audio tokens (EnCodec) 또는 latent diffusion.

### 6. Embodied AI / Robotics

**World Models**: Dreamer-style, PAI (Predictive World Models).

**Manipulation Diffusion**: action distributions for robot tasks.

---

## ⚖️ Frontier 의 도전과 한계

| 도전 | 현황 |
|------|------|
| **Long-context video** | Sora 가 60 sec, but 시간/비용 prohibitive |
| **Multi-modal coherence** | Cross-modal alignment 가 imperfect |
| **3D consistency** | SDS 의 Janus problem 미해결 |
| **Scientific accuracy** | AlphaFold 가 SOTA 이지만 not perfect |
| **Compute cost** | Production deployment 의 bottleneck |
| **Hallucination** | Generative model 의 fundamental issue |
| **Copyright / Ethics** | Training data, deepfake 등 |

---

## 📌 핵심 정리

| Frontier 영역 | Architecture | Status |
|--------------|-------------|--------|
| **Image (1024×1024)** | Latent Diffusion + Transformer | Mature |
| **Video (60 sec)** | DiT on video latent | Active research |
| **3D (text-to-mesh)** | NeRF + SDS, Gaussian Splatting | Active research |
| **Protein** | Flow Matching + SE(3) equivariance | AlphaFold 3 SOTA |
| **Molecules** | Diffusion + property guidance | Active research |
| **Materials** | Diffusion on crystal structures | MatterGen 등장 |
| **Audio** | AR + latent diffusion | MusicLM, Suno |
| **Embodied** | World models + diffusion policies | Early stage |

| Common Themes |  |
|--------------|---|
| **Diffusion / Flow Matching** | Universal generative framework |
| **Transformer** (DiT) | Scaling-friendly architecture |
| **Multimodal tokenization** | Cross-modal generation |
| **Pretrained large models** | Foundation 의 활용 |
| **Equivariance** | Physical / structural symmetry |

---

## 🤔 생각해볼 문제

**문제 1** (기초): 다음 frontier task 에 가장 적절한 generative model framework 을 선택하고 이유를 제시:
- (a) 1분 길이 영화 생성
- (b) 새로운 약물 분자 설계
- (c) 텍스트로부터 3D 게임 캐릭터 생성
- (d) 음성 → 자연스러운 음악 변환

<details>
<summary>해설</summary>

**(a) 1분 영화**: Diffusion Transformer (DiT) on video latent.
- Sora 의 architectural choice
- Long-context coherence 위해 Transformer 의 attention
- Latent space 로 compute 절약
- 28-50 step diffusion sampling

**(b) 약물 분자 설계**: Flow Matching with SE(3) equivariance.
- 분자의 3D 구조 = atom coordinates → continuous
- SE(3) equivariance for rotation symmetry
- Property targeting via classifier guidance (binding affinity, ADMET)
- Flow matching 의 fast sampling

**(c) 3D 게임 캐릭터**: NeRF + SDS or Gaussian Splatting + diffusion.
- Text → 3D 가 핵심
- DreamFusion-style SDS for high-quality
- Or 2D diffusion + multi-view reconstruction
- Game asset 으로는 mesh 또는 Gaussian Splatting

**(d) 음성 → 음악**: Hybrid AR + latent diffusion.
- AR Transformer (MusicLM, Suno style) for melodic structure
- EnCodec audio tokenization
- Latent diffusion for fine acoustic detail
- Conditioning via cross-attention

**General Pattern**:
- Modality 의 nature (continuous, discrete, sequential, structured)
- Symmetry (translation, rotation, time)
- Compute / latency requirement
- Available pre-trained foundation

</details>

**문제 2** (심화): Sora 의 Diffusion Transformer 와 Stable Diffusion 의 U-Net 의 architectural 차이를 분석하라. Why DiT for video?

<details>
<summary>해설</summary>

**Stable Diffusion U-Net**:
- Convolutional architecture (image-specific inductive bias)
- Hierarchical downsampling/upsampling
- Cross-attention to text tokens
- Designed for fixed-resolution images

**DiT (Diffusion Transformer, Peebles 2023)**:
- Patch-based tokenization (image → patches → tokens)
- Transformer self-attention (no spatial inductive bias)
- AdaLN for conditioning (timestep, class)
- Modality-agnostic (works for any patch type)

**Why DiT for Video (Sora)**:

1. **Scaling**: Transformer 이 잘 scale (parameters + compute). Image generation 의 size limit 을 넘기 위해.

2. **Spacetime Patches**: video 를 3D patches (2D space + 1D time) 로 tokenize. Same Transformer 가 image (2D) 와 video (3D) 를 unified 처리.

3. **Variable Resolution / Aspect Ratio**: Transformer 가 padding/positional encoding 으로 flexible.

4. **Long-Range Dependencies**: video 의 temporal coherence (1초 = 30 frames) 가 attention 으로 자연.

5. **Multimodal**: text tokens + image tokens + ... 모두 unified token sequence.

**Trade-off**:
- DiT: scalable, flexible, but compute heavy
- U-Net: efficient for image, but less scalable

**Empirical**: DiT-XL/2 가 SD 의 U-Net 능가 (FID 2.27 on ImageNet 256×256, vs U-Net 의 3.8). Sora 는 이를 video 로 확장.

**Modern Trend**:
- SD3: MMDiT (Multi-Modal DiT) 채택 — image + text 통합
- Pixart-α: DiT-based, open source
- Sora, Movie Gen: DiT for video

**시사점**:
- "Transformer 가 모든 곳에서 scaling" — 이미지, 텍스트, 오디오, 비디오
- Convolutional inductive bias 의 reduction
- Compute scaling 이 quality 의 main driver
- Universal architecture 의 trend

</details>

**문제 3** (논문 비평): AlphaFold 3 가 Diffusion (Flow Matching) 을 채택한 reasoning 을 분석하라. AlphaFold 2 의 attention-based prediction 과 비교한 advantages 는?

<details>
<summary>해설</summary>

**AlphaFold 2 (2021)**:
- Architecture: Evoformer + structure module
- Output: deterministic 3D structure
- Loss: FAPE + auxiliary
- Single prediction per sequence

**AlphaFold 3 (2024)**:
- Architecture: Pairformer + diffusion module
- Output: stochastic ensemble (multiple plausible structures)
- Loss: Flow Matching on atom positions
- Multi-modal (protein + RNA + DNA + ligands)

**Why Diffusion**:

1. **Stochastic Output / Ensembles**:
   - Real proteins: dynamic, multiple conformations
   - AlphaFold 2: single prediction (averaged)
   - AlphaFold 3: ensemble of plausible structures
   - Flow Matching 이 자연스럽게 distribution 표현

2. **Multi-Modal Unification**:
   - Diffusion 이 atom positions (continuous) 표현 자연
   - Different molecule types (protein, ligand) 이 same coordinate space
   - Single model handles all

3. **Generative Capability**:
   - Beyond prediction: design (target structure 의 sequence 또는 ligand 생성)
   - Property-conditioned (binding affinity, stability)

4. **SE(3) Equivariance**:
   - Diffusion 에 equivariance 통합 가능 (Equiformer 등)
   - AlphaFold 3 의 Pairformer 가 이를 활용

5. **Iterative Refinement**:
   - Diffusion 의 reverse process 가 iterative refinement
   - Initial guess → 점진적 improvement
   - Local minima 회피

**Trade-offs**:

- **Compute**: AlphaFold 3 가 AF2 보다 느림 (50 step diffusion vs single forward)
- **Determinism**: predictions 가 stochastic — reproducibility 약간 손해
- **Training Data**: structural data 가 충분해야

**Empirical Results** (Abramson 2024):
- Protein-protein interfaces: AF2 보다 50% 향상
- Protein-ligand binding: significantly better
- Nucleic acids: previously unsupported, now state-of-the-art

**Broader Impact**:
- Drug design: AlphaFold 3 의 ligand 처리가 pharma 에 직접 영향
- Synthetic biology: 단백질 설계 자동화
- Structural biology: 새 method 의 standard

**시사점**:
- Diffusion 이 분자 modeling 의 standard tool 로 진화
- Generative + predictive 의 fusion (단순 prediction 이 아닌 ensemble)
- Multi-modal generative model 이 different domains 에 적용 가능

**Future**:
- 더 큰 multi-modal model (cell-level simulation?)
- RNA folding, DNA structure 등 nucleic acid 확장
- In silico drug discovery 의 fundamental tool

</details>

---

<div align="center">

[◀ 이전 (03. EBM)](./03-ebm-revival.md) | [📚 README](../README.md) | [🏠 처음으로](../README.md)

**🎉 Generative Model Deep Dive 완주! 🎉**

</div>
