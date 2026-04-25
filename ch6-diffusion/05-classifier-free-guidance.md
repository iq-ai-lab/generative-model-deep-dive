# 05. Classifier-Free Guidance · 현대 응용

## 🎯 핵심 질문

- Classifier guidance (Dhariwal 2021): $\tilde\epsilon = \epsilon_\theta(x_t, t) - \sqrt{1-\bar\alpha_t} \cdot w \cdot \nabla_x \log p_\phi(y|x_t)$ — classifier 의 gradient 를 어떻게 활용하는가?
- Classifier-Free Guidance (Ho & Salimans 2022): $\tilde\epsilon = (1+w) \epsilon_\theta(x_t, y) - w \epsilon_\theta(x_t, \emptyset)$ — 어떻게 implicit classifier 를 만드는가?
- $w$ (guidance scale) 가 quality 와 diversity 를 어떻게 trade-off 하는가? Stable Diffusion 의 $w \approx 7.5$ 의 의미?
- DDIM (Song 2020) 의 non-Markovian sampling 이 어떻게 50-step 으로 sampling 가속하는가?
- Stable Diffusion, DALL-E 3, Imagen, Sora 의 architectural choices?

---

## 🔍 왜 Classifier-Free Guidance 가 결정적인가

Ho & Salimans 2022 "Classifier-Free Diffusion Guidance" — text-to-image diffusion 의 핵심 trick:

1. **No classifier needed** — joint conditional + unconditional model
2. **Quality vs diversity trade-off** — $w$ knob 으로 직접 조절
3. **Stable Diffusion, DALL-E, Imagen 의 표준** — modern text-to-image 의 universal component
4. **Implicit classifier** — Bayes 의 derivative 로 해석

이 단순한 trick (joint training + linear combination) 이 photorealistic text-to-image generation 의 quality 의 핵심. 이 문서에서는 classifier guidance 의 두 form 과 modern application 을 다룹니다.

---

## 📐 수학적 선행 조건

- 이전 문서들: 01-04
- [Probability Theory Deep Dive](https://github.com/iq-ai-lab/probability-theory-deep-dive): Bayes' rule
- Ch6-03: Score, score matching

---

## 📖 직관적 이해

### "조건을 더 강조하기"

Conditional generation: $p(x | y)$ 에서 sampling. 표준 방법: NN 이 $\epsilon_\theta(x, t, y)$ 학습 (text $y$ 를 cross-attention 등으로).

**문제**: 단순 conditional 이 "조건을 약하게 따름" — sample 이 너무 다양하지만 prompt 와 alignment 부족.

**해결**: guidance — sampling 시 condition $y$ 의 영향을 amplify.

### Classifier Guidance

Bayes 의 도구: $p(x | y) \propto p(x) p(y | x)$.

$$\nabla_x \log p(x | y) = \nabla_x \log p(x) + \nabla_x \log p(y | x)$$

Score = unconditional score + classifier gradient.

Sampling 시 classifier $\nabla \log p(y | x)$ 를 "amplify" by $w$:

$$\tilde s = \nabla \log p(x) + w \nabla \log p(y | x)$$

$w > 1$: classifier 더 강하게 — sample 이 $y$ 와 더 align.

**문제**: separate classifier $p_\phi(y | x)$ 학습 필요. Noisy $x_t$ 에 대해 robust 하게 분류해야.

### Classifier-Free Guidance

**Idea**: 같은 diffusion 모델이 conditional 과 unconditional 둘 다 학습.

Training:
- $p$-fraction (e.g., 10%) 의 batch 에서 condition 을 dropout ($y \to \emptyset$ token)
- 나머지: conditional training

이로 single NN $\epsilon_\theta(x, t, y)$ 가 conditional ($y$ given) 과 unconditional ($y = \emptyset$) 둘 다 학습.

**Sampling**:

$$\tilde\epsilon = (1+w) \epsilon_\theta(x_t, y) - w \epsilon_\theta(x_t, \emptyset)$$

= linear combination, no separate classifier.

**Implicit Classifier**:

$\epsilon_\theta(x_t, y) - \epsilon_\theta(x_t, \emptyset) \propto \nabla_x [\log p(y | x)]$ — implicit classifier gradient.

### $w$ Scale 의 효과

- $w = 0$: pure conditional ($\tilde\epsilon = \epsilon(y)$) — 무 guidance
- $w = 1$: 약간 amplification
- $w = 5-10$ (typical): strong guidance — sharp, prompt-following
- $w = 20+$: too much — saturated, lacks diversity

Stable Diffusion default: $w = 7.5$.

---

## ✏️ 엄밀한 정의·정리

### 정의 5.1 — Classifier Guidance (Dhariwal 2021)

별도 classifier $p_\phi(y | x_t)$ 학습 (noisy input 에서). Sampling:

$$\tilde\epsilon(x_t, t, y) = \epsilon_\theta(x_t, t) - \sqrt{1-\bar\alpha_t} \cdot w \cdot \nabla_{x_t} \log p_\phi(y | x_t)$$

(score-form: $\tilde s = s_\theta + w \nabla \log p_\phi(y | x)$, 둘 다 식 동등 with $\epsilon = -\sigma s$)

**$w$ scale**: classifier gradient amplification.

### 정의 5.2 — Classifier-Free Guidance (Ho & Salimans 2022)

Joint training: single $\epsilon_\theta(x, t, y)$ NN, $y$ 가 random 하게 $\emptyset$ (null condition) 으로 dropout.

**Training loss**:

$$L = \mathbb{E}_{x_0, \epsilon, t, c}[\|\epsilon - \epsilon_\theta(x_t, t, c)\|^2]$$

여기서 $c = y$ with prob $1 - p_\text{drop}$, else $c = \emptyset$. $p_\text{drop} = 0.1-0.2$ typical.

**Sampling**:

$$\tilde\epsilon = (1 + w) \epsilon_\theta(x_t, t, y) - w \epsilon_\theta(x_t, t, \emptyset)$$

equivalent to score:

$$\tilde s = (1 + w) s_\theta(x_t, t, y) - w s_\theta(x_t, t, \emptyset)$$

= $s_\theta(x_t, t, \emptyset) + (1 + w)(s_\theta(y) - s_\theta(\emptyset))$.

### 정리 5.3 — Implicit Classifier 의 Bayes 해석

Bayes:

$$p(x | y) = p(x) p(y | x) / p(y)$$

$$\nabla_x \log p(x | y) = \nabla_x \log p(x) + \nabla_x \log p(y | x)$$

따라서:

$$\nabla_x \log p(y | x) = \nabla_x \log p(x | y) - \nabla_x \log p(x) = s_\theta(x, y) - s_\theta(x, \emptyset)$$

**Conditional - unconditional = classifier gradient** (in score form).

CFG sampling:

$$\tilde s = s_\theta(x, \emptyset) + (1 + w)(s_\theta(x, y) - s_\theta(x, \emptyset))$$

$$= s_\theta(x, y) + w \cdot (s_\theta(x, y) - s_\theta(x, \emptyset))$$

$$= s_\theta(x, y) + w \cdot \nabla \log p(y | x)$$

= classifier guidance with **implicit classifier**.

### 정리 5.4 — Quality-Diversity Trade-off

Diffusion 의 distribution under guided sampling:

$$\tilde p(x | y) \propto p(x | y) \cdot p(y | x)^w = p(x) p(y | x)^{1+w} / p(y)$$

**$w = 0$**: $\tilde p = p(x | y)$ — original conditional.

**$w > 0$**: $p(y | x)^{1+w}$ — classifier 의 confident region 에 집중. Sharper, less diverse.

**$w \to \infty$**: pure mode-seeking on classifier — 모든 sample 이 most-confident class.

**Trade-off**:
- Precision (quality): $w$ 와 함께 증가
- Recall (diversity): $w$ 와 함께 감소
- 일반적 sweet spot: $w = 5-10$

### 정의 5.5 — DDIM (Song 2020)

Non-Markovian deterministic sampling. Reverse process:

$$x_{t-1} = \sqrt{\bar\alpha_{t-1}} \hat x_0 + \sqrt{1 - \bar\alpha_{t-1} - \sigma_t^2} \epsilon_\theta(x_t, t) + \sigma_t \epsilon$$

여기서:
- $\hat x_0 = (x_t - \sqrt{1-\bar\alpha_t} \epsilon_\theta) / \sqrt{\bar\alpha_t}$ — predicted $x_0$
- $\sigma_t \in [0, \tilde\beta_t]$ — interpolation parameter

**$\sigma_t = 0$**: deterministic ODE-style sampling. **50 steps 가능** (vs DDPM 1000).

**$\sigma_t = \tilde\beta_t$**: standard DDPM (stochastic).

### 정리 5.6 — DDIM 이 PF-ODE 의 Discretization

DDIM with $\sigma_t = 0$ 이 PF-ODE (Ch6-04) 의 specific discretization. 따라서 same exact likelihood, ODE solver 의 advantages.

**Modern usage**: DDIM 이 default sampling for most diffusion models. DPM-Solver (Lu 2022) 가 더 정확한 ODE solver.

### 정의 5.7 — Modern Text-to-Image Architectures

**Stable Diffusion** (Rombach 2022):
- Latent Diffusion: VAE encoder/decoder + diffusion in latent (4×64×64 for 512×512)
- U-Net with cross-attention to text (CLIP text encoder)
- CFG with $w = 7.5$

**DALL-E 3** (OpenAI 2023):
- Multi-stage: prompt rewriter (LLM) + diffusion
- Improved text-image alignment

**Imagen** (Saharia 2022):
- Cascaded diffusion: 64×64 → 256×256 → 1024×1024
- T5-XXL text encoder (강한 language understanding)

**Sora** (OpenAI 2024):
- Diffusion Transformer (DiT) on video latent
- Scaling to large compute

---

## 🔬 증명 및 수학적 유도

### 유도 1 — CFG 의 Score Form

CFG sampling formula:

$$\tilde\epsilon = (1+w)\epsilon_\theta(x_t, y) - w \epsilon_\theta(x_t, \emptyset)$$

Score equivalence ($s = -\epsilon / \sigma_t$):

$$\tilde s = -\tilde\epsilon / \sigma_t = (1+w) s_\theta(y) - w s_\theta(\emptyset)$$

$$= s_\theta(\emptyset) + (1+w)(s_\theta(y) - s_\theta(\emptyset))$$

$$= s_\theta(\emptyset) + (1+w) \nabla \log p(y|x)$$

(using Bayes: $s(y) - s(\emptyset) = \nabla \log p(y|x)$).

이는 unconditional score + amplified classifier gradient — **classifier guidance with implicit classifier**.

### 유도 2 — $w$ 의 Distribution Effect

Sampling 의 stationary distribution $\tilde p$ 가 score $\tilde s$ 에 대응. 즉 $\nabla \log \tilde p = \tilde s$.

$\tilde s = \nabla \log p(x) + (1+w) \nabla \log p(y|x) = \nabla [\log p(x) + (1+w) \log p(y|x)]$

따라서 $\log \tilde p \propto \log p(x) + (1+w) \log p(y|x)$:

$$\tilde p(x|y) \propto p(x) p(y|x)^{1+w}$$

**$w = 0$**: $\tilde p = p(x) p(y|x) = p(x, y) \propto p(x|y)$ ✓.

**$w > 0$**: sharper — confident classifier region 에 mass concentrated.

### 유도 3 — DDIM 의 Non-Markovian Form

DDIM 의 forward marginal 은 DDPM 과 동일 ($q(x_t | x_0) = \mathcal{N}(\sqrt{\bar\alpha_t} x_0, (1-\bar\alpha_t) I)$).

다만 reverse process 가 **non-Markovian**: $q(x_{t-1} | x_t, x_0)$ 의 family 정의:

$$q_\sigma(x_{t-1} | x_t, x_0) = \mathcal{N}\left(\sqrt{\bar\alpha_{t-1}} x_0 + \sqrt{1 - \bar\alpha_{t-1} - \sigma_t^2} \cdot \frac{x_t - \sqrt{\bar\alpha_t} x_0}{\sqrt{1-\bar\alpha_t}}, \sigma_t^2 I\right)$$

$\sigma_t^2$ 가 hyperparameter:
- $\sigma_t = \tilde\beta_t$: standard DDPM
- $\sigma_t = 0$: deterministic, ODE-style

**$\sigma_t = 0$ 의 의미**: $x_{t-1} = $ deterministic function of $(x_t, \hat x_0)$ — same noise → same trajectory.

**Sampling speed**: deterministic 이 적은 step 으로 충분. 50-step DDIM ≈ 1000-step DDPM (FID 비슷).

### 유도 4 — Latent Diffusion 의 Architectural Insight

**Stable Diffusion 의 핵심**: pixel space ($512 \times 512 \times 3 = 786K$) 가 너무 큰 — diffusion compute 매우 비쌈.

**Solution**: pretrained VAE 로 latent space ($4 \times 64 \times 64 = 16K$) 으로 압축. Diffusion 을 latent 에서 수행.

**Pipeline**:
1. Train VAE: $x \to z \to \hat x$ (perceptual + adversarial loss)
2. Train diffusion in $z$-space: $z_T \to z_0$ via U-Net + cross-attention
3. Decode: $\hat x = \text{Decoder}(z_0)$

**Compute**: $48 \times$ 적은 dim → 50× faster 학습/sampling.

**Sample quality**: VAE 의 perceptual reconstruction 이 high enough → little quality loss.

---

## 💻 실험 재현 / PyTorch 구현

### 실험 1 — Conditional Diffusion + CFG Training

```python
import torch
import torch.nn as nn

class ConditionalEpsNet(nn.Module):
    """U-Net with conditioning"""
    def __init__(self, n_classes=10, hidden=128):
        super().__init__()
        self.t_emb = nn.Embedding(1000, hidden)
        # Class embedding (with null token)
        self.class_emb = nn.Embedding(n_classes + 1, hidden)
        # null class index = n_classes
        # ... U-Net layers ...

    def forward(self, x, t, c):
        # c: class label or n_classes (null)
        t_e = self.t_emb(t)
        c_e = self.class_emb(c)
        cond = (t_e + c_e)[:, :, None, None]
        # forward through U-Net with cond
        return ...

# Training with classifier-free guidance
def cfg_loss(model, x_0, c, p_drop=0.1):
    bsz = x_0.size(0)
    # Random dropout of conditioning
    drop_mask = torch.rand(bsz, device=x_0.device) < p_drop
    c_input = torch.where(drop_mask, torch.tensor(model.n_classes, device=x_0.device), c)

    t = torch.randint(0, T, (bsz,), device=x_0.device)
    eps = torch.randn_like(x_0)
    x_t = q_sample(x_0, t, eps)
    eps_pred = model(x_t, t, c_input)
    return F.mse_loss(eps_pred, eps)
```

### 실험 2 — CFG Sampling

```python
@torch.no_grad()
def cfg_sample(model, shape, c, w=7.5):
    x = torch.randn(shape).cuda()
    null_c = torch.full_like(c, model.n_classes)
    for t in reversed(range(T)):
        t_tensor = torch.tensor([t]).expand(x.size(0)).cuda()
        # Conditional and unconditional predictions
        eps_cond = model(x, t_tensor, c)
        eps_uncond = model(x, t_tensor, null_c)
        # CFG combination
        eps_pred = (1 + w) * eps_cond - w * eps_uncond
        # Reverse step (DDPM)
        coef1 = 1 / alphas[t].sqrt()
        coef2 = betas[t] / (1 - alpha_bars[t]).sqrt()
        mu = coef1 * (x - coef2 * eps_pred)
        if t > 0:
            sigma = betas[t].sqrt()
            x = mu + sigma * torch.randn_like(x)
        else:
            x = mu
    return x

# Sampling with different w
samples_w0 = cfg_sample(model, shape, c, w=0)      # No guidance
samples_w7 = cfg_sample(model, shape, c, w=7.5)    # Default
samples_w20 = cfg_sample(model, shape, c, w=20)    # Strong
# w 클수록 sharp, less diverse
```

### 실험 3 — DDIM Sampling

```python
@torch.no_grad()
def ddim_sample(model, shape, c, n_steps=50, eta=0.0):
    """eta=0: deterministic, eta=1: DDPM"""
    x = torch.randn(shape).cuda()
    # Subsample timesteps (50 from 1000)
    timesteps = torch.linspace(T - 1, 0, n_steps).long()
    for i, t in enumerate(timesteps):
        t_prev = timesteps[i + 1] if i < n_steps - 1 else 0
        t_tensor = torch.tensor([t]).expand(x.size(0)).cuda()
        eps_pred = model(x, t_tensor, c)
        # DDIM formula
        ab_t = alpha_bars[t]
        ab_prev = alpha_bars[t_prev] if t_prev > 0 else torch.tensor(1.0)
        sigma_t = eta * ((1 - ab_prev) / (1 - ab_t)).sqrt() * (1 - ab_t / ab_prev).sqrt()
        # Predict x_0
        x_0_pred = (x - (1 - ab_t).sqrt() * eps_pred) / ab_t.sqrt()
        # Step
        x_prev = ab_prev.sqrt() * x_0_pred \
               + (1 - ab_prev - sigma_t**2).sqrt() * eps_pred
        if eta > 0:
            x_prev = x_prev + sigma_t * torch.randn_like(x)
        x = x_prev
    return x

# 50 step DDIM ≈ 1000 step DDPM in quality
```

### 실험 4 — $w$ Sweep 으로 Quality-Diversity 측정

```python
ws = [0, 1, 3, 5, 7.5, 10, 20]
for w in ws:
    samples = cfg_sample(model, (1000, 1, 28, 28), c, w=w)
    # Measure FID and diversity (e.g., LPIPS variance)
    fid = compute_fid(samples)
    diversity = compute_diversity(samples)
    print(f"w={w}: FID={fid:.2f}, Diversity={diversity:.3f}")

# 일반적 pattern:
# w=0: FID 높음, diversity 높음
# w=7.5: FID 낮음 (best), diversity 적당
# w=20: FID 낮지만 saturated, diversity 매우 낮음
```

### 실험 5 — Latent Diffusion Toy

```python
# Pretrained VAE for compression
vae = ...   # encoder/decoder

# Train diffusion in latent space
def latent_diffusion_step(vae, model, x_0):
    with torch.no_grad():
        z_0 = vae.encode(x_0)   # [B, 4, H/8, W/8]
    eps = torch.randn_like(z_0)
    t = torch.randint(0, T, (x_0.size(0),))
    z_t = q_sample(z_0, t, eps)
    eps_pred = model(z_t, t, ...)
    return F.mse_loss(eps_pred, eps)

@torch.no_grad()
def latent_diffusion_sample(vae, model, shape, c):
    z = ddim_sample(model, shape, c)
    x = vae.decode(z)
    return x
```

---

## 🔗 이론과 실전의 간극

### 1. Stable Diffusion 의 Pipeline

Stable Diffusion (latent diffusion + CFG + DDIM):
1. CLIP text encoder: text → embedding
2. U-Net (in latent space): conditioning via cross-attention
3. CFG with $w = 7.5$
4. DDIM sampling 50 steps
5. VAE decoder: latent → 512×512 image

**Total time**: ~5 sec on RTX 3090 — practical for production.

### 2. Multi-Stage Diffusion

Imagen, DALL-E 2: cascaded diffusion
- Stage 1: 64×64 conditioned on text
- Stage 2: 256×256 super-resolution diffusion
- Stage 3: 1024×1024 (optional)

각 stage 가 conditional diffusion. Cumulative quality.

### 3. Modern Variants

- **Stable Diffusion 3**: Rectified Flow (Ch7-02) + MMDiT + CFG
- **DALL-E 3**: GPT-based prompt enhancement + diffusion
- **Sora**: DiT (Diffusion Transformer) on video latent
- **Imagen 2**: T5 encoder + cascaded
- **Midjourney**: closed source, 자체 architecture

모두 CFG 와 DDIM-style sampling 을 핵심으로 사용.

---

## ⚖️ 가정과 한계

| 가정 | 한계 |
|------|------|
| CFG 가 항상 quality 향상 | $w$ 너무 크면 saturation, artifact |
| Single conditional + null model | Multi-modal conditioning 어려움 |
| DDIM 이 PF-ODE approximation | Higher-order solver (DPM-Solver) 가 더 정확 |
| Implicit classifier 가 well-defined | OOD prompt 에서 부정확 |
| Latent diffusion 이 quality 보존 | VAE bottleneck 이 fine detail 손실 |

---

## 📌 핵심 정리

$$\boxed{\tilde\epsilon = (1+w) \epsilon_\theta(x_t, y) - w \epsilon_\theta(x_t, \emptyset) \quad \text{(CFG)}}$$

$$\boxed{\tilde p(x | y) \propto p(x) \cdot p(y | x)^{1+w} \quad - \text{ guidance distribution}}$$

| Method | Form | Pros |
|--------|------|------|
| **Classifier Guidance** (Dhariwal 2021) | $+w \nabla \log p_\phi(y\|x)$ | Direct |
| **Classifier-Free** (Ho & Salimans 2022) | $(1+w) \epsilon(y) - w \epsilon(\emptyset)$ | No separate classifier |

| Modern Application | Architecture |
|-------------------|-------------|
| **Stable Diffusion** | Latent diffusion + CFG + DDIM |
| **DALL-E 3** | Multi-stage + GPT prompt rewriting |
| **Imagen** | Cascaded diffusion + T5 encoder |
| **Sora** | Diffusion Transformer (DiT) on video latent |

| Sampling | Steps | Speed |
|----------|-------|-------|
| **DDPM** | 1000 | Slow |
| **DDIM** | 50 | 20× fast |
| **DPM-Solver++** | 10-20 | 50-100× fast |
| **Heun (EDM)** | ~30 | Balanced |

---

## 🤔 생각해볼 문제

**문제 1** (기초): CFG sampling 에서 $w = 0$ 와 $w = 1$ 의 difference 를 explicit 으로 보여라. 어느 것이 더 quality 좋은가?

<details>
<summary>해설</summary>

**$w = 0$**: $\tilde\epsilon = (1) \epsilon(y) - 0 \cdot \epsilon(\emptyset) = \epsilon(y)$ — pure conditional.

**$w = 1$**: $\tilde\epsilon = 2 \epsilon(y) - \epsilon(\emptyset)$ — 약간 amplification.

**Score 의미**:
- $w = 0$: $\tilde s = s(y) = \nabla \log p(x|y)$ — original conditional score
- $w = 1$: $\tilde s = 2 s(y) - s(\emptyset) = s(y) + (s(y) - s(\emptyset)) = s(y) + \nabla \log p(y|x)$ — conditional + classifier gradient

**Distribution**:
- $w = 0$: $\tilde p = p(x|y)$
- $w = 1$: $\tilde p \propto p(x) p(y|x)^2 = p(x|y) p(y|x)$ — classifier 의 영향 한 번 더

**Empirical**:
- $w = 0$: 다양하지만 prompt 와 약하게 align
- $w = 1$: 약간 sharper, prompt-following 약간 강함
- $w = 7.5$ (typical): much sharper, strong alignment

**Quality**:
- 단순 sample sharpness: $w$ 크면 좋음 (one perspective)
- Diversity: $w$ 작으면 좋음
- Trade-off — application dependent

**실전**: Stable Diffusion 의 default $w = 7.5$ 가 quality-diversity sweet spot. 매우 specific prompt 면 더 큰 $w$ (15-20) 로.

</details>

**문제 2** (심화): CFG 의 implicit classifier $\nabla \log p(y|x) = s(x, y) - s(x, \emptyset)$ 의 정당성을 Bayes 의 derivative 로 보여라.

<details>
<summary>해설</summary>

**Bayes**:

$$p(x | y) = \frac{p(x) p(y | x)}{p(y)}$$

Take log:

$$\log p(x | y) = \log p(x) + \log p(y | x) - \log p(y)$$

$\log p(y)$ 는 $x$ 와 무관 (constant in $x$). Take $\nabla_x$:

$$\nabla_x \log p(x | y) = \nabla_x \log p(x) + \nabla_x \log p(y | x)$$

Rearranging:

$$\nabla_x \log p(y | x) = \nabla_x \log p(x | y) - \nabla_x \log p(x)$$

**Score notation**:
- $s(x, y) = \nabla_x \log p(x | y)$ — conditional score (NN learns)
- $s(x, \emptyset) = \nabla_x \log p(x)$ — unconditional score (NN learns when $y$ dropped)

따라서:

$$\nabla_x \log p(y | x) = s(x, y) - s(x, \emptyset)$$

= **implicit classifier gradient** computed from conditional and unconditional scores.

**시사점**:
- Classifier $\nabla \log p(y|x)$ 가 trained directly without separate classifier
- 동일 NN 의 두 forward pass (with $y$, with $\emptyset$) 의 difference
- "Classifier-free" — additional NN 없음

**왜 효과적**:
- Diffusion 모델이 noisy $x_t$ 에 robust 한 score 학습 → noisy classifier gradient 도 robust
- Single NN 이 둘 다 학습 → consistency
- Joint training 의 implicit regularization

이것이 CFG 의 elegance — Bayes 의 단순 derivative 가 modern text-to-image 의 핵심.

</details>

**문제 3** (논문 비평): Stable Diffusion 의 latent diffusion vs DALL-E 의 pixel diffusion (DALL-E 2 의 unCLIP) — 두 architectural choice 의 trade-off 를 분석하라.

<details>
<summary>해설</summary>

**Stable Diffusion (Latent Diffusion, Rombach 2022)**:
- Architecture: VAE (KL-regularized) + diffusion in latent + decoder
- Latent: 4×64×64 for 512×512 image (48× compression)
- Diffusion U-Net cost: small (latent space)
- VAE: pretrained, fixed during diffusion training

**Advantages**:
- ✅ Compute efficient: 48× less FLOPS in diffusion
- ✅ Faster training/sampling
- ✅ Open source feasible (smaller compute)
- ❌ VAE bottleneck — fine details lossy
- ❌ Two-stage pipeline complexity

**DALL-E 2 (unCLIP, Ramesh 2022)**:
- Architecture: CLIP text/image encoders + Diffusion prior + Diffusion decoder
- Pixel space (or low compression)
- Multiple diffusion stages

**Advantages**:
- ✅ Higher fidelity (pixel-accurate)
- ✅ CLIP semantic structure leveraged
- ✅ Image variations via CLIP embedding manipulation
- ❌ Compute heavy
- ❌ Complex multi-stage pipeline

**Imagen (Saharia 2022)**: similar to DALL-E 2 but cascaded.

**Quality Comparison** (subjective, 2022-2023 era):
- Stable Diffusion: sharper details up to 512×512, photorealistic
- DALL-E 2: better text-image alignment, more semantic
- Imagen: best text understanding (T5-XXL encoder)

**Modern Trend (2024)**:
- **Latent diffusion 이 dominant**: SD3, Pixart-alpha, Latent Consistency Model
- **Reasons**: compute efficiency 의 결정적 advantage
- VAE 의 quality 가 충분 (Latent Adversarial Diffusion 등의 개선)

**Pixel Diffusion 의 특수 위치**:
- Imagen: Google, large compute available
- Cascaded: low-res (pixel) → super-res (pixel) hybrid
- Research: pixel-level 의 fundamental quality limit

**General Principle**:
- Compression (VAE) 가 "lossy but cheap"
- Pixel 이 "lossless but expensive"
- Sweet spot 이 dataset/quality requirement 에 따라

**Stable Diffusion 3 의 진화**: Rectified Flow + MMDiT — latent diffusion 의 next generation. Open source community 가 latent diffusion 의 mainstream 으로 driven.

**시사점**: architectural choice 가 단순 "best practice" 아님 — compute, target quality, deployment 의 trade-off. Latent diffusion 이 democratized text-to-image (Stability AI 의 open source release) 의 enabler.

</details>

---

<div align="center">

[◀ 이전 (04. Score-SDE)](./04-score-sde.md) | [📚 README](../README.md) | [다음 ▶ (Ch7-01. 5대 계보 비교)](../ch7-unification/01-five-families-comparison.md)

</div>
