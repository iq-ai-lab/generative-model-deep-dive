# 04. Posterior Collapse — 원인과 해결

## 🎯 핵심 질문

- Posterior collapse $q_\phi(z|x) \to p(z)$ 가 정확히 어떤 현상인가? 왜 latent 가 무력화되는가?
- 강력한 decoder (autoregressive PixelCNN, attention) 가 왜 collapse 를 유발하는가? 약한 decoder (Gaussian) 와의 차이는?
- KL annealing, Free Bits, δ-VAE 같은 해결책의 정확한 메커니즘은?
- Language VAE 에서 collapse 가 더 심각한 이유는? Image VAE 와 다른 점은?
- Diffusion 모델은 왜 posterior collapse 가 없는가? Chain 분해의 효과는?

---

## 🔍 왜 Posterior Collapse 가 결정적 문제인가

VAE 의 구조 — encoder + latent + decoder — 의 의도는 latent $z$ 가 데이터 $x$ 의 의미 있는 representation 을 capture 하는 것. **Posterior collapse** 는 이 구조의 fundamental failure mode:

$$q_\phi(z|x) \approx p(z) \quad \text{for all } x$$

즉 encoder 가 $x$ 를 봐도 항상 같은 분포 (prior) 를 출력 → $z$ 가 $x$ 정보 없음 → decoder 가 **$x$ 와 무관한** $z$ 로부터 생성.

이 현상의 영향:
1. **Generation 불능**: latent 가 의미 없으므로 sampling 이 random
2. **Representation 무력화**: representation learning 의 핵심 목적 상실
3. **ELBO 의 misleading**: KL = 0 이지만 reconstruction loss 가 낮아 보일 수 있음

원인은 architectural + optimization 의 복합 — 강력한 decoder 가 latent 없이 reconstruction 가능하면, KL 페널티 회피 위해 encoder 가 latent 사용 포기. 이 문서에서는 collapse 의 수학적 분석과 다양한 해결책을 다룹니다.

---

## 📐 수학적 선행 조건

- 이전 문서들: 01-elbo, 02-reparameterization, 03-beta-vae
- [Information Theory Deep Dive](https://github.com/iq-ai-lab/information-theory-deep-dive): Mutual information

---

## 📖 직관적 이해

### "Encoder 가 일을 안 한다"

VAE 훈련 후 다음 현상 관찰:

1. KL ≈ 0 (모든 $x$ 에 대해 $\mu_\phi(x) \approx 0, \sigma_\phi(x) \approx 1$)
2. Reconstruction 은 얼마나 좋은가?
   - **약한 decoder**: bad — latent 없이는 $x$ 못 만듬
   - **강한 decoder** (autoregressive): OK — $x$ 자체의 internal pattern 으로 reconstruct
3. Sampling: random — latent 가 의미 없으므로

이때 모델은 **"VAE 처럼 보이지만 사실 그냥 unconditional generative model"**.

### "강한 Decoder 의 함정"

Decoder 가 충분히 강력하면 (예: PixelCNN, autoregressive) $z$ 가 informative 한 것보다 **$z$ 무시하고 데이터 자체에서 학습** 이 KL 페널티 피하기에 유리. ELBO optimization 이 이 "lazy" mode 로 수렴.

수학적으로:
- KL = 0: penalty 없음, optimization 이 선호
- Decoder 강력: 최대 reconstruction 어떤 $z$ 라도 (심지어 random) 가능
- 따라서 encoder 학습 안 함

### "Language VAE 의 특히 심각함"

자연어는 강한 sequential structure (다음 단어가 이전 단어로 거의 결정). Decoder = LSTM/Transformer language model 은 매우 강력. 따라서 language VAE 는 **거의 항상 collapse**.

해결: KL annealing 필수, free bits 일반적, 그래도 collapse 여전.

---

## ✏️ 엄밀한 정의·정리

### 정의 4.1 — Posterior Collapse

데이터 $x$ 에 대해:

$$\text{KL}(q_\phi(z|x) \| p(z)) \to 0 \quad \text{for almost all } x$$

또는 동등하게:

$$q_\phi(z|x) \approx p(z) \quad \text{(in total variation)}$$

**수치적 진단**: ELBO 에서 KL term 의 평균이 $< \epsilon$ (e.g., $\epsilon = 0.5$ nats), 또는 **active dimensions** (KL > threshold 인 dim 수) 가 0 또는 매우 적음.

### 정리 4.2 — Posterior Collapse 의 "Trivial" Mode

Decoder $p_\theta(x|z)$ 가 $z$-independent 인 경우 ($p_\theta(x|z) = p_\theta(x)$):

- $\mathbb{E}_q[\log p_\theta(x|z)] = \log p_\theta(x)$ — $z$ 와 무관
- $\text{KL}(q \| p(z))$ 는 $q$ 의 함수
- ELBO 최대화 → $q = p(z)$ for all $x$ → KL = 0

이 상태에서 ELBO = $\log p_\theta(x)$ — 모델이 그냥 unconditional density. 의미 있는 latent representation 학습 실패.

### 정리 4.3 — Strong Decoder + Limited Capacity Encoder = Collapse

가정:
- Decoder $p_\theta(x|z)$ 가 충분히 강력 (e.g., PixelCNN) — 즉 어떤 $z$ 에 대해서도 $\mathbb{E}[\log p_\theta(x|z)] \approx \log p_\theta(x)$ 가능
- Encoder $q_\phi$ family 가 제한 (e.g., factored Gaussian)

그러면 ELBO = $\mathbb{E}_q[\log p_\theta(x|z)] - \text{KL}$ 의 최적해가 KL = 0 (collapse) 로 수렴 — informative encoder 추가 학습이 reconstruction 이득 < KL 페널티.

### 정리 4.4 — KL Vanishing Schedule (Bowman 2016)

훈련 초기 KL 항을 0 으로 시작, 점진적으로 1 까지 증가:

$$\mathcal{L}_t = \mathbb{E}_q[\log p_\theta(x|z)] - w(t) \cdot \text{KL}(q \| p(z))$$

$w(t) = \min(1, t / T)$, $T$ 는 warmup.

**효과**: 초기 단계 (KL 무시) → encoder 가 informative latent 학습. 점진적 KL 추가 → 학습된 latent 위에 regularization. Collapse 방지.

### 정리 4.5 — Free Bits (Kingma 2016)

KL 항이 dimension-wise threshold $\lambda$ 이하면 페널티 없음:

$$\text{KL}_\text{free} = \sum_i \max(\lambda, \text{KL}_i(q_\phi(z_i|x) \| p(z_i)))$$

**메커니즘**: 각 latent dim 이 적어도 $\lambda$ nats 의 정보 사용 — $\lambda$ 이하는 "free" (페널티 없음). KL = 0 mode 가 더 이상 optimal 아님.

**효과**: collapse 방지, active dimensions 보장. 일반적 $\lambda \in [0.5, 2]$ nats.

### 정의 4.6 — δ-VAE (Razavi 2019)

Posterior 와 prior 의 minimum KL distance 보장:

$$q_\phi(z|x) = \mathcal{N}(\mu_\phi(x), \sigma_\phi^2(x) I), \quad \sigma_\phi^2(x) \leq 1 - \delta$$

(constraint on encoder output variance)

이로 KL 의 lower bound:

$$\text{KL}(q \| p) \geq -\frac{1}{2} \log(1 - \delta) > 0$$

**효과**: KL 이 0 으로 수렴 불가능 — collapse 막음.

---

## 🔬 증명 및 수학적 유도

### 유도 1 — Collapse 의 Optimization Landscape Argument

ELBO loss $L = -\text{recon} + \text{KL}$ 의 두 mode:

**Mode A (Informative)**: $\mu_\phi(x), \sigma_\phi(x)$ depend on $x$. KL > 0 (e.g., 5 nats). Recon: -recon = -100 (good). Total: -95.

**Mode B (Collapsed)**: $\mu_\phi(x) = 0, \sigma_\phi(x) = 1$. KL = 0. Recon: -recon = -110 (worse). Total: -110.

만약 decoder 약하면 Mode A 가 우월 → 학습. 강하면 두 mode 차이 적거나 Mode B 우월.

**왜 Mode B 가 attractor 인가**: gradient descent 의 초기 점부터 KL 항이 $\mu_\phi \to 0$ pressure. Reconstruction 이 $\mu$ 작아질 때 손실 적으면, $\mu \to 0$ 가 빠르게 수렴.

### 유도 2 — Strong Decoder 의 정확한 정의

Decoder $p_\theta(x|z)$ 가 **strong** 이라 함: $\theta$ 의 capacity 가 충분하여 $\theta^*$ 가 존재해

$$\log p_{\theta^*}(x|z) \approx \log p(x) \quad \forall z, x$$

즉 $z$ 의 정보 없이 $x$ 의 marginal 을 학습 가능. Examples:
- AR decoder (PixelCNN, RNN, Transformer): $p(x_i | x_{<i})$ 가 자체적으로 강력
- Deep CNN with skip connections
- Noise injection in early layer

**약한 decoder**:
- Shallow Gaussian decoder
- Mean-pooling architectures
- Gaussian without spatial structure

### 유도 3 — KL Annealing 의 정당성

Annealing schedule $w(t) = \min(1, t/T)$.

**초기 ($w = 0$)**: $L = -\text{recon}$. AE objective. Encoder 가 informative latent 학습.

**중간 ($0 < w < 1$)**: $L = -\text{recon} + w \cdot \text{KL}$. KL 페널티 점차 추가. 이미 학습된 informative encoder 가 prior 와 멀어진 것을 점차 normalize.

**최종 ($w = 1$)**: Full ELBO. 학습된 latent 의 regularization 완료.

**왜 작동**: collapse 의 attractor 가 초기에 weak. 모델이 informative mode 에 정착한 후, KL 추가가 mode shift 어려움 (saddle point 경로).

**한계**: 이미 strong decoder 라면 annealing 후에도 collapse 로 회귀 가능. Free bits 나 architecture 변경 필요.

### 유도 4 — Free Bits 의 Loss Modification

Standard ELBO loss:

$$L = -\text{recon} + \sum_i \text{KL}_i$$

Free bits ($\lambda$):

$$L_\text{FB} = -\text{recon} + \sum_i \max(\lambda, \text{KL}_i)$$

**Gradient**: $\text{KL}_i < \lambda$ 인 dim 에서 $\partial L / \partial \text{KL}_i = 0$ — KL 줄이는 incentive 없음. 따라서 encoder 가 그 dim 사용 free.

**Equivalent formulation**:
$$L_\text{FB} = -\text{recon} + \sum_i \text{KL}_i + \sum_i \max(0, \lambda - \text{KL}_i)$$

= ELBO + bonus for using dim. Forces dim usage above $\lambda$.

### 유도 5 — Diffusion 이 Collapse 가 없는 이유

Diffusion 의 ELBO 는 chain 분해:

$$\text{ELBO} = \sum_t \mathbb{E}_q[\text{KL}(q(x_{t-1} | x_t, x_0) \| p_\theta(x_{t-1} | x_t))]$$

각 step $t$ 에서 $x_{t-1}$ 이 $x_t$ 와 latent 처럼 동작, $x_0$ (clean) 으로 supervision.

**왜 collapse 없는가**:
1. **No latent variable to ignore**: $x_t$ 가 원래 데이터의 noised version, 무시 불가능
2. **Step-wise 작업**: 각 step 의 KL 이 정의되어 있음
3. **Forward process 가 fixed**: encoder 가 학습 가능한 게 아니라 noise schedule $q$

이것이 diffusion 이 VAE 의 한계를 극복한 architectural 이유 중 하나.

---

## 💻 실험 재현 / PyTorch 구현

### 실험 1 — Posterior Collapse 진단

```python
import torch
import torch.nn as nn

@torch.no_grad()
def diagnose_collapse(model, dataloader, threshold=0.1):
    """Active dimensions 와 평균 KL 측정"""
    z_dim = model.z_dim
    kl_per_dim = torch.zeros(z_dim)
    n = 0
    for x, _ in dataloader:
        x = x.cuda().view(-1, 784)
        mu, logvar = model.encode(x)
        # KL per dim per sample
        kl_dim = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp())
        kl_per_dim += kl_dim.mean(0).cpu()
        n += 1
    kl_per_dim /= n
    active = (kl_per_dim > threshold).sum().item()
    print(f"Active dims (KL > {threshold}): {active}/{z_dim}")
    print(f"Mean KL per dim: {kl_per_dim}")
    print(f"Total KL: {kl_per_dim.sum().item():.3f}")
    return active

# 표준 VAE: 일반적으로 5-15 active dims (z_dim=32)
# Strong decoder VAE: 종종 0-2 active dims (collapse)
```

### 실험 2 — KL Annealing 구현

```python
def train_with_annealing(model, opt, loader, n_epochs=50, T_warmup=10):
    """KL warmup over T_warmup epochs"""
    step = 0
    steps_per_warmup = T_warmup * len(loader)
    for epoch in range(n_epochs):
        for x, _ in loader:
            beta = min(1.0, step / steps_per_warmup)
            x = x.cuda().view(-1, 784)
            x_recon, mu, logvar = model(x)
            recon = F.binary_cross_entropy_with_logits(
                x_recon, x, reduction='sum') / x.size(0)
            kl = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp()).sum() / x.size(0)
            loss = recon + beta * kl
            opt.zero_grad(); loss.backward(); opt.step()
            step += 1
        if epoch % 5 == 0:
            print(f"Epoch {epoch}: β = {beta:.2f}, recon = {recon:.3f}, kl = {kl:.3f}")

# 비교: with vs without annealing
# - 그냥 ELBO 학습: language VAE 에서 거의 항상 collapse
# - Annealing: collapse 방지, informative latent 학습
```

### 실험 3 — Free Bits 구현

```python
def free_bits_kl(mu, logvar, free_bits=2.0):
    """Per-dim KL with free bits threshold"""
    kl_dim = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp())
    # max with free_bits per dimension
    kl_dim_clamped = torch.clamp(kl_dim, min=free_bits)
    return kl_dim_clamped.sum(-1)   # per sample

# Loss
def free_bits_loss(x, x_recon, mu, logvar, free_bits=2.0):
    recon = F.binary_cross_entropy_with_logits(x_recon, x, reduction='none').sum(-1)
    kl = free_bits_kl(mu, logvar, free_bits)
    return (recon + kl).mean()

# 효과: 각 dim 이 적어도 2 nats 의 정보 사용 보장
# Active dimensions ≈ z_dim
```

### 실험 4 — Strong vs Weak Decoder 비교

```python
class WeakDecoder(nn.Module):
    """단순 MLP — collapse 없음"""
    def __init__(self, z_dim=32, h=400):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(z_dim, h), nn.ReLU(),
            nn.Linear(h, 784),
        )
    def forward(self, z):
        return self.net(z)

class StrongDecoder(nn.Module):
    """PixelCNN-style autoregressive — collapse 가능성 높음"""
    def __init__(self, z_dim=32):
        super().__init__()
        self.z_to_h = nn.Linear(z_dim, 64 * 28 * 28)
        self.pixelcnn = SimplePixelCNN()  # Ch2-02 의 구현
    def forward(self, z, x):
        h_z = self.z_to_h(z).view(-1, 64, 28, 28)
        return self.pixelcnn(x) + h_z   # condition on z
    # x 가 input — autoregressive

# 두 decoder 로 VAE 학습
# Weak: KL > 0, active dims 많음
# Strong: KL → 0, active dims 적음 (collapse)
```

---

## 🔗 이론과 실전의 간극

### 1. Language VAE 의 어려움

자연어의 강한 sequential structure → LSTM/Transformer decoder 가 매우 강력 → collapse 거의 보장.

해결책 (다 부분적):
- **Aggressive KL annealing**: $T_\text{warmup}$ 길게
- **Word dropout**: input 의 일부 단어 랜덤 마스킹 → decoder 가 latent 의존 강화
- **Free bits $\lambda \approx 5$ nats**: 과감한 minimum capacity
- **Flow posterior**: factored Gaussian 의 한계 극복

그래도 image VAE 에 비해 latent 의 quality 떨어짐.

### 2. Diffusion 으로의 이동

VAE 의 collapse 와 ELBO looseness 이 결정적 한계 → 2020 년 이후 image generation 은 대부분 diffusion 으로 이동.

VAE 의 잔존 응용:
- **VQ-VAE** + AR decoder (DALL-E, Parti): discrete latent 가 collapse 회피
- **Latent Diffusion** (Stable Diffusion): VAE encoder 로 압축 + diffusion in latent space
- **β-VAE for representation learning**: disentanglement, anomaly detection

### 3. Hierarchical VAE 의 진화

NVAE (Vahdat 2020), VDVAE (Child 2021) 등 deep hierarchical VAE 가 collapse 회피하며 SOTA NLL. 핵심:
- Multi-level latent $z_1, z_2, \ldots, z_L$
- Top-down generative + bottom-up inference
- Skip connection + attention

ImageNet-32 NLL 에서 diffusion 과 경쟁력. 단, 구현 복잡.

---

## ⚖️ 가정과 한계

| 가정 | 한계 |
|------|------|
| Strong decoder = collapse | 항상 그런 것은 아님 (architecture, hyperparameter dependent) |
| KL annealing 이 collapse 막음 | 일부만 — strong decoder 에서는 후에 다시 collapse 가능 |
| Free bits 가 active dim 보장 | $\lambda$ tuning 필요, 너무 크면 reconstruction 손해 |
| δ-VAE 가 KL 0 막음 | Encoder family 제약 추가, expressiveness 감소 |
| Single solution exists | 다양한 해결책의 조합이 일반적, 단일 silver bullet 없음 |

---

## 📌 핵심 정리

$$\boxed{\text{Posterior Collapse: } q_\phi(z|x) \to p(z) \text{ for all } x \Rightarrow \text{Latent ignored}}$$

| 현상 | 진단 |
|------|------|
| **Active dims** | KL > threshold 인 dim 수 — collapse 시 0 |
| **Mean KL** | < 0.5 nats 면 collapse 의심 |
| **Sampling quality** | $z \sim p(z)$ 의 sample 이 random/blurry |

| 해결책 | 메커니즘 | 한계 |
|--------|---------|------|
| **KL annealing** | $\beta(t) = \min(1, t/T)$ — 초기 KL 끔 | Strong decoder 에서 후에 회귀 가능 |
| **Free bits** | $\max(\lambda, \text{KL}_i)$ per dim | $\lambda$ tuning, recon trade-off |
| **δ-VAE** | $\sigma^2_\phi \leq 1 - \delta$ — KL > 0 강제 | Encoder family 제약 |
| **Word dropout** | Input 마스킹 → decoder 약화 | Language 만 |
| **Flow posterior** | Factored Gaussian 보다 expressive $q$ | 구현 복잡 |
| **Hierarchical VAE** | Multi-level latent, skip | Architecture 복잡 |
| **Diffusion** | Chain 분해, no latent to collapse | 다른 model family |

---

## 🤔 생각해볼 문제

**문제 1** (기초): Free bits $\lambda = 2$ nats 일 때, ELBO 최적해는 어떻게 변하는가? Active dim 수의 hard lower bound 가 있는가?

<details>
<summary>해설</summary>

Free bits loss: $L = -\text{recon} + \sum_i \max(\lambda, \text{KL}_i)$.

**Optimal**: $\text{KL}_i < \lambda$ 인 dim 에서 partial gradient = 0 → 해당 dim 의 encoder/decoder 가 KL 줄이는 incentive 없음. 이론상 그 dim 은 random initialization 그대로 유지.

**Active dim hard bound**: 없음. 모든 dim 이 KL = $\lambda$ 이면 active 라고 봐도 무방, 모든 dim 이 KL < $\lambda$ 이면 inactive — 의미 있는 정보 없음.

**실전 결과**:
- $\lambda = 0$: 표준 ELBO, collapse 가능
- $\lambda = 0.5$: light, soft collapse 방지
- $\lambda = 2$ - 5: 강한 minimum capacity, collapse 거의 막음
- $\lambda > 5$: 모든 dim 사용 강제, recon quality 손해

**시사점**: $\lambda$ 가 dataset 과 z_dim 에 따라 tuning 필요. 일반적 시작점 $\lambda = 1$ - 2 nats per dim.

</details>

**문제 2** (심화): Language VAE 에서 word dropout (input 의 일부 단어를 [UNK] 로 대체) 가 collapse 를 어떻게 막는지 설명하라.

<details>
<summary>해설</summary>

**Standard language VAE**: input $x = (w_1, \ldots, w_T)$, decoder 는 LM $p(w_t | w_{<t}, z)$. Decoder 가 $w_{<t}$ 로 충분히 $w_t$ 예측 가능 → $z$ 무시 가능 → collapse.

**Word dropout**: training 시 $w_i$ 의 일부 (e.g., 30%) 를 [UNK] 로 마스킹. Decoder input 이 $\tilde x = (\ldots, [UNK], \ldots, w_t)$ — 정보 부족.

**효과**:
1. Decoder 가 $w_{<t}$ 만으로 $w_t$ 예측 어려움 → $z$ 정보 활용 incentive
2. Encoder 가 $z$ 에 informative content 넣어야 reconstruction 가능
3. **Encoder dependence on $x$**: $z = q_\phi(z | x_\text{full})$ — full input 봄 (NO dropout in encoder), 따라서 encoder 의 $z$ 가 dropout 된 단어들의 정보 supplement
4. **Decoder 가 두 source 의존**: $w_{<t}$ (incomplete) + $z$ (compensates dropouts)

**구체적**: Bowman 2016 의 SST language VAE 에서 word dropout p=0.5 + KL annealing 으로 active KL 약 2-5 nats 달성. 없으면 KL ≈ 0.

**한계**:
- Dropout rate 너무 크면 generation quality 손해
- Long-context 에서는 효과 감소 (multi-token dropout 필요)
- 본질적으로 collapse 의 symptom 만 부분 완화, root cause 가 architecture

**현대적 대안**: language VAE 보다는 LLM + 명시적 latent (e.g., topic models, condition embeddings).

</details>

**문제 3** (논문 비평): Diffusion model 이 posterior collapse 가 없다는 주장의 정확한 의미는? VAE 의 hierarchical 구조 (NVAE, VDVAE) 도 collapse 가 적은데, 이 둘의 차이는?

<details>
<summary>해설</summary>

**Diffusion 의 collapse 부재**:
- Latent variables $x_1, \ldots, x_T$ 가 **fixed forward process** 의 결과
- Encoder 가 학습 안 됨 → $q(x_t | x_{t-1})$ 고정
- 따라서 "encoder ignores input" 이라는 mode 자체가 정의 안 됨
- 각 step 의 reverse $p_\theta(x_{t-1} | x_t)$ 가 supervised learning (denoising)

**Hierarchical VAE (NVAE, VDVAE)**:
- Multi-level latent $z_1, z_2, \ldots, z_L$
- Top-down generative + bottom-up inference
- 일부 level 이 collapse 해도 다른 level 이 보완
- Skip connection 으로 encoder 가 모든 level 정보 공유
- Stochastic + deterministic feature paths

**공통점**: 둘 다 "single-shot" VAE 보다 collapse 가 적음 — chain/hierarchical 분해가 효과적.

**차이점**:
1. **Latent 의 의미**: Diffusion 의 $x_t$ 는 $x_0$ 의 noised version (interpretable). VAE 의 $z_l$ 은 abstract.
2. **Forward process**: Diffusion 은 fixed (no learning). VAE 는 모든 level 학습.
3. **Sample quality**: Diffusion 이 image 에서 우월. VAE 는 likelihood 우월 가능 (NVAE 의 ImageNet-32 SOTA).
4. **Sampling speed**: Diffusion 느림 (1000 step), VAE 빠름 (single decoder pass).

**현대적 통합**: Latent Diffusion = VAE encoder (compression) + Diffusion in latent space — 두 패러다임의 장점 결합.

**시사점**: Posterior collapse 는 single-shot bottleneck 의 fundamental issue, **chain 또는 hierarchy 분해**로 해결. Architecture 의 깊은 통찰.

</details>

---

<div align="center">

[◀ 이전 (03. β-VAE)](./03-beta-vae-ib.md) | [📚 README](../README.md) | [다음 ▶ (05. VQ-VAE)](./05-vq-vae.md)

</div>
