# 05. VQ-VAE 와 Discrete Latent (van den Oord 2017)

## 🎯 핵심 질문

- 왜 continuous latent 대신 discrete codebook 을 사용하는가? Continuous VAE 의 어떤 한계를 해결하는가?
- Vector Quantization (VQ) 의 정확한 수학 — $z_q = e_{k^*}, k^* = \arg\min_k \|z_e - e_k\|$ 의 의미와 미분가능성?
- Straight-Through Estimator (STE) 가 어떻게 non-differentiable argmin 을 통과시키는가? Bias 의 영향은?
- VQ-VAE 의 commitment loss $\|z_e - \text{sg}(e_{k^*})\|^2$ + codebook loss $\|\text{sg}(z_e) - e_{k^*}\|^2$ 의 분리 이유는?
- DALL-E, Jukebox, MAGVIT 가 왜 모두 VQ-VAE + AR Transformer 구조인가?

---

## 🔍 왜 VQ-VAE 가 결정적 발명인가

van den Oord 2017 "Neural Discrete Representation Learning" 의 VQ-VAE:

1. **Posterior collapse 문제 해결** — discrete codebook 이 always informative
2. **Tokenization for AR** — 이미지를 token sequence 로 변환, GPT-style modeling 가능
3. **DALL-E, Stable Diffusion, MUSE 의 기반** — 현대 multimodal generation 의 첫 단계
4. **Compression** — VQ-VAE encoder 가 256×256 → 32×32 grid of 8K vocabulary 로 압축

핵심 아이디어:
- Encoder 가 continuous $z_e \in \mathbb{R}^d$ 출력
- Codebook $\{e_1, \ldots, e_K\} \subset \mathbb{R}^d$ 의 가장 가까운 entry 로 quantize: $z_q = e_{k^*}$
- Decoder 가 $z_q$ 로부터 reconstruction
- 미분 안 되는 argmin → Straight-Through Estimator

이 단순한 trick 이 후속 모델 폭발의 기반. Latent Diffusion 도 VQ-VAE encoder 사용. 이 문서에서는 VQ-VAE 의 수학과 STE, codebook 학습, 그리고 후속 영향을 다룹니다.

---

## 📐 수학적 선행 조건

- 이전 문서들: 01-elbo, 02-reparameterization, 04-posterior-collapse
- [CNN Deep Dive](https://github.com/iq-ai-lab/cnn-deep-dive): Encoder/decoder architecture
- [Probability Theory Deep Dive](https://github.com/iq-ai-lab/probability-theory-deep-dive): Categorical distribution

---

## 📖 직관적 이해

### "어휘로 이미지를 표현"

VQ-VAE 의 핵심: 이미지를 **discrete token sequence** 로 표현. 비유:

- 이미지 = 그림
- Encoder = "그림을 단어들로 묘사" — 각 영역에 가장 적합한 "단어" 부여
- Codebook = 사용 가능한 "단어 사전" (예: 8192 개)
- Decoder = "단어들로부터 이미지 재구성"

이것이 자연어와 같은 abstract representation. 그 후 GPT 같은 AR model 이 이 token sequence 를 학습 → 이미지 생성.

### "왜 Continuous 대신 Discrete?"

**Posterior collapse 회피**: continuous Gaussian latent 는 prior 에 빠지는 mode 가 있음. Discrete codebook 은 entry 가 explicit, "사용 안 함" 이 어색.

**AR modeling 용이**: Discrete token sequence 가 GPT-style modeling 에 직접 호환. Continuous 는 normalize 어려움.

**Compression**: $32 \times 32$ codebook indices (8K vocab) = $32 \cdot 32 \cdot \log_2 8192 = 13312$ bits, 원본 $256 \cdot 256 \cdot 24 = 1.5M$ bits → **100배 이상 압축**.

### Vector Quantization 의 의미

연속 공간 $\mathbb{R}^d$ 를 $K$ 개의 cell 로 나눔 (Voronoi tessellation). Encoder output $z_e$ 가 어느 cell 에 속하는지 → 그 cell 의 representative $e_k$ 사용.

K-means clustering 과 유사 — codebook entry 들이 데이터의 "centroids" 역할.

---

## ✏️ 엄밀한 정의·정리

### 정의 5.1 — VQ-VAE Architecture

**Encoder** $z_e = E_\phi(x) \in \mathbb{R}^{H' \times W' \times D}$ — spatial feature map.

**Codebook** $\{e_1, \ldots, e_K\}, e_k \in \mathbb{R}^D$ — learnable.

**Quantization**: 각 spatial 위치 $(i, j)$ 에서:

$$k^*_{ij} = \arg\min_k \|z_e^{(i,j)} - e_k\|_2$$

$$z_q^{(i,j)} = e_{k^*_{ij}}$$

**Decoder** $\hat x = D_\theta(z_q)$.

### 정의 5.2 — VQ-VAE Loss

$$\mathcal{L} = \underbrace{\|x - \hat x\|^2}_{\text{Reconstruction}} + \underbrace{\|\text{sg}(z_e) - e_{k^*}\|^2}_{\text{Codebook}} + \underbrace{\beta \|z_e - \text{sg}(e_{k^*})\|^2}_{\text{Commitment}}$$

여기서 $\text{sg}(\cdot)$ 는 stop-gradient (no backprop through this).

**Reconstruction**: encoder + decoder 학습.

**Codebook**: codebook entries 를 encoder output 에 가까이.

**Commitment**: encoder output 이 codebook 에서 멀어지지 않게.

$\beta \in [0.25, 2]$ (원 논문은 0.25).

### 정리 5.3 — Straight-Through Estimator (STE)

Quantization $z_q = e_{k^*}$ 는 미분 불가능 ($\arg\min$). STE:

**Forward**: $z_q = e_{k^*}$.

**Backward**: $\partial z_q / \partial z_e = 1$ (identity gradient pass-through).

코드:
```python
z_q = z_e + (e[k_star] - z_e).detach()
# Forward: z_q == e[k_star]
# Backward: ∂z_q / ∂z_e = 1
```

**Bias**: STE 는 unbiased 가 아님 (true gradient ≠ 1), 하지만 실전에서 잘 작동.

### 정리 5.4 — Stop-Gradient 의 역할 분리

세 loss 의 backprop 영향:

**Reconstruction $\|x - \hat x\|^2$**: STE 통해 encoder + decoder 학습.

**Codebook $\|\text{sg}(z_e) - e_{k^*}\|^2$**: $\text{sg}(z_e)$ 가 freeze, codebook $e$ 만 학습. K-means update 와 유사.

**Commitment $\beta \|z_e - \text{sg}(e_{k^*})\|^2$**: $\text{sg}(e_{k^*})$ freeze, encoder $z_e$ 만 학습. Encoder 출력이 codebook 에서 멀어지지 않게.

**왜 sg 가 필요한가**: 각 loss 가 **다른 parameter** 만 학습하게 분리. 만약 sg 없으면 reconstruction loss 가 codebook 도 직접 학습 → unstable.

### 정리 5.5 — Codebook Update 의 EMA Alternative

Codebook loss 대신 **Exponential Moving Average** 로 codebook update:

$$N_k^{(t)} = \gamma N_k^{(t-1)} + (1 - \gamma) n_k^{(t)}$$
$$m_k^{(t)} = \gamma m_k^{(t-1)} + (1 - \gamma) \sum_{(i,j): k^*_{ij} = k} z_e^{(i,j)}$$
$$e_k^{(t)} = m_k^{(t)} / N_k^{(t)}$$

여기서 $n_k^{(t)}$ 는 batch 에서 $e_k$ 가 선택된 횟수, $\gamma \in [0.99, 0.999]$.

**장점**: hyperparameter $\beta$ tuning 회피, 더 안정적 codebook update. VQ-VAE-2 와 후속 모델의 표준.

### 정리 5.6 — VQ-VAE 의 ELBO 해석

VQ-VAE 의 latent 분포: $q(z|x) = \delta(z - z_q)$ — Dirac (deterministic encoder + quantization). Prior $p(z) = $ uniform on codebook.

$$\text{KL}(q \| p) = -\log(1/K) = \log K$$ (constant — entry 가 uniform 이므로)

따라서 KL 이 constant, ELBO 의 KL 항이 없어짐. 이것이 collapse 가 없는 이유 — KL 이 fixed.

**대안**: codebook prior 를 학습 (e.g., AR over latent indices) → second-stage AR Transformer 의 motivation.

---

## 🔬 증명 및 수학적 유도

### 유도 1 — STE 의 정당성 (informal)

True gradient $\partial z_q / \partial z_e$: $z_q$ 가 piecewise constant ($z_e$ 가 cell boundary 안에서 변해도 $z_q$ 같음, boundary 넘으면 jump). Almost everywhere 0, jump points 에서 정의 안 됨.

**STE 의 approximation**: identity gradient. 이는 gradient 의 "direction" 을 유지 (encoder 출력 변화가 quantized output 의 변화로 reflected).

**왜 작동하는가** (heuristic):
- Encoder 가 codebook 에 잘 맞도록 학습 → cell boundary 에서 멀어짐
- 따라서 작은 perturbation 이 quantization 결과 안 바꿈
- STE 가 정확한 gradient 와 유사

### 유도 2 — Commitment Loss 의 수학적 motivation

$z_e$ 가 codebook entry 와 멀어지면:
1. Quantization error $\|z_e - z_q\|$ 큼 → reconstruction 어려움
2. STE gradient 가 부정확 — encoder 가 wild 하게 움직일 수 있음

Commitment loss $\|z_e - \text{sg}(e_{k^*})\|^2$ 가 encoder 를 codebook 에 anchor.

**$\beta$ 의 역할**:
- $\beta$ 작음: encoder 자유, 더 expressive 하지만 codebook 사용 sparse 위험
- $\beta$ 큼: encoder 가 codebook 에 수렴, 안정적이지만 표현력 손실
- 일반적 $\beta = 0.25$ 가 잘 작동

### 유도 3 — Codebook Collapse

Posterior collapse 와 유사한 다른 현상: **codebook collapse** — 일부 codebook entry 만 사용, 나머지는 dead.

**원인**:
- 초기화 시 일부 entry 가 데이터 분포에 가까움
- 가까운 entries 만 빈번히 선택 → reconstruction loss 만 학습
- Codebook loss 가 dead entries update 못함 (해당 batch 에 선택 안 됨)

**해결**:
- **EMA update** (Razavi 2019): 자주 선택 안 된 entry 도 점진 update
- **Codebook reset**: Dead entry 를 random 으로 reinit
- **Random restart**: $\epsilon$ probability 로 random entry 사용
- **Codebook regularization**: entropy 페널티 (Yu 2022)

### 유도 4 — Hierarchical VQ-VAE-2

VQ-VAE 의 한 level 은 256×256 → 32×32 압축 정도. 더 압축하려면 hierarchical:

**VQ-VAE-2** (Razavi 2019):
- Top level: 32×32 → 8×8 (global structure)
- Middle level: 32×32 (mid-level details)
- Bottom level: 64×64 (fine details)

각 level 의 token sequence 에 대해 별도 AR Transformer. 256×256 high-res 이미지 생성 가능.

### 유도 5 — VQ-VAE → DALL-E 의 Pipeline

**Stage 1** (VQ-VAE): 이미지 ↔ token sequence 학습. Reconstruction loss 만.

**Stage 2** (AR over latents): GPT-style transformer 가 $p(z_1, \ldots, z_n)$ 학습. 또는 text + image conditional $p(z_\text{img} | z_\text{text})$.

**Inference**: 
1. Text → text tokens
2. AR Transformer: text → image tokens (sampling)
3. VQ-VAE decoder: image tokens → image

이 two-stage 가 DALL-E (2021), MUSE (2023), Parti (2022) 의 공통 구조.

---

## 💻 실험 재현 / PyTorch 구현

### 실험 1 — VQ-VAE 구현

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class VectorQuantizer(nn.Module):
    def __init__(self, num_embeddings=512, embedding_dim=64, beta=0.25):
        super().__init__()
        self.K = num_embeddings
        self.D = embedding_dim
        self.beta = beta
        # Codebook
        self.embeddings = nn.Embedding(num_embeddings, embedding_dim)
        self.embeddings.weight.data.uniform_(-1.0/num_embeddings, 1.0/num_embeddings)

    def forward(self, z_e):
        # z_e: [B, D, H, W]
        z_e_perm = z_e.permute(0, 2, 3, 1).contiguous()   # [B, H, W, D]
        flat = z_e_perm.view(-1, self.D)                   # [BHW, D]

        # Distance to codebook entries
        dist = (flat.pow(2).sum(1, keepdim=True)
              - 2 * flat @ self.embeddings.weight.t()
              + self.embeddings.weight.pow(2).sum(1))      # [BHW, K]
        encoding_idx = dist.argmin(1)                      # [BHW]

        z_q = self.embeddings(encoding_idx).view_as(z_e_perm)
        z_q = z_q.permute(0, 3, 1, 2).contiguous()         # [B, D, H, W]

        # Losses
        codebook_loss = F.mse_loss(z_q, z_e.detach())      # codebook learns
        commitment_loss = F.mse_loss(z_e, z_q.detach())    # encoder commits
        loss = codebook_loss + self.beta * commitment_loss

        # Straight-through estimator
        z_q = z_e + (z_q - z_e).detach()

        return z_q, loss, encoding_idx.view(z_e.shape[0], z_e.shape[2], z_e.shape[3])

class VQVAE(nn.Module):
    def __init__(self, K=512, D=64):
        super().__init__()
        # Encoder: 28x28 → 7x7 with D channels
        self.enc = nn.Sequential(
            nn.Conv2d(1, 32, 4, 2, 1), nn.ReLU(),       # 14x14
            nn.Conv2d(32, 64, 4, 2, 1), nn.ReLU(),      # 7x7
            nn.Conv2d(64, D, 1),                         # [B, D, 7, 7]
        )
        self.vq = VectorQuantizer(K, D)
        # Decoder
        self.dec = nn.Sequential(
            nn.Conv2d(D, 64, 1), nn.ReLU(),
            nn.ConvTranspose2d(64, 32, 4, 2, 1), nn.ReLU(),
            nn.ConvTranspose2d(32, 1, 4, 2, 1),
        )

    def forward(self, x):
        z_e = self.enc(x)
        z_q, vq_loss, indices = self.vq(z_e)
        x_recon = self.dec(z_q)
        return x_recon, vq_loss, indices

# 훈련 (MNIST)
loader = torch.utils.data.DataLoader(
    torchvision.datasets.MNIST('~/data', download=True, transform=torchvision.transforms.ToTensor()),
    batch_size=128, shuffle=True
)
model = VQVAE().cuda()
opt = torch.optim.Adam(model.parameters(), lr=1e-3)

for epoch in range(20):
    for x, _ in loader:
        x = x.cuda()
        x_recon, vq_loss, _ = model(x)
        recon = F.mse_loss(x_recon, x)
        loss = recon + vq_loss
        opt.zero_grad(); loss.backward(); opt.step()
    print(f"Epoch {epoch}: recon = {recon:.4f}, vq = {vq_loss:.4f}")
```

### 실험 2 — Codebook 사용률 측정

```python
@torch.no_grad()
def measure_codebook_usage(model, loader):
    model.eval()
    counts = torch.zeros(model.vq.K)
    for x, _ in loader:
        _, _, indices = model(x.cuda())
        idx_flat = indices.cpu().flatten()
        counts.index_add_(0, idx_flat, torch.ones_like(idx_flat, dtype=torch.float))
    used = (counts > 0).sum().item()
    print(f"Used codebook entries: {used}/{model.vq.K} ({used/model.vq.K*100:.1f}%)")
    print(f"Top 10 usage: {counts.sort(descending=True).values[:10].int().tolist()}")
    return counts

# 일반적: 잘 학습된 VQ-VAE 가 80-95% codebook 사용
# Codebook collapse: < 30% 사용 — EMA update 또는 reset 필요
```

### 실험 3 — Two-Stage: VQ-VAE + AR over Tokens

```python
# Stage 1: VQ-VAE 학습 (위와 동일)

# Stage 2: 이미지 → token → AR Transformer
@torch.no_grad()
def encode_to_tokens(model, x):
    z_e = model.enc(x)
    _, _, indices = model.vq(z_e)
    return indices   # [B, 7, 7]

# 모든 학습 데이터의 token sequence 추출
all_tokens = []
for x, _ in loader:
    tokens = encode_to_tokens(model, x.cuda()).flatten(1)   # [B, 49]
    all_tokens.append(tokens.cpu())
all_tokens = torch.cat(all_tokens)    # [N, 49]

# AR Transformer (mini GPT) 학습
gpt = MiniGPT(vocab_size=512, d_model=128, n_heads=4, n_layers=4, max_len=49+1).cuda()
# token sequence (BOS + 49 tokens) 의 next-token prediction

# Sampling: GPT 가 token 생성 → VQ-VAE decoder 가 image
@torch.no_grad()
def generate_image(gpt, vqvae, n=4):
    tokens = torch.zeros(n, 1, dtype=torch.long).cuda()    # BOS
    for _ in range(49):
        logits, _ = gpt(tokens)
        probs = torch.softmax(logits[:, -1], dim=-1)
        next_tok = torch.multinomial(probs, 1)
        tokens = torch.cat([tokens, next_tok], dim=-1)
    indices = tokens[:, 1:].view(n, 7, 7)
    z_q = vqvae.vq.embeddings(indices).permute(0, 3, 1, 2)
    return vqvae.dec(z_q)

# 이 pipeline 이 DALL-E, Parti, MUSE 의 핵심 구조
```

### 실험 4 — EMA Codebook Update

```python
class VectorQuantizerEMA(nn.Module):
    def __init__(self, K=512, D=64, beta=0.25, gamma=0.99):
        super().__init__()
        self.K, self.D = K, D
        self.beta, self.gamma = beta, gamma
        self.register_buffer('embeddings', torch.randn(K, D) * 0.01)
        self.register_buffer('N', torch.zeros(K))
        self.register_buffer('m', torch.zeros(K, D))

    def forward(self, z_e):
        flat = z_e.permute(0, 2, 3, 1).contiguous().view(-1, self.D)
        dist = (flat.pow(2).sum(1, keepdim=True)
              - 2 * flat @ self.embeddings.t()
              + self.embeddings.pow(2).sum(1))
        idx = dist.argmin(1)
        z_q_flat = self.embeddings[idx]
        z_q = z_q_flat.view_as(z_e.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)

        if self.training:
            # EMA update of codebook
            onehot = F.one_hot(idx, self.K).float()        # [N, K]
            self.N.mul_(self.gamma).add_((1 - self.gamma) * onehot.sum(0))
            self.m.mul_(self.gamma).add_((1 - self.gamma) * (onehot.t() @ flat))
            # Smoothed counts (Laplace)
            n = self.N.sum()
            counts = (self.N + 1e-5) / (n + self.K * 1e-5) * n
            self.embeddings = self.m / counts.unsqueeze(1)

        # Commitment only (codebook 자동 update 됨)
        commitment = F.mse_loss(z_e, z_q.detach()) * self.beta
        z_q = z_e + (z_q - z_e).detach()                    # STE
        return z_q, commitment, idx.view(z_e.shape[0], z_e.shape[2], z_e.shape[3])
```

---

## 🔗 이론과 실전의 간극

### 1. Codebook Collapse 의 실전 처치

원 VQ-VAE 의 codebook loss + commitment 가 collapse 흔함. 현대적 처치:
- **EMA update**: Razavi 2019 (VQ-VAE-2)
- **Codebook reset**: Williams 2020 의 random restart
- **Vector quantization with normalization** (Yu 2022 ViT-VQGAN): l2-normalize codebook
- **Codebook entropy regularization**: 균일 사용 강제

### 2. VQ-VAE 의 후속 발전

- **VQ-VAE-2** (Razavi 2019): hierarchical, ImageNet 256×256
- **VQGAN** (Esser 2021): adversarial loss + perceptual loss → sharper reconstruction
- **ViT-VQGAN** (Yu 2022): Transformer encoder/decoder
- **MAGVIT** (Yu 2023): Video tokenization
- **MUSE** (Chang 2023): Masked AR over VQ-VAE tokens
- **EnCodec** (Défossez 2022): Audio tokenization

### 3. Latent Diffusion Models 의 VQ-VAE 사용

**Stable Diffusion** (Rombach 2022): VQ-VAE encoder 로 이미지 → latent (continuous, not quantized) → diffusion in latent space → VQ-VAE decoder.

VQ-VAE 의 quantization 단계 생략 — 단지 encoder/decoder architecture 활용 (compression). 이를 KL-regularized AE 로 부름.

**Trade-off**: discrete tokens (DALL-E, Parti) vs continuous latent (Stable Diffusion). Discrete 는 AR 과 호환, continuous 는 diffusion 과 호환.

---

## ⚖️ 가정과 한계

| 가정 | 한계 |
|------|------|
| STE 가 valid gradient approximation | Biased, no convergence guarantee |
| Codebook 이 데이터 매니폴드 cover | Codebook collapse 위험 |
| Discrete latent 가 continuous 보다 우월 | Domain dependent — image 는 둘 다 작동 |
| Stage 1 + Stage 2 분리 | Joint training 어려움 (gradient flow 막힘) |
| Reconstruction quality | VQ 의 quantization error 가 lossy |

---

## 📌 핵심 정리

$$\boxed{z_q^{(i,j)} = e_{k^*_{ij}}, \quad k^*_{ij} = \arg\min_k \|z_e^{(i,j)} - e_k\|_2}$$

$$\boxed{\mathcal{L} = \|x - \hat x\|^2 + \|\text{sg}(z_e) - e_{k^*}\|^2 + \beta \|z_e - \text{sg}(e_{k^*})\|^2}$$

| 구성요소 | 역할 |
|---------|------|
| **Encoder $E_\phi$** | $x \to z_e$ continuous |
| **Codebook $\{e_k\}$** | Discrete vocabulary |
| **Quantize** | Nearest neighbor, $z_q = e_{k^*}$ |
| **STE** | Forward $z_q$, backward $\partial / \partial z_e = 1$ |
| **Codebook loss** | $z_e$ freeze, codebook 학습 |
| **Commitment loss** | Codebook freeze, encoder anchor |
| **Decoder** | $z_q \to \hat x$ |

| 모델 | 활용 |
|------|------|
| **VQ-VAE-2** | Hierarchical, ImageNet 256 |
| **VQGAN** | + adversarial, sharper |
| **DALL-E** | VQ-VAE + AR Transformer (text → image) |
| **Parti** | Pure AR token model |
| **Stable Diffusion** | VQ-VAE 의 encoder 만 (compression) |
| **MUSE** | Masked AR over VQ-tokens |

---

## 🤔 생각해볼 문제

**문제 1** (기초): VQ-VAE 의 KL term 이 constant 인 이유와, 이것이 posterior collapse 회피에 어떻게 기여하는지 설명하라.

<details>
<summary>해설</summary>

VQ-VAE 의 latent: $z = $ discrete codebook index. Encoder $q(z|x) = \delta(z - k^*)$ — Dirac (deterministic). Prior $p(z) = $ uniform on $\{1, \ldots, K\}$.

$$\text{KL}(q \| p) = -\log(1/K) = \log K \quad (\text{constant})$$

ELBO 의 KL 항은 변하지 않음 → Loss 에 영향 없음 → optimization 이 reconstruction 만 추구.

**Posterior collapse 와의 관계**:
- Continuous VAE: KL 이 0 으로 가면 collapse → encoder 무력화
- VQ-VAE: KL 이 fixed → encoder 가 KL 줄이려는 incentive 없음 → 항상 informative
- 하지만 **codebook collapse** 라는 다른 문제 — 일부 entry 만 사용

**시사점**: VQ-VAE 는 posterior collapse 를 회피하지만 codebook collapse 라는 새 문제 도입. 둘은 다른 현상.

</details>

**문제 2** (심화): Stop-gradient 의 위치를 바꿔서 다음 loss 들의 행동을 비교하라:
(a) $\|z_e - e_{k^*}\|^2$ — sg 없음
(b) $\|\text{sg}(z_e) - e_{k^*}\|^2$ — codebook 만 학습 (원 논문)
(c) $\|z_e - \text{sg}(e_{k^*})\|^2$ — encoder 만 학습 (commitment)

<details>
<summary>해설</summary>

**(a) sg 없음**: Reconstruction loss 와 합쳐지면 encoder 와 codebook 모두 한 loss 가 학습. Gradient 가 두 방향에서 동시에 → unstable, codebook 이 reconstruction loss 의 wild gradient 받음.

**(b) Codebook loss**: $\partial / \partial z_e = 0$ (sg 으로 freeze), $\partial / \partial e = 2(e_{k^*} - z_e)$. Codebook 만 학습 → K-means 의 mean update 와 유사. Stable.

**(c) Commitment loss**: $\partial / \partial e = 0$, $\partial / \partial z_e = 2(z_e - e_{k^*})$. Encoder 만 학습 → encoder 출력이 codebook 에 anchored.

**왜 (b) + (c) 분리해야 하는가**:
- 각 loss 가 **다른 parameter 만 학습** → optimization 안정
- $\beta$ 로 두 항의 상대적 강도 조절 (encoder 의 explorativity vs commitment)
- Reconstruction loss 가 STE 로 encoder + decoder 학습 (codebook 은 reconstruction 으로부터는 안 받음)

**시사점**: VQ-VAE 의 sg pattern 은 단순한 trick 이 아니라 **각 component 의 학습 dynamics 분리**. 이 분리가 K-means-like update 를 NN context 로 가져옴.

</details>

**문제 3** (논문 비평): DALL-E (2021), Parti (2022), MUSE (2023), Stable Diffusion (2022) 의 architecture 를 비교하라. 모두 VQ-VAE-style tokenizer 를 사용하지만 후속 modeling 이 다르다 — AR vs masked AR vs diffusion. 각 선택의 장단점은?

<details>
<summary>해설</summary>

| 모델 | Tokenizer | Generative model | 특징 |
|------|-----------|------------------|------|
| **DALL-E** | dVAE | AR Transformer | First text-to-image at scale |
| **Parti** | ViT-VQGAN | AR Transformer | Pure AR, scaling law follow |
| **MUSE** | VQGAN | Masked AR (BERT-style) | Faster inference, parallel |
| **Stable Diffusion** | KL-regularized AE (VQ encoder, no quantization) | Diffusion in latent | High quality, slow |

**AR (DALL-E, Parti)**:
- ✅ Tractable likelihood
- ✅ Scaling law clear
- ❌ Sequential sampling 느림
- ❌ Token 단위 mistake cascade

**Masked AR (MUSE)**:
- ✅ Parallel decoding (BERT-style mask + iterative refine)
- ✅ Faster than AR
- ❌ NLL 측정 어려움
- ❌ Quality 가 AR 약간 부족

**Diffusion in latent (Stable Diffusion)**:
- ✅ High quality (현재 SOTA)
- ✅ Continuous latent — VQ 의 quantization error 없음
- ❌ Slow (50+ steps)
- ❌ Likelihood 평가 ELBO 만

**Tokenizer 선택**:
- **Discrete VQ tokens**: AR, masked AR 에 자연
- **Continuous latent**: diffusion 에 자연

**미래**: 두 패러다임의 hybrid. 예: Diffusion + Transformer (DiT, MMDiT in SD3), AR with continuous latent (Diffusion-LM).

**일반 원칙**: tokenizer (compression) 와 generation (modeling) 의 분리가 핵심 architectural insight — VQ-VAE 가 이 분리를 establish 한 결정적 작업.

</details>

---

<div align="center">

[◀ 이전 (04. Posterior Collapse)](./04-posterior-collapse.md) | [📚 README](../README.md) | [다음 ▶ (Ch4-01. Change of Variables)](../ch4-flow/01-change-of-variables.md)

</div>
