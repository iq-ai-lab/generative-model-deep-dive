# 04. Autoregressive Transformer — GPT as Generative Model

## 🎯 핵심 질문

- Self-attention + causal mask 가 어떻게 chain rule $p(x) = \prod p(x_i | x_{<i})$ 의 자연스러운 구현이 되는가?
- GPT 의 architecture 가 modality-agnostic 인 이유 — 왜 같은 구조로 text, image, audio, video 가 모두 작동하는가?
- Scaling law (Kaplan 2020, Hoffmann 2022) 의 power-law 형태와 그 함의는?
- Image-as-sequence (ImageGPT, Parti) 가 PixelCNN 대비 갖는 장점은? 왜 VQ-VAE + AR Transformer 가 DALL-E 의 기반이 되었는가?
- KV cache 가 어떻게 generation 의 비용을 $O(T^2)$ 에서 $O(T)$ 로 감소시키는가?

---

## 🔍 왜 GPT 가 "모든 것의 생성 모델" 이 되었는가

GPT (Radford 2018, 2019, Brown 2020) 는 **AR + Transformer + 큰 데이터 + 큰 모델** 의 단순한 조합. 하지만 그 결과는 혁명적:

1. **언어 생성** — GPT-3, GPT-4 의 자연어 능력
2. **이미지 생성** — ImageGPT (pixel-level), Parti (VQ-VAE + AR)
3. **오디오** — AudioLM, MusicLM
4. **비디오** — VideoGPT
5. **멀티모달** — GPT-4V, Gemini

핵심 통찰:
- **Self-attention** 이 임의 길이 dependency 를 한 layer 에서 처리
- **Causal mask** 가 chain rule factorization 을 자연스럽게 강제
- **Tokenization** 으로 임의 modality 를 sequence 로 변환
- **Scale** 이 emergent capability 를 이끌어냄

이 문서에서는 Transformer 의 구조를 **생성 모델 관점** 에서 다시 보고, 다양한 modality 로의 확장 그리고 scaling 의 수학을 다룹니다.

---

## 📐 수학적 선행 조건

- [Transformer Deep Dive](https://github.com/iq-ai-lab/transformer-deep-dive): Self-attention, multi-head, positional encoding
- 이전 문서들: 01-chain-rule, 02-pixelcnn, 03-wavenet
- [Optimization Theory Deep Dive](https://github.com/iq-ai-lab/optimization-theory-deep-dive): Adam, learning rate scheduling

---

## 📖 직관적 이해

### "Self-Attention 은 chain rule 의 자연 구현"

PixelCNN 은 dilated conv 로 RF 확장, WaveNet 은 dilated causal conv. Transformer 는 한 layer 에서 모든 이전 token 을 한 번에 보는 **direct** approach:

$$\text{attention}(Q, K, V) = \text{softmax}\left(\frac{Q K^\top}{\sqrt{d_k}}\right) V$$

**Causal mask**: $K^\top$ 의 미래 위치를 $-\infty$ 로 → softmax 후 0. 따라서 token $i$ 는 token $\leq i$ 만 attention.

이 단일 mechanism 으로:
- $RF = T$ (한 layer 가 전체 context 봄, $L \cdot k$ 의 conv 와 다름)
- Parallel training (모든 위치 동시)
- Sequential sampling (autoregressive nature)

### "왜 Modality-Agnostic 인가"

Transformer 에 입력은 단지 **임의 차원의 token sequence**. Modality 별 차이는 tokenizer 에만:

| Modality | Tokenizer | Vocabulary |
|----------|-----------|------------|
| Text | BPE (Byte Pair Encoding) | 50k |
| Image | VQ-VAE codebook | 8k |
| Audio | EnCodec, SoundStream | 1k-16k |
| Video | TVQ-VAE, MAGVIT | 8k |

Tokenize 하면 모두 sequence — 같은 Transformer architecture 사용 가능. 이것이 GPT-4V, Gemini, Claude 같은 multimodal 모델의 기반.

### Scaling Law

Kaplan 2020 의 발견: language modeling NLL 이 모델 크기 $N$, 데이터 $D$, compute $C$ 에 대해 **power law**:

$$L(N) \approx \left(\frac{N_c}{N}\right)^{\alpha_N}, \quad \alpha_N \approx 0.076$$

수십 년 간의 incremental improvement 가 단지 scaling 의 다른 점 — 새 architecture 보다 더 큰 모델이 우월. Hoffmann 2022 (Chinchilla) 가 compute-optimal scaling 을 정정.

---

## ✏️ 엄밀한 정의·정리

### 정의 4.1 — Causal Self-Attention

Sequence $X \in \mathbb{R}^{T \times d}$, weight matrices $W_Q, W_K, W_V \in \mathbb{R}^{d \times d_k}$:

$$Q = X W_Q, \quad K = X W_K, \quad V = X W_V$$

$$\text{Attention}(X) = \text{softmax}\left(\frac{Q K^\top + M}{\sqrt{d_k}}\right) V$$

여기서 causal mask $M_{ij} = 0$ if $j \leq i$, else $-\infty$.

### 정의 4.2 — GPT Architecture

$L$ 개 transformer block, 각 block:

$$h^{(\ell)} = h^{(\ell-1)} + \text{MLP}(\text{LN}(h^{(\ell-1)} + \text{Attention}(\text{LN}(h^{(\ell-1)}))))$$

(Pre-LN architecture, GPT-2 부터 표준).

Final output: $\text{logits}_t = h^{(L)}_t W_\text{out}$, $p(x_t | x_{<t}) = \text{softmax}(\text{logits}_t)$.

### 정리 4.3 — Causal Mask 가 AR Factorization 을 보장

**주장**: Causal-masked self-attention stack 의 output $h^{(L)}_t$ 는 $x_{1:t}$ 의 함수, $x_{t+1:T}$ 와 무관.

**증명**: 귀납. Layer 1: attention output $h^{(1)}_t = \sum_{j \leq t} \alpha_{tj} V_j$ — $j \leq t$ 만. MLP 는 pointwise. 따라서 $h^{(1)}_t$ 가 $x_{1:t}$ 의 함수.

Layer $\ell$ 까지 $h^{(\ell-1)}$ 가 그렇다고 가정. Layer $\ell$ 의 attention 도 같은 causal mask, 따라서 $h^{(\ell)}_t$ 도 $h^{(\ell-1)}_{1:t}$ 의 함수 = $x_{1:t}$ 의 함수. $\square$

### 정리 4.4 — KV Cache 의 Sampling 가속

Naive sampling: 각 step $t$ 마다 $X_{1:t}$ 전체에 대해 self-attention → $O(t^2 d)$. 총 $\sum_t O(t^2) = O(T^3)$.

**KV Cache**: 각 step 에서 새 token 의 $K, V$ 만 계산하고 누적 → step 비용 $O(t \cdot d)$. 총 $O(T^2 d)$.

**증거**: $K_{1:t}, V_{1:t}$ 가 step $t-1$ 에서 이미 계산. Step $t$ 에서 추가만:

$$K_{1:t} = [K_{1:t-1}; k_t], \quad V_{1:t} = [V_{1:t-1}; v_t]$$

새 query $q_t$ 만으로 attention: $\text{softmax}(q_t K_{1:t}^\top / \sqrt{d_k}) V_{1:t}$.

### 정리 4.5 — Scaling Law (Kaplan 2020)

Language modeling cross-entropy loss $L$ 이 model size $N$ (parameters), dataset size $D$ (tokens), compute $C$ (FLOPs) 에 대해 power law:

$$L(N, D) = \left(\frac{N_c}{N}\right)^{\alpha_N} + \left(\frac{D_c}{D}\right)^{\alpha_D} + L_\infty$$

empirical fit: $\alpha_N \approx 0.076$, $\alpha_D \approx 0.095$, $L_\infty \approx$ 1 nat/token (data 의 entropy 한계).

**Compute-optimal**: 주어진 $C$ 에 대해 $N$ 과 $D$ 의 최적 분할. Chinchilla (Hoffmann 2022): $D / N \approx 20$ tokens/parameter (이전 추정 5×보다 많음).

### 정리 4.6 — Image-as-Sequence with VQ-VAE

이미지 $x \in \mathbb{R}^{H \times W \times 3}$ 를 VQ-VAE encoder 로 token sequence $z \in \{1, ..., K\}^{h \times w}$ 변환 ($h \cdot w \ll H \cdot W$). AR Transformer 가 token 분포 $p(z)$ 학습:

$$p(z_1, \ldots, z_n) = \prod_i p(z_i | z_{<i})$$

Sampling: $z \sim p_\theta(z)$, then $x = \text{Decoder}(z)$.

**비교**: PixelCNN ($H \cdot W \cdot 3 = 3072$ tokens for CIFAR), VQ-VAE+AR (예: $32 \times 32 = 1024$ tokens) — **3배 짧은 sequence**, longer-range dependency 모델링 쉬움.

---

## 🔬 증명 및 수학적 유도

### 유도 1 — Self-Attention 의 Receptive Field

표준 conv: stack $L$ layer with kernel $k$ → RF $= L (k - 1) + 1$.

Self-attention: 단일 layer 가 전체 context $T$ 를 봄. **모든 layer 가 RF $= T$**. 이 점이 long-range dependency 모델링의 결정적 장점.

대신 비용: $O(T^2 d)$ (attention matrix). $T = 2048$ 까지는 OK, $T = 1M$ 이면 prohibitive → sliding window, linear attention 등의 대안.

### 유도 2 — GPT-1, GPT-2, GPT-3, GPT-4 의 Scaling

| 모델 | Params | Tokens | $L \cdot d$ | NLL (PG-19) |
|------|--------|--------|-------------|------|
| GPT-1 (2018) | 117M | 5B | 12 × 768 | 1.95 |
| GPT-2 (2019) | 1.5B | 40B | 48 × 1600 | 1.55 |
| GPT-3 (2020) | 175B | 300B | 96 × 12288 | 1.20 |
| GPT-4 (est.) | 1T+ | 13T+ | (mixture-of-experts) | < 1.0 |

매 generation 의 NLL 감소 ≈ 0.4 nats/token. Power law 따름.

### 유도 3 — Image GPT 의 Pixel Sequence Modeling

ImageGPT (Chen 2020): $32 \times 32 \times 3$ image 를 $32 \cdot 32 \cdot 3 = 3072$ pixel sequence (each 256-way categorical) 로 modeling. 표준 GPT architecture, only tokenizer 만 다름.

**결과**: ImageGPT-L (1.4B params) achieves CIFAR-10 87.6% linear probe acc — pretrained representation 으로 분류 가능. 단, generation 은 PixelCNN 과 비슷한 quality, **VQ-VAE + AR** 의 Parti, DALL-E 가 더 우월.

### 유도 4 — Why VQ-VAE + AR > Pixel-Level AR

| Approach | Sequence Length | Vocabulary | RF coverage |
|----------|-----------------|------------|-------------|
| Pixel-level (ImageGPT) | $H \cdot W \cdot C$ ≈ 3072 | 256 | Hard for global |
| VQ-VAE + AR (Parti) | $h \cdot w$ ≈ 1024 | 8192 | Easy global, semantic tokens |

**시사점**:
- Shorter sequence → easier long-range modeling
- Semantic tokens → coarse-to-fine (token decoder 가 fine detail)
- DALL-E 의 zero-shot text-to-image 는 이 구조에서 가능

Diffusion model 은 다른 접근 (continuous denoising), 두 패러다임 공존.

### 유도 5 — Multimodal as Unified Tokenization

GPT-4V, Gemini: text + image + audio 를 unified token sequence:

```
[BOS] <image_tokens> [SEP] "Describe this:" <text_tokens> [EOS]
```

각 modality 의 tokenizer 가 분리, 그 후 단일 GPT 로 처리. Vocabulary 는 합집합. Cross-modal attention 자연 발생.

이것이 Multimodal generative AI 의 기본 paradigm.

---

## 💻 실험 재현 / PyTorch 구현

### 실험 1 — 미니 GPT 구현

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class CausalSelfAttention(nn.Module):
    def __init__(self, d_model, n_heads):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        self.qkv = nn.Linear(d_model, 3 * d_model)
        self.out = nn.Linear(d_model, d_model)

    def forward(self, x, kv_cache=None):
        B, T, D = x.shape
        qkv = self.qkv(x).reshape(B, T, 3, self.n_heads, self.d_k).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]   # [B, h, T, d_k]

        if kv_cache is not None:
            past_k, past_v = kv_cache
            k = torch.cat([past_k, k], dim=2)
            v = torch.cat([past_v, v], dim=2)
        new_cache = (k, v)

        # causal mask
        T_full = k.shape[2]
        mask = torch.triu(torch.full((T, T_full), float('-inf'), device=x.device),
                          diagonal=T_full - T + 1)
        scores = (q @ k.transpose(-2, -1)) / self.d_k**0.5 + mask
        attn = F.softmax(scores, dim=-1)
        out = (attn @ v).transpose(1, 2).reshape(B, T, D)
        return self.out(out), new_cache

class TransformerBlock(nn.Module):
    def __init__(self, d_model, n_heads, d_ff=None):
        super().__init__()
        d_ff = d_ff or 4 * d_model
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = CausalSelfAttention(d_model, n_heads)
        self.ln2 = nn.LayerNorm(d_model)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, d_ff), nn.GELU(),
            nn.Linear(d_ff, d_model),
        )

    def forward(self, x, kv_cache=None):
        attn_out, new_cache = self.attn(self.ln1(x), kv_cache)
        x = x + attn_out
        x = x + self.mlp(self.ln2(x))
        return x, new_cache

class MiniGPT(nn.Module):
    def __init__(self, vocab_size, d_model=128, n_heads=4, n_layers=4, max_len=256):
        super().__init__()
        self.tok_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Embedding(max_len, d_model)
        self.blocks = nn.ModuleList([
            TransformerBlock(d_model, n_heads) for _ in range(n_layers)
        ])
        self.ln = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size)

    def forward(self, x, kv_caches=None):
        B, T = x.shape
        positions = torch.arange(T, device=x.device).unsqueeze(0)
        h = self.tok_emb(x) + self.pos_emb(positions)
        new_caches = []
        for i, block in enumerate(self.blocks):
            kv = kv_caches[i] if kv_caches else None
            h, new_kv = block(h, kv)
            new_caches.append(new_kv)
        return self.head(self.ln(h)), new_caches

# Toy: Shakespeare 문자 수준 학습
# (실제 구현에서는 BPE tokenizer 사용)
```

### 실험 2 — KV Cache 의 Sampling 가속

```python
@torch.no_grad()
def generate_no_cache(model, prompt, max_new=50):
    tokens = prompt
    for _ in range(max_new):
        logits, _ = model(tokens)
        next_tok = logits[:, -1].argmax(-1, keepdim=True)
        tokens = torch.cat([tokens, next_tok], dim=-1)
    return tokens

@torch.no_grad()
def generate_with_cache(model, prompt, max_new=50):
    tokens = prompt
    logits, caches = model(tokens)
    output = [tokens]
    for _ in range(max_new):
        next_tok = logits[:, -1].argmax(-1, keepdim=True)
        output.append(next_tok)
        # 새 token 만 forward, cache 사용
        logits, caches = model(next_tok, caches)
    return torch.cat(output, dim=-1)

import time
prompt = torch.randint(0, 1000, (1, 10))

start = time.time()
out1 = generate_no_cache(model, prompt, 100)
t1 = time.time() - start

start = time.time()
out2 = generate_with_cache(model, prompt, 100)
t2 = time.time() - start

print(f"Without cache: {t1:.2f}s")
print(f"With cache:   {t2:.2f}s ({t1/t2:.1f}× speedup)")
# 일반적으로 5-10배 가속 (sequence 길이에 따라)
```

### 실험 3 — Scaling Law 시연

```python
# 다양한 model size 로 같은 데이터 학습 → final loss 비교
sizes = [(64, 2, 2), (128, 4, 4), (256, 4, 6), (512, 8, 8)]
final_losses = []
n_params = []

for d, h, L in sizes:
    model = MiniGPT(vocab_size=1000, d_model=d, n_heads=h, n_layers=L)
    n = sum(p.numel() for p in model.parameters())
    n_params.append(n)
    # ... training ...
    final_losses.append(final_loss)

import matplotlib.pyplot as plt
plt.loglog(n_params, final_losses, 'o-')
plt.xlabel('Parameters'); plt.ylabel('NLL')
plt.title('Scaling Law (toy)')
# 예상: log-log 직선 — power law
# 기울기 ≈ -0.076 (Kaplan 2020 의 alpha_N)
```

---

## 🔗 이론과 실전의 간극

### 1. Long Context 의 도전

표준 self-attention $O(T^2)$ — $T = 100$만 (책 한 권) 이상에서 prohibitive. 해결책:

- **Sliding window** (Longformer, BigBird): local + selected global
- **Sparse attention** (Reformer, LSH): hash-based
- **Linear attention** (Performer, Linformer): kernel approximation
- **Mamba, RWKV** (2023-24): SSM-based, $O(T)$
- **ALiBi** (Press 2022): position-aware bias for length extrapolation

GPT-4 turbo 는 128k context, Claude 200k+, Gemini 1.5 1M+ — 새로운 연구 영역.

### 2. Inference 비용

생성 model 의 inference cost:
- **Prefill** (prompt 처리): $O(T_p^2 \cdot N)$ — 한 번
- **Decode** (token by token): $O(T_d \cdot N)$ — sequential

Production 에서는 latency (decode) 가 주요 병목. Speculative decoding, quantization (GPTQ, AWQ), distillation 으로 개선.

### 3. Multimodal Tokenization 의 한계

Image 를 token 으로 quantize 하면 fine detail 손실. 두 접근:

- **Discrete tokens + AR** (Parti, MUSE): semantic tokens, fast inference
- **Continuous + Diffusion** (Stable Diffusion, DALL-E 3): high fidelity, slow

GPT-4V 는 vision encoder 의 continuous embedding 을 input 으로 받음 — token 화 없이 직접. 이것이 multimodal 의 미래.

---

## ⚖️ 가정과 한계

| 가정 | 한계 |
|------|------|
| $O(T^2)$ self-attention 가능 | Long context 에서 prohibitive — sparse/linear 대안 필요 |
| Power-law scaling 무한 지속 | Eventual data limit ($L_\infty$), capability plateau 가능 |
| Tokenization 이 자연스러움 | BPE 의 한계 (단어 단위 vs character), multimodal 의 different rate |
| Sequential decode | KV cache 후에도 latency 병목 |
| 단일 architecture for all | Domain-specific architecture (예: protein) 가 더 나을 수 있음 |
| Causal mask | Bidirectional context 활용 못 함 (BERT 와 trade-off) |

---

## 📌 핵심 정리

$$\boxed{\text{Attention}(X) = \text{softmax}\left(\frac{Q K^\top + M}{\sqrt{d_k}}\right) V \text{ — Causal mask } M}$$

$$\boxed{p_\theta(x_1, \ldots, x_T) = \prod_{t=1}^T p_\theta(x_t | x_{<t}), \text{ where } x_t \text{ is any modality}}$$

| 구성요소 | 역할 |
|---------|------|
| **Causal self-attention** | $RF = T$ at every layer, parallel training |
| **Causal mask** | Future token 차단, AR factorization |
| **Tokenizer** | Modality → discrete sequence (BPE, VQ-VAE, EnCodec) |
| **KV cache** | Sampling cost $O(T^2)$ → $O(T)$ effectively |
| **Scaling law** | $L(N, D) \propto N^{-0.076} + D^{-0.095}$ (Kaplan 2020) |
| **Multimodal** | Unified token sequence across modalities |

---

## 🤔 생각해볼 문제

**문제 1** (기초): Causal mask matrix $M$ 의 정확한 형태를 $T = 4$ 일 때 그려라. Softmax 후 attention weight 행렬은 어떤 모양인가?

<details>
<summary>해설</summary>

$M_{ij} = -\infty$ if $j > i$, else $0$. $T = 4$:

$$M = \begin{pmatrix} 0 & -\infty & -\infty & -\infty \\ 0 & 0 & -\infty & -\infty \\ 0 & 0 & 0 & -\infty \\ 0 & 0 & 0 & 0 \end{pmatrix}$$

Softmax 후 (각 row 가 sum 1, $-\infty$ 는 0):

$$A = \begin{pmatrix} 1 & 0 & 0 & 0 \\ a_{21} & a_{22} & 0 & 0 \\ a_{31} & a_{32} & a_{33} & 0 \\ a_{41} & a_{42} & a_{43} & a_{44} \end{pmatrix}, \quad \sum_j a_{ij} = 1$$

**Lower triangular** 형태. Token $i$ 가 token $\leq i$ 만 attend, 정확히 chain rule 의 dependency 와 일치.

</details>

**문제 2** (심화): Self-attention 의 $O(T^2)$ 비용을 줄이기 위해 sliding window attention (window size $w$) 을 사용하면, RF 와 layer 수의 관계는 어떻게 되는가? 표준 self-attention 과 비교하라.

<details>
<summary>해설</summary>

**Sliding window**: 각 token 이 직전 $w$ token 만 attend → 단일 layer RF = $w$.

**Stack $L$ layer**: 각 layer 가 RF 를 $w$ 만큼 확장. **Total RF = $L \cdot w$** (또는 그 근처, exact 공식은 padding 처리에 따라).

비교:
- 표준 self-attention: 단일 layer RF = $T$, $L$ layer RF = $T$ (포화)
- Sliding window: RF = $L \cdot w$, $T$ 도달하려면 $L \geq T / w$

**Trade-off**: window 작으면 cost $O(T \cdot w)$ 로 효율적이지만 RF 위해 깊은 모델 필요. Longformer, Mistral 의 sliding window attention 이 이 접근.

**Hybrid**: sliding window + 일부 global attention (BigBird, GPT-3.5) — 효율과 long-range 둘 다.

</details>

**문제 3** (논문 비평): Kaplan 2020 의 scaling law $L \propto N^{-0.076}$ 을 보면 모델이 무한히 커지면 loss 가 0 에 도달. 하지만 실제로 데이터의 entropy $L_\infty > 0$ 이 한계. Hoffmann 2022 (Chinchilla) 가 어떻게 이 trade-off 를 정정했는지 설명하라.

<details>
<summary>해설</summary>

**Kaplan 2020 의 추정**: 주어진 compute $C$ 에 대해 $N \propto C^{0.73}$, $D \propto C^{0.27}$ — model 크기에 더 많이 투자.

**문제**: Kaplan 의 권장대로 GPT-3 (175B params, 300B tokens) 는 $D / N \approx 1.7$ — 데이터 부족.

**Hoffmann 2022 (Chinchilla) 의 정정**: $D / N \approx 20$ tokens/parameter 가 compute-optimal. 즉:
- 같은 compute 에서 $N$ 줄이고 $D$ 늘려야 함
- Chinchilla (70B params, 1.4T tokens) > GPT-3 in many tasks despite 1/2.5 size

**왜 Kaplan 이 틀렸는가**: 그들의 fit 이 작은 model 에서 했는데, 큰 model 에서 데이터 부족이 더 심각. Loss curve 의 fit 을 다시 측정하면 다른 power.

**시사점**:
- "Scaling" 은 단순히 model 크기 키우는 것 ≠ data 도 비례 늘려야 함
- LLaMA, Mistral 등이 Chinchilla optimal 에 가까운 (smaller model, more data) — 실제로 GPT-3 보다 효율적
- 데이터 한계 ($L_\infty$) 에 도달하기 전 compute-optimal sweet spot

**미래 연구**: $L_\infty$ 자체를 낮추기 위한 더 좋은 데이터 (synthetic, curated), 또는 RL/RAG 같은 architecture 외 개선.

</details>

---

<div align="center">

[◀ 이전 (03. WaveNet)](./03-wavenet.md) | [📚 README](../README.md) | [다음 ▶ (Ch3-01. ELBO 유도)](../ch3-vae/01-elbo-derivation.md)

</div>
