# 01. Chain Rule Factorization

## 🎯 핵심 질문

- 확률의 chain rule $p(x_1, \ldots, x_n) = \prod_i p(x_i | x_{<i})$ 은 어떤 가정 하에서 성립하는가? (사실 가정 없이 성립하는 항등식)
- Autoregressive (AR) 모델이 이 항등식을 어떻게 architecture 로 변환하는가? 각 $p(x_i | x_{<i})$ 를 NN 으로 모델링한다는 것의 정확한 의미는?
- 순서 (ordering) 의 선택 — raster-scan vs zigzag vs random — 이 modeling 결과에 어떤 영향을 미치는가?
- AR 의 "tractable likelihood, sequential sampling" 이라는 trade-off 가 왜 architectural 으로 강제되는가?
- Teacher forcing 으로 훈련하는데 sampling 시 exposure bias 가 왜 생기고, 어떻게 완화하는가?

---

## 🔍 왜 Chain Rule 이 AR 의 기반인가

생성 모델의 가장 직관적이고 가장 오래된 접근 — **데이터를 순서대로 한 element 씩 생성하는 것**. 이 단순한 아이디어가 GPT, PixelCNN, WaveNet 의 기반입니다.

수학적 기반은 **확률의 chain rule**:

$$p(x_1, \ldots, x_n) = p(x_1) \cdot p(x_2 | x_1) \cdot p(x_3 | x_1, x_2) \cdots p(x_n | x_{<n})$$

이는 **항등식** 입니다 — 가정 없이 모든 분포에 대해 성립. 단지 conditional probability 의 정의에서 따라옴.

AR 모델의 핵심 통찰:

1. **항등식을 architecture 로**: 각 $p(x_i | x_{<i})$ 를 NN 으로 parameterize → 전체 분포 결정
2. **Tractable likelihood**: $\log p(x) = \sum_i \log p(x_i | x_{<i})$ — 각 항이 forward pass 1회로 평가
3. **Universal modeling**: chain rule 은 identity, 따라서 충분히 expressive 한 NN 이면 임의의 분포 표현 가능

이 단순한 통찰이 왜 GPT, ImageGPT, MusicLM 같은 기적적인 결과를 만드는지 — 그 수학적 기반을 다룹니다.

---

## 📐 수학적 선행 조건

- [Probability Theory Deep Dive](https://github.com/iq-ai-lab/probability-theory-deep-dive): 조건부 확률, Bayes' theorem, 확률의 chain rule
- [Information Theory Deep Dive](https://github.com/iq-ai-lab/information-theory-deep-dive): Entropy, conditional entropy, mutual information
- 이전 챕터: Ch1 전체 (특히 Ch1-02 explicit/implicit)

---

## 📖 직관적 이해

### "이야기를 이어 쓰기"

문장 "The cat sat on the ___" 의 다음 단어를 예측한다고 생각해봅시다. 인간은 자연스럽게 "mat" 또는 "chair" 같은 단어를 떠올립니다. AR 모델은 이를 정확히 흉내 — 이전 단어들을 보고 다음 단어의 확률 분포를 예측, 거기서 샘플.

이미지의 경우: 왼쪽 위 픽셀부터 시작해 raster-scan 순서로 한 픽셀씩 — 각 픽셀의 색을 이전 픽셀들 기반으로 예측. PixelCNN 의 핵심 아이디어.

오디오의 경우: 16kHz 오디오는 1초에 16000 samples — 각 샘플을 이전 샘플들로 예측. WaveNet 의 핵심.

**모든 경우에 공통**:
- Sequential **dependency** (이전을 보고 다음을 예측)
- Sequential **sampling** (한 번에 하나씩 생성)
- **Parallel training** (teacher forcing — 정답을 알고 동시에 모든 위치 예측)

### Chain Rule 의 수학적 보편성

확률 변수 $X_1, \ldots, X_n$ 에 대해 (어떤 분포라도):

$$p(x_1, x_2, x_3) = p(x_1, x_2) \cdot p(x_3 | x_1, x_2) = p(x_1) \cdot p(x_2 | x_1) \cdot p(x_3 | x_1, x_2)$$

귀납적으로:

$$p(x_1, \ldots, x_n) = \prod_{i=1}^n p(x_i | x_{<i})$$

이는 가정 (independence, Markov 등) 없이 성립하는 **항등식**. 따라서 AR 은 **분포에 대한 가정 없이 universal**.

### 순서 선택의 영향

같은 분포 $p(x_1, \ldots, x_n)$ 도 다른 순서 $\pi$ 에 대해:

$$p(x) = \prod_{i=1}^n p(x_{\pi(i)} | x_{\pi(<i)})$$

수학적으로는 동일 값이지만, 각 conditional 의 "어려움" 이 다릅니다. 예를 들어 이미지의 raster-scan 순서는 자연스럽지만, "오른쪽 아래부터" 시작하면 같은 모델 architecture 가 다른 inductive bias.

**XLNet** (Yang 2019) 은 모든 순서를 평균하여 unified — 양방향 정보 사용.

---

## ✏️ 엄밀한 정의·정리

### 정리 1.1 — 확률의 Chain Rule (항등식)

확률 변수 $X_1, \ldots, X_n$ 에 대해 (joint distribution 이 정의되면):

$$p(x_1, \ldots, x_n) = p(x_1) \prod_{i=2}^n p(x_i | x_1, \ldots, x_{i-1})$$

**증명**: $i = 2$: $p(x_1, x_2) = p(x_1) p(x_2 | x_1)$ — 조건부 확률의 정의.

귀납 가정 $p(x_1, \ldots, x_{n-1}) = \prod_{i=1}^{n-1} p(x_i | x_{<i})$. 그러면:

$$p(x_1, \ldots, x_n) = p(x_1, \ldots, x_{n-1}) \cdot p(x_n | x_1, \ldots, x_{n-1})$$

귀납 가정 대입하면:

$$= \left(\prod_{i=1}^{n-1} p(x_i | x_{<i})\right) p(x_n | x_{<n}) = \prod_{i=1}^n p(x_i | x_{<i}) \quad \square$$

### 정의 1.2 — Autoregressive Model

순서 $\pi$ 와 conditional 모델 family $\{p_\theta(x_i | x_{<i})\}_{i, \theta}$ 가 주어졌을 때:

$$p_\theta(x) = \prod_{i=1}^n p_\theta(x_{\pi(i)} | x_{\pi(<i)})$$

를 **autoregressive model** 이라 한다. 일반적으로 $\pi = $ identity (raster-scan, left-to-right).

### 정리 1.3 — AR 의 Universal Approximation

각 $p_\theta(x_i | x_{<i})$ 가 universal approximator (충분히 큰 NN) 이면, 전체 $p_\theta$ 는 임의의 joint distribution $p$ 를 KL divergence sense 로 임의로 잘 근사할 수 있다.

**증명 스케치**: $\text{KL}(p \| p_\theta) = \sum_i \mathbb{E}[\text{KL}(p(x_i | x_{<i}) \| p_\theta(x_i | x_{<i}))]$. 각 conditional 이 $\epsilon$-가까이 근사되면 합은 $n\epsilon$. UAT 로 각 conditional 을 임의로 근사 가능. $\square$

### 정의 1.4 — Teacher Forcing vs Autoregressive Sampling

**Training (teacher forcing)**: ground truth $x_{<i}$ 를 알고 $p_\theta(x_i | x_{<i})$ 의 likelihood 동시 계산 — **parallel** in $i$.

**Sampling (autoregressive)**: $\hat x_1 \sim p_\theta(x_1)$, then $\hat x_2 \sim p_\theta(x_2 | \hat x_1)$, ... — **sequential**, $O(n)$ time.

이 비대칭이 AR 의 trade-off: training 빠름, sampling 느림.

### 정리 1.5 — Exposure Bias

Teacher forcing 으로 훈련된 모델이 sampling 시 자기 출력을 입력으로 사용 → 누적 오차 (cascading error).

**수학적 분석**: training 시 $\mathbb{E}_{x_{<i} \sim p_d}[\log p_\theta(x_i | x_{<i})]$, sampling 시 $\mathbb{E}_{\hat x_{<i} \sim p_\theta}[p_\theta(x_i | \hat x_{<i})]$ — 다른 분포에서 평가. Train-test mismatch.

**결과**: 초반 작은 오차가 후반에 증폭. 예: language model 이 5번째 단어에서 약간 이상한 단어 선택 → 그 후로 점점 부자연스러워짐.

---

## 🔬 증명 및 수학적 유도

### 유도 1 — 순서 무관성 (Sum-rule)

서로 다른 순서 $\pi_1, \pi_2$ 에 대한 두 AR factorization:

$$\prod_{i} p(x_{\pi_1(i)} | x_{\pi_1(<i)}) = \prod_i p(x_{\pi_2(i)} | x_{\pi_2(<i)}) = p(x_1, \ldots, x_n)$$

**수학적으로 동일** (true distribution 일 때). 하지만 **NN 으로 근사할 때는 다름** — 각 conditional 의 학습 어려움이 순서에 따라 다름.

**예시**: 이미지에서 raster-scan vs zigzag — 동일한 픽셀들을 condition 으로 보지만, NN architecture (특히 masked conv) 가 raster-scan 에 최적화. Zigzag 는 다른 mask 가 필요.

### 유도 2 — AR 의 Sampling Cost

$x = (x_1, \ldots, x_n)$ 를 생성하려면:

- $x_1 \sim p_\theta(x_1)$: forward pass 1회
- $x_2 \sim p_\theta(x_2 | x_1)$: forward pass 1회 (또는 partial conditional)
- ...
- $x_n \sim p_\theta(x_n | x_{<n})$: forward pass 1회

총 **$n$ 번의 sequential forward pass**. 이미지 $32 \times 32 \times 3 = 3072$ 이면 3072 번. WaveNet 1초 오디오면 16000 번.

이것이 AR 의 fundamental bottleneck. 해결책:
- **Caching** (KV cache in Transformer): 이전 forward 결과 재사용으로 각 step 의 cost 감소
- **Parallel WaveNet** (van den Oord 2017): IAF distillation 으로 parallel sampling
- **Speculative decoding** (Leviathan 2023): 작은 모델의 prediction 을 큰 모델로 verify

### 유도 3 — Likelihood 의 Exact Computation

$$\log p_\theta(x) = \sum_{i=1}^n \log p_\theta(x_i | x_{<i})$$

**모든 $i$ 의 conditional 을 동시에 계산** 가능 (teacher forcing): $x_{<i}$ 가 모두 input 에 있으므로 한 번의 forward pass 로 모든 위치의 logits 산출. 따라서 likelihood 평가 = **forward pass 1회** = $O(\text{NN cost})$.

이것이 AR 이 explicit (tractable likelihood) 인 이유.

### 유도 4 — Markov Assumption 과의 관계

n-gram 모델: $p(x_i | x_{<i}) \approx p(x_i | x_{i-k}, \ldots, x_{i-1})$ (k-th order Markov).

AR 모델 (NN-based): truncation 없음 — 이론상 모든 이전을 봄. RNN/Transformer 가 long-range dependency 를 capture.

**Markov 가정의 한계**: bigram (k=1) 으로는 "the cat ___" 의 다음을 정확히 예측 불가능 (문맥 부족). NN AR 은 임의 길이 문맥 처리.

---

## 💻 실험 재현 / PyTorch 구현

### 실험 1 — 1D AR 모델 직접 구현

```python
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

class AR1D(nn.Module):
    """간단한 1D AR — 각 위치에서 mixture of Gaussians"""
    def __init__(self, seq_len=10, hidden=64, n_mix=4):
        super().__init__()
        self.seq_len = seq_len
        self.n_mix = n_mix
        # 각 위치 i 에 대한 NN: x_{<i} → (pi, mu, sigma) for mixture
        self.layers = nn.ModuleList([
            nn.Sequential(
                nn.Linear(max(i, 1), hidden), nn.ReLU(),
                nn.Linear(hidden, 3 * n_mix),  # logits, mu, log_sigma
            ) for i in range(seq_len)
        ])
        # i = 0 은 unconditional
        self.unconditional = nn.Parameter(torch.zeros(3 * n_mix))

    def conditional_params(self, x_prev, i):
        if i == 0:
            params = self.unconditional
        else:
            params = self.layers[i](x_prev)
        logits, mus, log_sigmas = params.chunk(3, dim=-1)
        return logits, mus, log_sigmas

    def log_p(self, x):
        """x: [batch, seq_len], returns log p(x) per sample"""
        log_p = torch.zeros(x.shape[0])
        for i in range(self.seq_len):
            logits, mus, log_sigmas = self.conditional_params(x[:, :i], i)
            sigmas = log_sigmas.exp()
            log_pi = torch.log_softmax(logits, -1)
            log_p_k = -0.5 * ((x[:, i:i+1] - mus) / sigmas)**2 - log_sigmas \
                     - 0.5 * np.log(2*np.pi)
            log_p += torch.logsumexp(log_p_k + log_pi, -1)
        return log_p

    @torch.no_grad()
    def sample(self, n=1):
        x = torch.zeros(n, 0)
        for i in range(self.seq_len):
            logits, mus, log_sigmas = self.conditional_params(x, i)
            sigmas = log_sigmas.exp()
            cat = torch.distributions.Categorical(logits=logits)
            k = cat.sample()                   # mixture component
            mu_k = mus.gather(-1, k.unsqueeze(-1))
            sigma_k = sigmas.gather(-1, k.unsqueeze(-1))
            x_i = mu_k + sigma_k * torch.randn_like(mu_k)
            x = torch.cat([x, x_i], dim=-1)
        return x

# Toy data: AR(1) Gaussian process — x_i = 0.8 x_{i-1} + ε
def gen_data(n=1024, seq_len=10):
    x = torch.zeros(n, seq_len)
    x[:, 0] = torch.randn(n)
    for i in range(1, seq_len):
        x[:, i] = 0.8 * x[:, i-1] + torch.randn(n) * 0.5
    return x

model = AR1D(seq_len=10)
data = gen_data(2048)
opt = torch.optim.Adam(model.parameters(), lr=1e-2)

for step in range(2000):
    idx = torch.randperm(len(data))[:128]
    loss = -model.log_p(data[idx]).mean()
    opt.zero_grad(); loss.backward(); opt.step()
    if step % 200 == 0:
        print(f"Step {step}: NLL = {loss.item():.3f}")

samples = model.sample(500).numpy()
print("Sample mean:", samples.mean(axis=0).round(2))
print("Data mean:  ", data.numpy().mean(axis=0).round(2))
```

### 실험 2 — Order Sensitivity 측정

```python
# 동일 분포를 두 다른 순서로 학습 → NN 학습 결과 비교
# (수학적으로 동일하지만 NN 근사 시 다를 수 있음)

class AR1DReversed(AR1D):
    """오른쪽에서 왼쪽으로 학습"""
    def log_p(self, x):
        # x reversed
        x_rev = torch.flip(x, [-1])
        return super().log_p(x_rev)

model_fwd = AR1D(seq_len=10)
model_rev = AR1DReversed(seq_len=10)

# 두 모델을 같은 데이터에 동시 훈련
# AR(1) 데이터의 reverse 도 AR(1) (linear Gaussian property), 따라서 이론상 동등
# 하지만 NN 근사에서는 약간의 차이 — 측정해 보면 흥미로움
```

### 실험 3 — Teacher Forcing vs Autoregressive Sampling 의 Exposure Bias 시연

```python
# 잘 훈련된 model 로:
# (a) Teacher forcing 시 NLL: 데이터의 평균 NLL
# (b) Autoregressive sampling 후 NLL: 자기 sample 의 평균 NLL
# (b) > (a) — exposure bias

with torch.no_grad():
    nll_data = -model.log_p(data).mean().item()
    samples = model.sample(2048)
    nll_self = -model.log_p(samples).mean().item()
print(f"NLL on data:    {nll_data:.3f}")
print(f"NLL on samples: {nll_self:.3f}")
# 이상적으로 두 값이 일치, 실전에서는 NLL on samples 가 종종 작음
# (모델이 자기 분포를 잘 fit 한다는 의미, 진짜와 다르더라도)
```

---

## 🔗 이론과 실전의 간극

### 1. 순서가 정말로 중요하지 않은가

이론적으로는 동등하지만 실전에서는:
- **이미지**: raster-scan 이 자연스러움 (왼쪽 위 → 오른쪽 아래의 인과적 직관)
- **언어**: left-to-right 가 인간 텍스트 생성과 일치
- **분자 구조**: 트리 구조 (graph AR) 또는 SMILES sequence — 다양한 순서

XLNet (Yang 2019), Permutation Language Modeling: 다양한 순서를 평균 → 양방향 정보. 실전에서 BERT (마스킹) vs GPT (단방향) 의 trade-off 이슈.

### 2. Exposure Bias 의 실제 영향

이론적으로 큰 문제이지만 실전에서 GPT 가 잘 작동하는 이유:
- **Big data**: 충분한 데이터로 분포가 잘 학습되면 sampling 분포 ≈ training 분포
- **Temperature sampling**: $p^{1/T}$ 로 sharp 또는 smooth 조절
- **Top-k, top-p sampling**: 낮은 확률 단어 제외로 catastrophic mistake 방지

Scheduled sampling (Bengio 2015), Professor Forcing (Lamb 2016) 등의 해결책도 있지만, 현재 LLM 은 simply scale + nucleus sampling 으로 충분.

### 3. "Sequential" 의 진짜 의미

GPT 는 KV cache 로 매 step 의 cost 를 줄이지만 여전히 sequential. 이를 회피하는 시도:
- **Diffusion + AR hybrid**: GLIDE, Diffusion-LM
- **Speculative decoding**: 작은 모델로 multi-step 동시 예측 → 큰 모델로 verify
- **Parallel decoding**: Mask + iterative refine (MaskGIT, MAGE)

본질적으로 AR 의 sequential 은 chain rule 의 dependency 에서 옴 — 순수 AR 안에서는 회피 불가.

---

## ⚖️ 가정과 한계

| 가정 | 한계 |
|------|------|
| Chain rule 은 항등식 | 가정 없음, 모든 분포에 성립 |
| 순서 선택은 동등 | NN 근사에서는 차이, 도메인별 자연 순서 |
| Teacher forcing 으로 training | Exposure bias 문제 — sampling 분포와 mismatch |
| Sequential sampling 은 unavoidable | KV cache, speculative decoding 으로 완화 가능 |
| Tractable likelihood | Sample quality 가 NLL 과 항상 비례하지 않음 |

---

## 📌 핵심 정리

$$\boxed{p(x_1, \ldots, x_n) = \prod_{i=1}^n p(x_i | x_{<i}) \text{ — 확률의 chain rule (항등식)}}$$

$$\boxed{\log p_\theta(x) = \sum_i \log p_\theta(x_i | x_{<i}) \text{ — Tractable likelihood, parallel training}}$$

| 특성 | 값 |
|------|------|
| **Likelihood** | Exact, $O(1)$ forward pass |
| **Training** | Parallel (teacher forcing) |
| **Sampling** | Sequential, $O(n)$ |
| **Universal approximation** | Yes (충분한 NN) |
| **순서 의존성** | 수학적 동등, NN 근사 시 차이 |
| **Exposure bias** | Train-sample 분포 mismatch |

---

## 🤔 생각해볼 문제

**문제 1** (기초): 이산 확률 변수 $X, Y, Z$ 에 대해 $p(x, y, z) = p(x) p(y|x) p(z|x, y)$ 와 $p(x, y, z) = p(z) p(y|z) p(x|y, z)$ 둘 다 성립함을 보이라. 이게 chain rule 의 순서 무관성과 어떤 관련이 있는가?

<details>
<summary>해설</summary>

첫 번째: 정의로 $p(x, y, z) = p(x) \cdot \frac{p(x,y)}{p(x)} \cdot \frac{p(x,y,z)}{p(x,y)} = p(x) p(y|x) p(z|x,y)$. ✓

두 번째: 같은 방식, 순서만 바꿔서. $p(x, y, z) = p(z) \cdot \frac{p(y,z)}{p(z)} \cdot \frac{p(x,y,z)}{p(y,z)} = p(z) p(y|z) p(x|y,z)$. ✓

**시사점**: 같은 joint distribution 의 다른 factorization 모두 valid. 어떤 것을 선택할지는 (i) 어떤 conditional 이 모델링하기 쉬운가, (ii) 어떤 sampling 순서가 자연스러운가에 따라.

</details>

**문제 2** (심화): AR 모델의 NLL 이 모든 $i$ 에 대해 동시에 계산 가능한 이유 (teacher forcing) 를 설명하라. 왜 sampling 은 그렇게 안 되는가?

<details>
<summary>해설</summary>

**Training**: 데이터 $x = (x_1, \ldots, x_n)$ 가 모두 알려져 있음. 각 위치 $i$ 의 conditional $p_\theta(x_i | x_{<i})$ 평가 시 $x_{<i}$ 가 input 에 있음. Causal masking (Transformer) 또는 masked convolution (PixelCNN) 으로 한 번의 forward pass 에서 모든 $i$ 의 conditional 을 동시 산출.

**Sampling**: $x$ 가 아직 없음. $x_1$ 을 sampling 한 후에야 $p_\theta(x_2 | x_1)$ 평가 가능. 그 뒤에야 $x_2$ sampling. 각 $x_i$ 가 직전 단계 결과에 의존하므로 **데이터 의존성** 이 sequential.

**비유**: Crossword puzzle 의 정답을 보면서 답을 채우는 것 (training, parallel) vs 한 칸씩 풀어가는 것 (sampling, sequential).

**해결책**: speculative decoding, parallel decoding, distillation (Parallel WaveNet) 등이 sampling 을 가속하지만, fundamental sequential 성을 완전히 깨지는 못함.

</details>

**문제 3** (논문 비평): GPT 같은 큰 AR 모델이 exposure bias 문제에도 잘 작동하는 이유를 (i) 데이터 양, (ii) sampling 전략, (iii) capacity 측면에서 논의하라. 작은 데이터/모델에서는 어떻게 다를 것 같은가?

<details>
<summary>해설</summary>

**(i) 데이터 양**: GPT-3 의 학습 데이터 $\sim 500$B tokens. 데이터 분포가 매우 정확히 학습되어 sampling 분포 $p_\theta$ 가 진짜 $p_d$ 와 매우 가까움 → train-test mismatch 작음.

**(ii) Sampling 전략**: top-k, top-p (nucleus) sampling, temperature scaling 으로 catastrophic 한 low-probability 단어 제외. Beam search 도 일정 정도 도움.

**(iii) Capacity**: 큰 모델은 long-range dependency 를 더 잘 capture, 따라서 자기 출력에 의존해도 일관성 유지.

**작은 데이터/모델에서는**:
- 데이터 부족 → $p_\theta$ 가 진짜 분포에서 멀음 → exposure bias 심각
- Capacity 부족 → 자기 mistake 를 long-term 으로 cascade
- Scheduled sampling, professor forcing, reinforcement learning (RLHF) 등의 보조 기법이 더 중요해짐

**시사점**: Scaling 이 많은 문제를 해결하지만, fundamental 한 issue 는 여전히 존재. 이론적 이해는 작은 시나리오에서 더 중요.

</details>

---

<div align="center">

[◀ 이전 (Ch1-04. Evaluation)](../ch1-taxonomy/04-evaluation-metrics.md) | [📚 README](../README.md) | [다음 ▶ (02. PixelRNN/PixelCNN)](./02-pixelrnn-pixelcnn.md)

</div>
