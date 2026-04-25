# 03. WaveNet (van den Oord 2016)

## 🎯 핵심 질문

- 16kHz 오디오 1초 = 16000 samples 의 long-range dependency 를 어떻게 효율적으로 capture 하는가?
- Causal convolution 의 정확한 정의는? Standard conv 와 어떻게 다른가?
- Dilated convolution 의 receptive field 가 왜 $RF = k + (k-1)(d-1)$ 인가? Exponential dilation $d_l = 2^{l-1}$ 로 어떻게 $O(2^L)$ RF 를 달성하는가?
- WaveNet 의 generation 이 sequential 로 매우 느린 이유와, Parallel WaveNet 의 IAF distillation 이 어떻게 이를 해결하는가?
- WaveNet 의 gated activation $\tanh(W_f x) \odot \sigma(W_g x)$ 이 ReLU 보다 좋은 이유는?

---

## 🔍 왜 WaveNet 이 결정적인가

van den Oord 2016 "WaveNet: A Generative Model for Raw Audio" 는 **raw waveform 직접 모델링** 으로 음성 합성 (TTS) 의 패러다임을 바꿨습니다. 핵심 기여:

1. **Dilated Causal Convolution** — long-range dependency 와 efficient computation 을 결합
2. **Raw audio AR** — 16kHz 의 원시 waveform 을 categorical 로 모델링 (μ-law)
3. **Gated activations** — LSTM-inspired 활성화로 표현력 향상
4. **Conditional generation** — 화자 · text · pitch 조건부 생성

이 작업은 후속 Music generation (MusicLM, AudioLM), Sora 의 video DiT, 그리고 long-context language model 까지 영향. 핵심 통찰은 **dilated conv 로 RF 를 지수적으로 확장** — 이 아이디어 자체가 architecture 설계의 일반적 도구가 됨.

---

## 📐 수학적 선행 조건

- [CNN Deep Dive](https://github.com/iq-ai-lab/cnn-deep-dive): Conv, dilation, receptive field
- 이전 문서: 02-pixelrnn-pixelcnn.md
- [Probability Theory Deep Dive](https://github.com/iq-ai-lab/probability-theory-deep-dive): Categorical distribution

---

## 📖 직관적 이해

### "오디오 = 매우 긴 1D sequence"

이미지 $32 \times 32 \times 3 = 3072$ 차원이지만, 1초 16kHz 오디오는 **16000 차원**. 음악 파일 1분은 거의 백만 차원. 매우 긴 sequence 의 long-range dependency (예: 음악의 비트, 화자의 톤) 를 모델링해야 합니다.

**문제**: 표준 stacked conv 는 RF 가 $L \cdot (k-1) + 1$ — 30 개 layer with $k=3$ 이면 RF = 61 samples = 4ms. 음악적 의미 없음.

**해결**: Dilated convolution — 같은 layer 수로 RF 지수적 확장. $d_l = 2^{l-1}$ 일 때 30 layer RF $\approx 2^{30}$ samples.

### Causal Convolution

표준 conv 는 양쪽을 봄. AR 모델은 미래를 보면 안 됨 → causal padding:

```
표준 conv (k=3):    padding 양쪽 1
[? x x x x x ?]
   ↓ ↓ ↓ ↓ ↓
   y_1 ... y_5

Causal conv (k=3): padding 왼쪽 2
[? ? x x x x x]
       ↓ ↓ ↓ ↓ ↓
       y_1 ... y_5    (y_i 가 x_{<=i} 만 봄)
```

수학적으로: $y_i = \sum_{k=0}^{K-1} W_k \cdot x_{i-k}$ — $i - k \leq i$, **항상 과거 또는 현재**.

### Dilated Convolution

$d$ 만큼 떨어진 input 을 봄:

$$y_i = \sum_{k=0}^{K-1} W_k \cdot x_{i - d \cdot k}$$

$d = 1$: 표준. $d = 2$: 한 칸씩 건너뜀. $d = 4$: 두 칸씩.

**RF**: 단일 layer $RF = k + (k-1)(d-1) = 1 + (k-1)d$. Stack:

$$RF_L = 1 + \sum_{l=1}^L (k - 1) d_l$$

Exponential dilation $d_l = 2^{l-1}$, $k = 2$:

$$RF_L = 1 + \sum_{l=1}^L 2^{l-1} = 2^L$$

**$L = 10$ 이면 $RF = 1024$ samples = 64ms. $L = 30$ 이면 $RF \approx 10^9$** — 충분히 long range.

---

## ✏️ 엄밀한 정의·정리

### 정의 3.1 — Causal Convolution

길이 $T$ sequence $x \in \mathbb{R}^T$ 와 kernel $W \in \mathbb{R}^k$ 에 대해:

$$y_i = \sum_{j=0}^{k-1} W_j \cdot x_{i - j}, \quad i = 0, \ldots, T-1$$

$x_{i - j}$ 가 $j > i$ 일 때는 zero padding (left padding).

**효과**: $y_i$ 는 $x_0, x_1, \ldots, x_i$ 만 봄 — causal.

### 정의 3.2 — Dilated Convolution

Dilation rate $d$ 와 kernel $W \in \mathbb{R}^k$:

$$y_i = \sum_{j=0}^{k-1} W_j \cdot x_{i - d \cdot j}$$

$d = 1$: 표준 (causal) conv. $d > 1$: input 을 건너뛰며 sample.

### 정의 3.3 — Stacked Dilated Causal Conv

WaveNet 의 architecture: $L$ layer, kernel $k = 2$, dilation $d_l = 2^{l-1}$ for $l = 1, \ldots, L$.

각 layer:
$$h^{(\ell)}_i = \tanh(W_f^{(\ell)} \cdot h^{(\ell-1)}_{i-d_\ell:i}) \odot \sigma(W_g^{(\ell)} \cdot h^{(\ell-1)}_{i-d_\ell:i})$$

**Gated activation** — Tanh 와 sigmoid 의 element-wise 곱.

전체 RF:

$$RF_L = 1 + \sum_{l=1}^L (k-1) d_l = 1 + \sum_{l=1}^L 2^{l-1} = 2^L$$

### 정리 3.4 — Dilated Conv 의 RF 공식

$L$ 개 dilated conv layer (kernel $k$, dilation rates $d_1, \ldots, d_L$) 의 receptive field:

$$RF_L = 1 + \sum_{l=1}^L (k - 1) d_l$$

**증명**: Layer $l$ 추가 시 RF 가 $(k-1) d_l$ 만큼 확장. Initial RF = 1. 귀납. $\square$

### 정리 3.5 — 24-bit Audio 의 16-bit μ-Law Compression

원시 16-bit (또는 24-bit) audio 를 256-way categorical 로 만들기 위해 μ-law:

$$f(x) = \text{sign}(x) \frac{\ln(1 + \mu |x|)}{\ln(1 + \mu)}, \quad \mu = 255$$

이 비선형 함수가 small amplitude 를 더 정밀하게 quantize → perceptual quality 향상.

WaveNet 은 256-way softmax 로 modeling: $p(x_t | x_{<t}) = \text{softmax}(\text{logits})_{c}$, $c$ = quantized class.

### 정리 3.6 — Generation 비용

WaveNet sampling 은 sequential — sample 당 1 forward pass:

$$\text{비용} = T \cdot \text{(forward pass cost)}$$

1초 16kHz audio = 16000 forward pass. NN inference 가 1ms 라도 → 16초 generation. Real-time impossible without optimization.

**Parallel WaveNet** (van den Oord 2017): IAF (Inverse Autoregressive Flow) student 가 teacher WaveNet 을 distillation → parallel sampling 가능.

---

## 🔬 증명 및 수학적 유도

### 유도 1 — Exponential RF 의 정확한 계산

$d_l = 2^{l-1}$, $k = 2$, $L$ layer:

$$RF_L = 1 + \sum_{l=1}^L (2 - 1) \cdot 2^{l-1} = 1 + (1 + 2 + 4 + \cdots + 2^{L-1}) = 1 + (2^L - 1) = 2^L$$

**$L = 10$**: $RF = 1024$
**$L = 20$**: $RF = 1.05M$
**$L = 30$**: $RF = 1.07B$ (16kHz 에서 67000 초 = 18.6시간)

WaveNet 원 논문의 30-layer 구조는 long enough RF.

### 유도 2 — Cycle 패턴: Multi-Stack with Reset

Pure exponential dilation 은 RF 가 너무 빠르게 커짐. 실전 WaveNet 은 **cycle pattern**: $d = 1, 2, 4, ..., 512, 1, 2, 4, ..., 512$ (10 layer cycle, 3-4 cycles).

이유:
- 다양한 scale 에서 정보 (short + long) 모두 capture
- Local pattern (음소) 와 global pattern (악구) 동시 모델링

각 cycle 의 RF 는 $2^{10} = 1024$, 3 cycles total RF $\approx 3 \times 1024 = 3072$. 충분.

### 유도 3 — Gated Activation 의 효과

WaveNet 은 ReLU 대신 $\tanh \odot \sigma$ 사용:

$$z = \tanh(W_f \cdot x) \odot \sigma(W_g \cdot x)$$

**LSTM 의 input gate** 와 유사 — sigmoid 가 "얼마나 통과할지" 결정, tanh 가 "값" 결정.

**Empirical 효과**: 더 부드러운 gradient flow, NLL 약 5-10% 개선 (논문 보고). 단, ReLU/GELU 도 잘 작동, gated 의 우월성은 architectural detail 에 의존.

### 유도 4 — Residual + Skip Connection

WaveNet 의 각 layer:

$$h^{(\ell)} = h^{(\ell-1)} + \text{GatedActivation}(h^{(\ell-1)})$$

추가로 **skip connection** 으로 모든 layer 출력을 final 로 모음 → 다양한 scale 의 feature 활용.

이 구조가 ResNet 의 identity mapping 과 동일한 gradient flow 이점.

### 유도 5 — Parallel WaveNet 의 IAF Distillation

Teacher: AR WaveNet ($p_T(x | y)$ — sequential).
Student: IAF (Kingma 2016) — parallel sampling, sequential density.

훈련: KL divergence:

$$\text{KL}(p_S \| p_T) = \mathbb{E}_{x \sim p_S}[\log p_S(x) - \log p_T(x)]$$

$p_S$ 에서 sampling 은 IAF 로 parallel, $p_S(x)$ 와 $p_T(x)$ 모두 평가는 forward pass. 따라서 각 step 의 비용이 NN forward + reverse pass.

훈련 후 student 만 사용 → parallel sampling. 음질 약간 손실, 속도는 1000배 향상 (실시간 TTS 가능).

---

## 💻 실험 재현 / PyTorch 구현

### 실험 1 — Dilated Causal Conv 구현 + RF 측정

```python
import torch
import torch.nn as nn

class CausalConv1d(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size=2, dilation=1):
        super().__init__()
        self.padding = (kernel_size - 1) * dilation
        self.conv = nn.Conv1d(in_ch, out_ch, kernel_size,
                               padding=self.padding, dilation=dilation)
    def forward(self, x):
        out = self.conv(x)
        # right-side trimming (causal)
        return out[..., :-self.padding] if self.padding > 0 else out

# RF 측정 — input 의 어느 위치를 perturb 했을 때 output 변하는지
def measure_rf(model, T=2048):
    x = torch.zeros(1, 1, T)
    x[0, 0, T // 2] = 1.0   # perturb center
    y = model(x)
    perturbed = (y.abs().sum(1) > 1e-6).squeeze().nonzero().flatten()
    if len(perturbed) == 0: return 0
    return perturbed[-1].item() - T // 2

# Stack 10 layer, exponential dilation
class WaveNetBlock(nn.Module):
    def __init__(self, channels=64, n_layers=10, kernel=2):
        super().__init__()
        self.layers = nn.ModuleList([
            CausalConv1d(channels, channels, kernel, dilation=2**l)
            for l in range(n_layers)
        ])
    def forward(self, x):
        for layer in self.layers:
            x = x + torch.tanh(layer(x))   # 간단화 (gated 생략)
        return x

model = WaveNetBlock(n_layers=10).eval()
rf = measure_rf(model)
print(f"Measured RF: {rf}")
print(f"Theoretical RF: {2**10 - 1}")  # 2^L - 1 (excluding center)
# 예상: 두 값이 일치
```

### 실험 2 — Gated Activation vs ReLU 비교

```python
class GatedConv(nn.Module):
    def __init__(self, in_ch, out_ch, kernel, dilation):
        super().__init__()
        self.conv = CausalConv1d(in_ch, 2 * out_ch, kernel, dilation)
    def forward(self, x):
        h = self.conv(x)
        f, g = h.chunk(2, dim=1)
        return torch.tanh(f) * torch.sigmoid(g)

# Toy 1D AR (sin wave 학습)
T = 256
freq = 5
x = torch.sin(2 * torch.pi * freq * torch.linspace(0, 1, T)).unsqueeze(0).unsqueeze(0)

# Gated vs ReLU 두 모델 비교
class WNModel(nn.Module):
    def __init__(self, gated=True, channels=32, n_layers=8):
        super().__init__()
        self.first = nn.Conv1d(1, channels, 1)
        if gated:
            self.layers = nn.ModuleList([
                GatedConv(channels, channels, 2, 2**l) for l in range(n_layers)
            ])
        else:
            self.layers = nn.ModuleList([
                nn.Sequential(
                    CausalConv1d(channels, channels, 2, 2**l), nn.ReLU()
                ) for l in range(n_layers)
            ])
        self.out = nn.Conv1d(channels, 1, 1)
    def forward(self, x):
        h = self.first(x)
        for layer in self.layers:
            h = h + layer(h)
        return self.out(h)

# 두 모델 동일하게 훈련 → final loss 비교
# 일반적으로 Gated 가 약 5% NLL 우월
```

### 실험 3 — Sampling 속도 측정

```python
import time

@torch.no_grad()
def autoregressive_sample(model, T=1000):
    x = torch.zeros(1, 1, 1)
    for t in range(T):
        h = model(x)
        next_x = h[:, :, -1:].tanh()  # 또는 categorical sampling
        x = torch.cat([x, next_x], dim=-1)
    return x

start = time.time()
sample = autoregressive_sample(model, T=2000)
elapsed = time.time() - start
print(f"Generated 2000 samples in {elapsed:.2f}s ({2000/elapsed:.1f} samples/sec)")

# 16kHz audio 의 1초 = 16000 samples
# 이 속도로는 1초 audio 가 20-30초 걸림 → real-time 불가
# Parallel WaveNet 또는 streaming 으로 해결
```

---

## 🔗 이론과 실전의 간극

### 1. Cycled Dilation 의 실전적 동기

순수 exponential ($d_l = 2^{l-1}$ for all layers) 보다 **multi-stack** ($d = 1, 2, ..., 512, 1, 2, ..., 512$) 이 더 잘 작동:
- Local + global 정보 모두 capture
- Gradient flow 가 short-distance dependency 에서도 통과

DeepMind 의 production WaveNet 도 이 구조 사용.

### 2. Real-Time Synthesis 의 도전

Naive WaveNet: 16kHz 1초 = 16000 forward pass. RTX 3090 도 forward pass < 1ms 라도 → 16초. 실시간 TTS 불가능. 해결책:

- **Parallel WaveNet** (van den Oord 2017): IAF distillation
- **WaveRNN** (Kalchbrenner 2018): RNN-based, smaller
- **WaveGlow** (Prenger 2019): Flow-based, parallel
- **HiFi-GAN** (Kong 2020): GAN-based vocoder, real-time
- **현대 TTS** (Tacotron 2 + HiFi-GAN, VITS): mostly GAN/Flow vocoder + AR text encoder

### 3. WaveNet 의 후속 영향

WaveNet 의 dilated causal conv 아이디어가:
- **TCN** (Bai 2018): Temporal Convolutional Network, RNN 대체
- **Transformer**: long-range dependency 의 다른 해결 (attention)
- **Long-context LLM**: Sliding window, ALiBi, Linear attention 등은 모두 long RF 의 다른 접근

WaveNet 자체는 raw audio 모델로는 대체되었지만 (mostly GAN/Flow), 그 architectural insight 는 살아있음.

---

## ⚖️ 가정과 한계

| 가정 | 한계 |
|------|------|
| Sequential audio | Polyphonic music 의 multi-channel 처리 어려움 |
| 8-bit μ-law | 16-bit 손실, 고음질에서 quality 한계 |
| Causal | Real-time 만 가능, look-ahead 활용 못 함 |
| Sequential sampling | 16k samples/sec → naive 로 real-time 불가 |
| Bounded RF | $2^L$ 이지만 실제로는 cycled 로 제한 |
| Local conv | Global semantic (악구, 전체 구조) 파악 한계 |

---

## 📌 핵심 정리

$$\boxed{y_i = \sum_{k=0}^{K-1} W_k \cdot x_{i - d \cdot k} \text{ — Dilated causal conv}}$$

$$\boxed{RF_L = 1 + \sum_{l=1}^L (k-1) d_l, \quad d_l = 2^{l-1} \Rightarrow RF = 2^L}$$

| 구성요소 | 역할 |
|---------|------|
| **Causal conv** | Future 차단, AR factorization 보장 |
| **Dilated conv** | RF 지수적 확장, $O(2^L)$ |
| **Cycle pattern** | Multi-scale (short + long range) |
| **Gated activation** | $\tanh \odot \sigma$, LSTM-style |
| **Residual + skip** | Gradient flow + multi-scale aggregation |
| **256-way softmax** | μ-law quantized output |
| **Sampling cost** | $O(T)$ sequential — Parallel WaveNet 으로 해결 |

---

## 🤔 생각해볼 문제

**문제 1** (기초): Kernel size $k = 3$, dilation $d = 4$ 인 single dilated conv layer 의 RF 를 계산하라. 이를 stack 5 layer with $d_l = 4^{l-1}$ 했을 때 RF 는?

<details>
<summary>해설</summary>

**Single layer**: $RF = 1 + (k - 1) d = 1 + 2 \cdot 4 = 9$.

**Stack with $d_l = 4^{l-1}$, $L = 5$**:
$$RF = 1 + \sum_{l=1}^5 (k-1) d_l = 1 + 2(1 + 4 + 16 + 64 + 256) = 1 + 2 \cdot 341 = 683$$

따라서 $RF = 683$. 비교: $k=2, d_l = 2^{l-1}$ 5 layer 이면 $RF = 32$. **$k$ 가 클수록 RF 빨리 확장**.

</details>

**문제 2** (심화): Cycled dilation $d = 1, 2, 4, \ldots, 512, 1, 2, 4, \ldots, 512$ (2 cycles, 20 layer) 의 RF 와 pure exponential $d = 1, 2, \ldots, 2^{19}$ (20 layer) 의 RF 를 비교하라. 왜 cycled 이 실전적으로 우월한가?

<details>
<summary>해설</summary>

**Cycled** ($k = 2$):
$$RF = 1 + 2 \sum_{l=0}^{9} 2^l = 1 + 2 \cdot 1023 = 2047$$

**Pure exponential**:
$$RF = 1 + \sum_{l=0}^{19} 2^l = 1 + (2^{20} - 1) = 2^{20} = 1048576$$

차이가 **500배**. Pure 가 RF 가 훨씬 큼.

**Cycled 의 우월성**:
1. **Multi-scale**: 각 cycle 이 다양한 dilation 을 거치며 short + long range 모두 capture
2. **Gradient flow**: 작은 dilation 이 자주 등장 → short-distance dependency 의 gradient 가 잘 통과
3. **Computational cost**: 큰 dilation 의 효용 marginal — $RF = 2^{20}$ 은 audio 의 길이 (~16k) 를 훨씬 초과, 낭비

따라서 cycled 가 **practical RF + multi-scale + efficient gradient** 의 균형. 음질 평가에서 cycled 가 더 좋음.

</details>

**문제 3** (논문 비평): WaveNet 의 sequential sampling 이 real-time 합성을 막는데, Parallel WaveNet 은 IAF distillation 으로 이를 해결한다. (i) Distillation 의 KL loss 형태와 (ii) 음질 손실의 원인을 설명하라.

<details>
<summary>해설</summary>

**(i) Distillation 의 KL loss**:

Teacher $p_T(x | c)$ (AR WaveNet, sequential), Student $p_S(x | c)$ (IAF, parallel).

$$\text{KL}(p_S \| p_T) = \mathbb{E}_{x \sim p_S}[\log p_S(x) - \log p_T(x)]$$

- $x \sim p_S$: IAF 로 parallel sampling — fast
- $\log p_S(x)$: IAF density (sequential, 학습 시는 OK)
- $\log p_T(x)$: AR WaveNet 으로 평가 (parallel, teacher forcing)

**(ii) 음질 손실 원인**:
1. **Mode-seeking reverse KL**: $\text{KL}(p_S \| p_T)$ 는 reverse KL — student 가 teacher 의 일부 mode 만 cover. 일부 음질 detail 손실.
2. **IAF 의 architecture 제약**: IAF 는 invertible + tractable det 의 제약 → expressiveness 한계.
3. **Auxiliary losses**: 원 논문은 power loss, contrastive loss 등 추가하여 quality 향상.

결과적으로 음질이 teacher 와 거의 비슷하지만 약간의 loss, 속도는 1000배+ 향상. 실시간 TTS 가 가능해진 핵심.

**현대적 비교**: Parallel WaveNet 은 distillation 의 복잡함 때문에 production 에서 **HiFi-GAN** (GAN-based vocoder) 으로 대체되는 추세. GAN 이 더 sharp + simpler training.

</details>

---

<div align="center">

[◀ 이전 (02. PixelRNN/PixelCNN)](./02-pixelrnn-pixelcnn.md) | [📚 README](../README.md) | [다음 ▶ (04. GPT as Generative)](./04-gpt-as-generative.md)

</div>
