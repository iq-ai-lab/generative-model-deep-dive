# 02. PixelRNN · PixelCNN (van den Oord 2016)

## 🎯 핵심 질문

- 이미지를 sequence 로 만드는 raster-scan 의 수학적 정의는? RGB 채널을 어떻게 처리하는가?
- Masked convolution 의 mask A 와 mask B 의 차이는 무엇이고, 왜 두 가지가 모두 필요한가?
- PixelRNN (row LSTM, diagonal BiLSTM) vs PixelCNN 의 architectural trade-off 는?
- Conditional PixelCNN 이 어떻게 class-conditional 또는 text-conditional 생성을 지원하는가?
- Why are PixelCNN samples often blurry compared to GAN/Diffusion despite tractable likelihood?

---

## 🔍 왜 PixelCNN 이 결정적인 일보였는가

van den Oord 2016 "Pixel Recurrent Neural Networks" 는 이미지에 처음으로 AR 을 본격 적용하여 **MNIST, CIFAR-10 의 SOTA NLL** 을 달성. 이 논문이 결정적인 이유:

1. **Masked Convolution** — 이미지 AR 의 architectural innovation, 후속 모든 image AR 의 기반
2. **Tractable likelihood for images** — bits per dimension (bpd) metric 의 표준 정착
3. **Conditional generation** — class-label 기반 image 생성의 첫 강력한 결과
4. **이론과 실전의 다리** — Chain rule 항등식이 어떻게 architecture 로 구현되는지의 모범

ImageGPT (Chen 2020) 같은 후속 image-as-sequence 작업, 그리고 VQ-VAE + AR 의 DALL-E 까지 모두 이 접근의 연장선. 이 문서에서는 **masked convolution 의 정확한 수학** 과 **mask A vs mask B 의 미묘한 차이** 를 중심으로 다룹니다.

---

## 📐 수학적 선행 조건

- [CNN Deep Dive](https://github.com/iq-ai-lab/cnn-deep-dive): Convolution, padding, receptive field
- 이전 문서: 01-chain-rule-factorization.md
- [Probability Theory Deep Dive](https://github.com/iq-ai-lab/probability-theory-deep-dive): Discrete distribution, mixture

---

## 📖 직관적 이해

### "왼쪽 위에서 오른쪽 아래로 그리기"

화가가 그림을 그릴 때 보통 한 영역씩 채우지만, AR 모델은 **픽셀 단위로 raster-scan 순서** (왼쪽 위 → 오른쪽 → 다음 줄 왼쪽 → ...) 로 그림. 각 픽셀의 색은 이미 그린 픽셀들에만 의존.

수학적으로:

$$p(I) = \prod_{i=1}^H \prod_{j=1}^W p(I_{ij} | I_{<ij})$$

여기서 $I_{<ij}$ = "현재 픽셀 (i, j) 보다 raster-scan 에서 앞" = $\{(i', j') : i' < i \text{ or } (i' = i \text{ and } j' < j)\}$.

### Masked Convolution 의 핵심

표준 CNN convolution 은 미래 픽셀도 봄 — AR 모델에서는 부적절. 따라서 **kernel 의 일부를 0 으로 mask** 하여 미래 차단:

```
Mask A (자기 자신도 차단):       Mask B (자기 자신 참조 허용):
1 1 1                            1 1 1
1 0 0   ← center                1 1 0   ← center
0 0 0                            0 0 0
```

**Mask A**: 첫 layer 에서 사용 — 입력의 현재 픽셀을 보면 안 됨 (그것을 예측해야 하므로).

**Mask B**: 두 번째 이후 layers 에서 사용 — 이전 layer 의 "현재 위치" feature 를 봐도 됨 (그 feature 는 이미 mask A 로 만들어진 것이므로 진짜 현재 픽셀 정보 없음).

### RGB 채널 처리

각 픽셀이 R, G, B 3 채널일 때:

$$p(I_{ij}) = p(R_{ij} | I_{<ij}) \cdot p(G_{ij} | I_{<ij}, R_{ij}) \cdot p(B_{ij} | I_{<ij}, R_{ij}, G_{ij})$$

따라서 같은 픽셀 안에서도 R → G → B 순서로 dependency. 추가로 channel-wise mask 가 필요.

### PixelRNN vs PixelCNN

**PixelRNN** (Row LSTM, Diagonal BiLSTM): 각 row 또는 diagonal 을 LSTM 으로 처리. 이론적으로 unbounded receptive field, 하지만 **sequential** computation → 느림.

**PixelCNN**: Masked convolution 의 stack. **Parallel** training, bounded RF (kernel 누적). Receptive field 키우려면 layer 많이 쌓아야 함.

후속 작업 (Gated PixelCNN, PixelCNN++): vertical + horizontal stack 으로 blind spot 제거.

---

## ✏️ 엄밀한 정의·정리

### 정의 2.1 — Raster-Scan Order

$H \times W$ 이미지에서 픽셀 $(i, j)$ 의 raster-scan index:

$$\text{idx}(i, j) = i \cdot W + j$$

순서: $(0, 0), (0, 1), \ldots, (0, W-1), (1, 0), \ldots, (H-1, W-1)$.

"$(i, j) < (i', j')$" 는 $\text{idx}(i, j) < \text{idx}(i', j')$ 로 정의.

### 정의 2.2 — Masked Convolution

$k \times k$ kernel $K$ 에 mask $M$ ($k \times k$ 0/1 matrix) 를 element-wise 곱한 effective kernel $\tilde K = M \odot K$.

**Mask A** ($k \times k$, $k$ 홀수, center at $\lfloor k/2 \rfloor$):
$$M_A[u, v] = \begin{cases} 1 & \text{if } u < \lfloor k/2 \rfloor \\ 1 & \text{if } u = \lfloor k/2 \rfloor \text{ and } v < \lfloor k/2 \rfloor \\ 0 & \text{otherwise} \end{cases}$$

**Mask B**: $u = \lfloor k/2 \rfloor$ and $v = \lfloor k/2 \rfloor$ (center) 가 1, 나머지는 mask A 와 동일.

### 정의 2.3 — Stacked Masked Convolution

PixelCNN 의 forward:

$$h^{(0)} = x \quad (\text{입력 이미지})$$
$$h^{(1)} = \sigma(h^{(0)} * \tilde K^{(1)}_A + b^{(1)}) \quad (\text{mask A})$$
$$h^{(\ell)} = \sigma(h^{(\ell-1)} * \tilde K^{(\ell)}_B + b^{(\ell)}) \quad \ell \geq 2 \text{ (mask B)}$$
$$\text{logits} = h^{(L)} * W_\text{out}$$

마지막 layer 의 output 이 각 픽셀에 대한 categorical 분포 (256 classes for 8-bit images).

### 정리 2.4 — Mask A 가 첫 layer 에서 필수인 이유

**주장**: AR 의 valid factorization 을 위해 첫 layer 가 mask A 여야 한다.

**증명**: Conditional $p(x_{ij} | x_{<ij})$ 의 평가는 $x_{ij}$ 자체를 input 으로 받지 않아야 함. 첫 layer 에서 mask B 를 쓰면 center pixel ($x_{ij}$) 이 input → "자기 자신을 보는" cheating, $p(x_{ij}) = 1$ 이 가능 (degenerate). $\square$

### 정리 2.5 — Mask B 가 deeper layer 에서 valid 한 이유

첫 layer 에서 mask A 로 만들어진 feature $h^{(1)}_{ij}$ 는 $x_{<ij}$ 만 의존, $x_{ij}$ 는 보지 않음. 따라서 $h^{(1)}_{ij}$ 를 deeper layer 에서 center 로 봐도 (mask B), $x_{ij}$ 정보가 leak 되지 않음.

**귀납**: 모든 layer 의 feature $h^{(\ell)}_{ij}$ 는 $x_{<ij}$ 만 함수. 따라서 final logits 가 $p(x_{ij} | x_{<ij})$ 의 valid parameterization. $\square$

### 정리 2.6 — Receptive Field 와 RF Blind Spot

$k = 3$ Masked conv $L$ layers stack:

- Mask A 첫 layer: RF 가 $(0, 0), (0, 1), (1, 0)$ 등 — 위쪽 절반 + 같은 row 의 왼쪽
- Mask B 추가 layer: RF 확장하지만 같은 패턴 유지

**Blind spot**: Mask A 의 sym 깨진 구조로 인해, 같은 row 의 오른쪽 위 영역 일부가 영원히 못 보는 영역 발생. **Gated PixelCNN** (van den Oord 2016b) 이 vertical + horizontal stack 으로 해결.

---

## 🔬 증명 및 수학적 유도

### 유도 1 — Bits per Dimension 계산

$32 \times 32 \times 3$ CIFAR-10 이미지에 대해, NLL 단위 변환:

$$\text{bpd} = \frac{-\log p_\theta(x) / \log 2}{32 \cdot 32 \cdot 3} = \frac{\text{NLL}}{3072 \cdot \log 2}$$

PixelCNN 의 typical CIFAR-10 bpd: 3.03 (Salimans 2017 PixelCNN++). 이론 한계 (uniform distribution): $\log_2 256 = 8$ bpd. 따라서 실제 데이터는 이론보다 **약 2.6배** 적은 정보로 표현 가능 — natural image 의 강한 structure 반영.

### 유도 2 — Mask A 의 Causal Equivalence with Sequential Index

Mask A $k \times k$ centered conv 의 output at $(i, j)$:

$$y_{ij} = \sum_{(u, v) \in \mathcal{S}_A} K[u, v] \cdot x_{i + u - c, j + v - c}$$

여기서 $c = \lfloor k/2 \rfloor$, $\mathcal{S}_A$ 는 mask A 의 1 위치 집합. 이 집합은 $(u, v)$ 가 "row $u < c$" 또는 "$u = c, v < c$" 를 만족 — 즉 $i + u - c \leq i$ and ($i + u - c < i$ 이거나 $j + v - c < j$).

$\mathcal{S}_A$ 가 정확히 raster-scan 의 "이전 픽셀" 들과 매칭되도록 설계됨. 따라서 conv 가 chain rule 의 conditional dependency 를 정확히 구현.

### 유도 3 — Multi-Channel RGB AR

각 픽셀 $(i, j)$ 의 R, G, B:

$$p(R_{ij}, G_{ij}, B_{ij} | I_{<ij}) = p(R_{ij} | I_{<ij}) \cdot p(G_{ij} | I_{<ij}, R_{ij}) \cdot p(B_{ij} | I_{<ij}, R_{ij}, G_{ij})$$

NN architecture 의 channel-wise mask:
- R prediction: 이전 픽셀의 R, G, B 모두, 같은 pixel 은 X
- G prediction: 이전 픽셀의 R, G, B, 같은 pixel 의 R
- B prediction: 이전 픽셀의 R, G, B, 같은 pixel 의 R, G

이 nested mask 로 conditional 의 valid factorization 보장.

### 유도 4 — PixelCNN++ 의 Discretized Logistic Mixture

PixelCNN 의 256-way softmax 는 "near-256 = far-256" 로 동일 페널티 — 인접 색 정보 무시. PixelCNN++ (Salimans 2017) 의 해결: discretized mixture of logistics:

$$p(x | \pi, \mu, s) = \sum_k \pi_k \cdot [\text{CDF}((x + 0.5 - \mu_k) / s_k) - \text{CDF}((x - 0.5 - \mu_k) / s_k)]$$

연속 logistic 분포에 0.5 width 의 bin 으로 discretize. 이로 색 인접성을 자연스럽게 표현, NLL 0.1 bpd 정도 개선.

---

## 💻 실험 재현 / PyTorch 구현

### 실험 1 — Masked Convolution 직접 구현

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class MaskedConv2d(nn.Conv2d):
    def __init__(self, mask_type, *args, **kwargs):
        super().__init__(*args, **kwargs)
        assert mask_type in ('A', 'B')
        self.register_buffer('mask', self.weight.data.clone())
        _, _, kH, kW = self.weight.size()
        self.mask.fill_(1)
        # 가운데 행의 가운데 이후를 0
        self.mask[:, :, kH // 2, kW // 2 + (mask_type == 'B'):] = 0
        # 가운데 행 다음 (아래) 모두 0
        self.mask[:, :, kH // 2 + 1:] = 0

    def forward(self, x):
        self.weight.data *= self.mask
        return super().forward(x)

# 간단한 PixelCNN
class SimplePixelCNN(nn.Module):
    def __init__(self, channels=64, n_layers=8, n_categories=256):
        super().__init__()
        self.conv1 = MaskedConv2d('A', 1, channels, 7, padding=3)
        self.layers = nn.ModuleList([
            MaskedConv2d('B', channels, channels, 7, padding=3)
            for _ in range(n_layers)
        ])
        self.out = nn.Conv2d(channels, n_categories, 1)
    def forward(self, x):
        # x: [B, 1, H, W] (grayscale, 0~1 → quantize)
        h = F.relu(self.conv1(x))
        for layer in self.layers:
            h = F.relu(layer(h))
        return self.out(h)   # [B, 256, H, W] logits

# MNIST 훈련
import torchvision
import torchvision.transforms as T
train_data = torchvision.datasets.MNIST(
    '~/data', download=True, train=True,
    transform=T.Compose([T.ToTensor()])
)
loader = torch.utils.data.DataLoader(train_data, batch_size=128, shuffle=True)

model = SimplePixelCNN().cuda()
opt = torch.optim.Adam(model.parameters(), lr=3e-4)

for epoch in range(5):
    for x, _ in loader:
        x = x.cuda()
        x_int = (x * 255).long().squeeze(1)   # [B, H, W] integer 0~255
        logits = model(x)                      # [B, 256, H, W]
        loss = F.cross_entropy(logits, x_int)
        opt.zero_grad(); loss.backward(); opt.step()
    print(f"Epoch {epoch}: NLL = {loss.item():.3f}")

# Sampling (느림 — H*W 번 forward pass)
@torch.no_grad()
def sample(model, H=28, W=28, batch=4):
    img = torch.zeros(batch, 1, H, W).cuda()
    for i in range(H):
        for j in range(W):
            logits = model(img)[:, :, i, j]    # [B, 256]
            probs = torch.softmax(logits, -1)
            sample_pixel = torch.multinomial(probs, 1).float() / 255.0
            img[:, :, i, j] = sample_pixel
    return img

samples = sample(model)
# 시각화: torchvision.utils.make_grid 후 imshow
```

### 실험 2 — Mask 가 미래를 정말 차단하는지 검증

```python
# 입력의 일부 픽셀을 perturb 하고 output 의 어느 위치가 변하는지 확인
def test_causality(model, image, pixel_pos):
    """pixel_pos = (i, j) 를 perturb 했을 때, output 이 (i, j) 이전 위치에서 변하는지"""
    image2 = image.clone()
    image2[0, 0, pixel_pos[0], pixel_pos[1]] += 1.0   # perturb

    out1 = model(image)
    out2 = model(image2)
    diff = (out1 - out2).abs().sum(1).squeeze()       # [H, W]
    return diff > 1e-6   # boolean map of changed positions

image = torch.rand(1, 1, 8, 8).cuda()
changed = test_causality(model, image, (4, 4))
# 예상: changed[i, j] = True 만 (i, j) > (4, 4) 인 곳 (raster-scan)
# 그 외는 False — 진짜 causal 이라는 의미
```

### 실험 3 — Class-Conditional PixelCNN

```python
class ConditionalPixelCNN(nn.Module):
    def __init__(self, n_classes=10, channels=64, n_layers=8):
        super().__init__()
        self.embed = nn.Embedding(n_classes, channels)
        self.conv1 = MaskedConv2d('A', 1, channels, 7, padding=3)
        self.layers = nn.ModuleList([
            MaskedConv2d('B', channels, channels, 7, padding=3)
            for _ in range(n_layers)
        ])
        self.out = nn.Conv2d(channels, 256, 1)
    def forward(self, x, y):
        h = F.relu(self.conv1(x))
        cond = self.embed(y).unsqueeze(-1).unsqueeze(-1)   # [B, C, 1, 1]
        for layer in self.layers:
            h = F.relu(layer(h) + cond)        # broadcast addition
        return self.out(h)

# class label 에 따라 다른 digit 생성
```

---

## 🔗 이론과 실전의 간극

### 1. PixelCNN의 Blind Spot

기본 mask A 는 같은 row 의 오른쪽 위 일부 영역을 영원히 못 봄 (RF 의 비대칭 구조). Gated PixelCNN (van den Oord 2016b) 이 **vertical stack** (위쪽 모든 row) + **horizontal stack** (같은 row 왼쪽) 으로 분리하여 blind spot 제거. PixelCNN++ 가 이를 채택.

### 2. Categorical vs Discretized Mixture

8-bit pixel = 256-way softmax 는 표현력 강하지만 **인접 색 사이 smoothness** 무시. Discretized Logistic Mixture (PixelCNN++) 가 인접성 활용 → NLL 개선 + sample quality 개선.

현대적 변형: VQ-VAE 의 codebook + AR Transformer 가 더 일반적 (DALL-E, Parti).

### 3. Sample Quality 의 한계

PixelCNN 은 NLL 좋지만 sample 이 종종 noisy/blurry — 이론적 이유:
- Forward KL 의 mass-covering → 모든 mode 평균
- High-frequency content 모델링 어려움 (Spectral bias)
- Sequential sampling 의 누적 오차

이것이 GAN, Diffusion 으로 이동하는 동기 중 하나. 단, **언어 모델 (GPT) 의 경우** sequence 의 discreteness 와 huge data 로 AR 이 압도적 — modality 별로 우열.

---

## ⚖️ 가정과 한계

| 가정 | 한계 |
|------|------|
| Raster-scan 이 자연 순서 | 다른 순서 (zigzag, Hilbert) 에서 결과 다를 수 있음 |
| 8-bit categorical | 인접 색 smoothness 무시 (PixelCNN++ 에서 해결) |
| Bounded receptive field | 깊은 layer 필요, blind spot (Gated PixelCNN 해결) |
| Tractable likelihood = good model | Sample quality 가 NLL 과 약한 상관 |
| RGB sub-pixel order | Order convention (RGB) 에 의존, 일반적이지 않음 |
| Sequential sampling | $H \cdot W$ forward passes — 매우 느림 |

---

## 📌 핵심 정리

$$\boxed{p(I) = \prod_{i,j,c} p(I_{i,j,c} | I_{<(i,j,c)}) \text{ — Raster-scan + sub-pixel AR}}$$

$$\boxed{\text{Mask A (자기 차단, 첫 layer), Mask B (자기 OK, 깊은 layer)}}$$

| 구성요소 | 역할 |
|---------|------|
| **Raster-scan** | 2D 이미지를 1D sequence 로 |
| **Masked conv** | Future pixel 차단, parallel training 가능 |
| **Mask A** | 첫 layer, center 차단 — $x_{ij}$ 자체 불참조 |
| **Mask B** | Deeper layer, center 허용 — $x_{<ij}$ 만 의존하는 feature |
| **Categorical out** | 256-way softmax (또는 discretized mixture) |
| **Conditional** | Class/text embedding 을 broadcast 추가 |
| **Sampling cost** | $O(H \cdot W \cdot C)$ sequential forward pass |

---

## 🤔 생각해볼 문제

**문제 1** (기초): $5 \times 5$ mask A 와 mask B 를 그림으로 그리고, 각각의 1 의 개수를 세어라. Receptive field 가 어떻게 차이 나는가?

<details>
<summary>해설</summary>

**Mask A** ($5 \times 5$, center at (2, 2)):
```
1 1 1 1 1
1 1 1 1 1
1 1 0 0 0
0 0 0 0 0
0 0 0 0 0
```
1의 개수: 12 (위쪽 2줄 10개 + 가운데 줄 왼쪽 2개)

**Mask B**: 같지만 center (2, 2) = 1 → 1의 개수: 13

**RF 차이**: B 가 center 1 추가 — 이전 layer 의 같은 위치 feature 를 봄. Stack 하면 RF 가 같은 위치 정보를 누적, 더 풍부한 표현.

**Blind spot**: 두 mask 모두 (2, 3), (2, 4) 등 같은 row 의 오른쪽은 가리지만, 깊이 쌓아도 (1, 3), (1, 4) 등 위쪽 row 의 오른쪽 일부는 영원히 RF 안에 들어오지 않음 (mask 의 비대칭 구조). 이를 **blind spot** 이라 함.

</details>

**문제 2** (심화): RGB 채널 처리 시 channel-wise mask 의 정확한 구조를 그리고, R/G/B 각각의 prediction 에서 어느 input 채널을 보는지 명시하라.

<details>
<summary>해설</summary>

Output channel 을 $\{R_\text{out}, G_\text{out}, B_\text{out}\}$, input channel 을 $\{R_\text{in}, G_\text{in}, B_\text{in}\}$ 라 하자. Center pixel 의 mask:

**Mask A** (첫 layer):
| Out \\ In | $R_\text{in}$ | $G_\text{in}$ | $B_\text{in}$ |
|----------|---------------|---------------|---------------|
| $R_\text{out}$ | 0 | 0 | 0 |
| $G_\text{out}$ | 1 | 0 | 0 |
| $B_\text{out}$ | 1 | 1 | 0 |

**Mask B** (deep layer):
| Out \\ In | $R_\text{in}$ | $G_\text{in}$ | $B_\text{in}$ |
|----------|---------------|---------------|---------------|
| $R_\text{out}$ | 1 | 0 | 0 |
| $G_\text{out}$ | 1 | 1 | 0 |
| $B_\text{out}$ | 1 | 1 | 1 |

**해석**: $G_\text{out}$ 은 같은 픽셀의 $R$ 을 봄 (R 이 G 의 conditional 이므로). $B_\text{out}$ 은 R, G 모두. R 은 자기 자신만 (mask B 에서) 봄.

이 구조로 $p(R, G, B | \text{prev pixels}) = p(R | \cdot) p(G | \cdot, R) p(B | \cdot, R, G)$ 의 valid factorization.

</details>

**문제 3** (논문 비평): PixelCNN 의 NLL 이 Diffusion 보다 비슷하거나 더 좋을 수 있는데, 왜 sample quality 는 Diffusion 이 압도적인가? Theis 2016 의 관점에서 설명하라.

<details>
<summary>해설</summary>

**관찰**: CIFAR-10 에서 PixelCNN++ NLL ≈ 3.03 bpd, DDPM NLL ≈ 3.17 bpd — PixelCNN 이 NLL 약간 좋음. 하지만 FID: PixelCNN ≈ 50, DDPM ≈ 3 — DDPM 이 압도적.

**Theis 2016 관점**: NLL 은 데이터 분포 위의 평균 likelihood, sample quality (FID) 는 generator 분포의 perceptual quality.

**왜 PixelCNN 의 sample 이 좋지 않은가**:
1. **Sequential 누적 오차**: 한 픽셀의 약간 이상한 sampling 이 다음 픽셀에 영향 → cascade
2. **Forward KL 의 mass-covering**: 모든 mode cover 하려고 평균화 → blurry
3. **Per-pixel categorical 의 한계**: 색의 spatial correlation 모델링 약함

**왜 Diffusion sample 이 좋은가**:
1. **Score-based**: $\nabla \log p$ 학습 — 데이터 manifold 로의 "끌어당김" 직접 학습
2. **Iterative refinement**: 점진적으로 noise 제거 → cascade error 없음
3. **Perceptually-aligned loss**: $L_\text{simple}$ 이 perceptual quality 와 잘 align

**일반화**: NLL 과 FID 는 다른 측면 — 단일 metric 으로 모델 비교 부적절. Diffusion 이 두 metric 에서 모두 잘하는 이유는 architecture + objective 의 inherent advantage.

</details>

---

<div align="center">

[◀ 이전 (01. Chain Rule)](./01-chain-rule-factorization.md) | [📚 README](../README.md) | [다음 ▶ (03. WaveNet)](./03-wavenet.md)

</div>
