# 04. 평가 지표 — IS · FID · Precision/Recall · NLL

## 🎯 핵심 질문

- Inception Score $\text{IS} = \exp(\mathbb{E}_x[\text{KL}(p(y|x) \| p(y))])$ 의 수학적 의미는? 왜 단점이 있고 왜 여전히 보고되는가?
- FID $= \|\mu_r - \mu_g\|^2 + \text{tr}(\Sigma_r + \Sigma_g - 2(\Sigma_r \Sigma_g)^{1/2})$ 의 Fréchet distance 형태는 어떻게 유도되며, 왜 Inception feature 위에서 측정하는가?
- Precision 과 Recall (Kynkäänniemi 2019) 이 어떻게 quality 와 diversity 를 분해하는가? FID 만으로는 왜 부족한가?
- NLL 과 perceptual metric 이 약한 상관관계인 이유 (Theis 2016) 는? 두 척도를 모두 보고해야 하는가?
- LPIPS · KID · CLIP score 등 현대 metric 은 어떤 한계를 보완하는가?

---

## 🔍 왜 평가 지표가 결정적인가

생성 모델 연구의 가장 어려운 부분 중 하나는 **"좋은 모델" 의 정의**입니다. 분류기는 accuracy 가 명확하지만, 생성 모델은:

1. **무엇을 측정해야 하는가** — Sample quality? Mode coverage? Diversity? Fidelity to data?
2. **어떻게 객관적으로 측정하는가** — Human evaluation 은 비용 · 비일관성, automated metric 은 proxy

이 문서에서는 4가지 주요 metric — **IS, FID, Precision/Recall, NLL** — 의 수학적 정의, 측정 방법, 한계를 다룹니다. 이 metric 들이 어떻게 **각 family 의 차이를 드러내는지** 가 핵심.

이 metric 들은 단순히 "비교용 숫자" 가 아니라, **모델의 어떤 측면을 평가하는지** 를 결정합니다:
- IS: per-sample confidence + diversity
- FID: distribution-level distance
- Precision/Recall: quality vs coverage 분해
- NLL: 데이터 분포 fitting 의 information-theoretic 척도

---

## 📐 수학적 선행 조건

- [Probability Theory Deep Dive](https://github.com/iq-ai-lab/probability-theory-deep-dive): KL divergence, multivariate Gaussian
- [Information Theory Deep Dive](https://github.com/iq-ai-lab/information-theory-deep-dive): Entropy, mutual information
- 이전 문서: 03-kl-minimization-unification.md

---

## 📖 직관적 이해

### "사진 콘테스트의 평가위원들"

생성된 이미지를 평가하는 4가지 시각:

**IS — "각 사진은 명확한 주제이지만 콘테스트 전체는 다양한 주제인가"**: 각 이미지가 분류기에서 confident (entropy 낮음) + 전체 분포가 다양 (marginal entropy 높음).

**FID — "생성 이미지 그룹과 진짜 이미지 그룹의 통계적 거리"**: Inception 의 feature space 에서 두 그룹의 평균과 공분산 비교.

**Precision/Recall — "사진의 품질" 과 "주제의 다양성" 을 분리**:
- Precision: 생성된 이미지가 진짜 manifold 에 얼마나 가까운가 (quality)
- Recall: 진짜 이미지가 생성된 manifold 로 cover 되는가 (diversity)

**NLL — "데이터 분포 매뉴얼에 새 사진이 얼마나 맞는가"**: $\log p_\theta(x_\text{test})$ 의 평균 — explicit 모델만 가능.

### Metric 별 강점·약점

| Metric | 측정 대상 | 장점 | 약점 |
|--------|----------|------|------|
| **IS** | Per-sample confidence + class diversity | 간단, 빠름 | Mode coverage 무시, ImageNet 의존 |
| **FID** | Distribution distance | 표준, sample quality 와 상관 | Inception 의존, mode collapse 둔감 |
| **Precision/Recall** | Quality vs diversity 분리 | 두 측면 분해 | k-NN graph hyperparameter 의존 |
| **NLL** | Likelihood fitting | 정보론적 의미 | Implicit 모델 불가, FID 와 상관 약함 |

---

## ✏️ 엄밀한 정의·정리

### 정의 4.1 — Inception Score (Salimans 2016)

ImageNet pretrained Inception classifier $p(y | x)$ 가 주어졌을 때, 생성 분포 $p_\theta$ 의 IS:

$$\text{IS}(p_\theta) = \exp\left(\mathbb{E}_{x \sim p_\theta}\left[\text{KL}(p(y|x) \| p(y))\right]\right)$$

여기서 marginal $p(y) = \mathbb{E}_{x \sim p_\theta}[p(y|x)]$.

**해석**:
- 각 $x$ 에 대해 $p(y|x)$ 가 sharp (한 class 에 confident) 면 좋음 → high IS
- $p(y)$ 가 uniform (다양한 class 에 분산) 이면 좋음 → high IS
- 두 조건이 모두 충족되면 IS 가 큼

**범위**: $1 \leq \text{IS} \leq C$ ($C$ = class 수, e.g., ImageNet 1000)

### 정의 4.2 — Fréchet Inception Distance (Heusel 2017)

Inception feature extractor $\phi: x \to \mathbb{R}^{2048}$ (pool3 layer) 에 대해, 진짜 데이터 features 의 평균 $\mu_r$, 공분산 $\Sigma_r$, 생성 features 의 $\mu_g, \Sigma_g$. **두 분포를 multivariate Gaussian 으로 가정**하고 Fréchet distance 계산:

$$\text{FID} = \|\mu_r - \mu_g\|^2 + \text{tr}\left(\Sigma_r + \Sigma_g - 2(\Sigma_r \Sigma_g)^{1/2}\right)$$

**범위**: $\text{FID} \geq 0$, $\text{FID} = 0 \iff$ 두 Gaussian 이 일치.

### 정의 4.3 — Precision and Recall (Kynkäänniemi 2019)

진짜 sample set $X_r = \{\phi(x_i)\}_i$, 생성 sample set $X_g = \{\phi(\tilde x_j)\}_j$ 의 $k$-NN manifold:

$$\text{manifold}(X) = \bigcup_{x \in X} B(x, r_k(x))$$

여기서 $r_k(x)$ 는 $x$ 의 $k$-NN 거리.

**Precision**: $\frac{1}{|X_g|} \sum_{\tilde x \in X_g} \mathbb{1}[\tilde x \in \text{manifold}(X_r)]$
— 생성된 sample 중 진짜 manifold 안에 있는 비율 (quality).

**Recall**: $\frac{1}{|X_r|} \sum_{x \in X_r} \mathbb{1}[x \in \text{manifold}(X_g)]$
— 진짜 sample 중 생성된 manifold 로 cover 되는 비율 (diversity).

**Trade-off**: GAN 은 보통 high precision, low recall (mode collapse). VAE 는 medium precision, high recall (mode coverage). Diffusion 은 둘 다 높음.

### 정의 4.4 — Negative Log-Likelihood

테스트 데이터 $\{x_i^{\text{test}}\}_{i=1}^n$ 에 대해:

$$\text{NLL} = -\frac{1}{n}\sum_i \log p_\theta(x_i^{\text{test}})$$

**Bits per dimension** (bpd) 표현:

$$\text{bpd} = \frac{\text{NLL}}{d \log 2}$$

여기서 $d$ 는 입력 차원 (예: $32 \times 32 \times 3 = 3072$ for CIFAR-10).

**Implicit 모델은 NLL 직접 평가 불가**, lower bound 또는 변분 근사 사용.

### 정리 4.5 — IS 의 정보론적 분해

$$\log \text{IS}(p_\theta) = \mathbb{E}_x[\text{KL}(p(y|x) \| p(y))] = I(X; Y) = H(Y) - H(Y|X)$$

여기서 $I$ 는 mutual information.

**해석**:
- $H(Y)$: 분류기 출력의 marginal entropy (큰 것 좋음 — diversity)
- $H(Y|X)$: 조건부 entropy (작은 것 좋음 — sharpness)

따라서 IS 는 **mutual information 의 지수**.

**증명**: $\mathbb{E}_x[\text{KL}(p(y|x) \| p(y))] = \mathbb{E}_x \mathbb{E}_{y|x}[\log p(y|x) - \log p(y)] = \mathbb{E}_{x, y}[\log p(y|x)] - \mathbb{E}_y[\log p(y)] = -H(Y|X) + H(Y) = I(X; Y)$. $\square$

### 정리 4.6 — FID 의 Fréchet Distance 유도

두 multivariate Gaussian $\mathcal{N}(\mu_r, \Sigma_r), \mathcal{N}(\mu_g, \Sigma_g)$ 의 Wasserstein-2 distance:

$$W_2^2 = \|\mu_r - \mu_g\|^2 + \text{tr}\left(\Sigma_r + \Sigma_g - 2(\Sigma_r^{1/2} \Sigma_g \Sigma_r^{1/2})^{1/2}\right)$$

trace cyclic property 로 $\text{tr}(\Sigma_r^{1/2} \Sigma_g \Sigma_r^{1/2})^{1/2} = \text{tr}(\Sigma_r \Sigma_g)^{1/2}$ (적절한 정의 하에). 따라서 FID = $W_2^2$ 의 Gaussian 가정 형태. (Dowson & Landau 1982)

---

## 🔬 증명 및 수학적 유도

### 유도 1 — IS 가 Mode Coverage 에 둔감한 이유

Mode collapse 된 generator: 모든 sample 이 같은 class 의 다른 instance. 예: 모든 이미지가 "고양이". 이때:

- $p(y|x)$ — 모두 "고양이" 에 confident → $H(Y|X) = 0$ (low)
- $p(y) = $ 모든 이미지의 평균 ≈ 고양이 class concentrated → $H(Y)$ 도 low

따라서 $I(X; Y) = H(Y) - H(Y|X)$ 가 작아 보임 → IS 낮음. **하지만 IS 가 낮은 게 mode collapse 만으로 설명되지는 않음** — 분류기가 잘 못 분류하면 (sample 이 모두 noise) 도 IS 낮음.

**반대 방향**: sample 이 매우 sharp 이고 다양한 class 에 cover 되면 high IS — 단, **각 class 안의 다양성** (within-class diversity) 은 IS 가 무시. 예: 모든 "고양이" 가 같은 고양이 image 의 작은 perturbation 이어도, 다른 class 들이 cover 되면 IS 높음.

### 유도 2 — FID 가 Inception Feature 를 사용하는 이유

Pixel space 에서의 distance (예: MSE) 는 perceptual quality 와 무관. 예: 1픽셀 shift 된 이미지가 perceptually 동일하지만 MSE 큼.

Inception (또는 다른 pretrained CNN) 의 mid/high-level feature 는 **semantic content** 를 encode. 따라서 feature space 에서의 distance 가 perceptual distance 와 더 잘 일치.

**대안들**:
- LPIPS (Zhang 2018): Learned perceptual similarity, AlexNet/VGG features
- KID (Bińkowski 2018): Kernel Inception Distance, biased estimator 없음
- CLIP score: CLIP feature 위의 cosine similarity, text-conditioned

### 유도 3 — Precision/Recall Manifold-Based 정의의 동기

기존 metric (FID) 의 한계: 두 분포가 Gaussian-equivalent 면 같은 FID 라도 mode coverage 가 다를 수 있음. Precision/Recall 은 manifold 의 **set membership** 으로 분리.

수학적 motivation: 진짜 데이터 분포 $p_d$ 의 support 를 $\mathcal{S}_d$, 생성 분포 $p_g$ 의 support 를 $\mathcal{S}_g$ 라 하자.

$$\text{Precision} \approx \frac{p_g(\mathcal{S}_d \cap \mathcal{S}_g)}{p_g(\mathcal{S}_g)}, \quad \text{Recall} \approx \frac{p_d(\mathcal{S}_d \cap \mathcal{S}_g)}{p_d(\mathcal{S}_d)}$$

Empirical 추정은 $k$-NN ball 로 manifold 근사.

### 유도 4 — Theis 2016: NLL 과 Perceptual Quality 의 약한 상관

**핵심 주장**: 같은 NLL 을 가진 두 모델이 sample quality 가 매우 다를 수 있음.

**예시 1**: $p_d$ 와 거의 일치하지만 약간 noisy 한 $p_\theta$ — 평균 NLL 은 좋지만 각 sample 이 noisy.

**예시 2**: $p_d$ 의 작은 부분만 정확히 cover 하는 $p_\theta$ — 평균 NLL 은 나쁘지만 cover 한 부분의 sample 은 sharp.

**수학적 분석**: NLL = $-\mathbb{E}_{p_d}[\log p_\theta]$ 는 **데이터 분포 위의 평균**. 한 mode 가 정확히 modeled 되면 그 mode 의 contribution 이 NLL 을 크게 감소. 반대로, 모든 mode 를 약간 modeled 하면 평균은 medium.

**시사점**: NLL 은 mode coverage 에 민감하지만 per-sample sharpness 에 둔감. FID 는 반대 — 둘 다 보고해야 함.

---

## 💻 실험 재현 / PyTorch 구현

### 실험 1 — IS · FID 계산

```python
import torch
import torch.nn as nn
import torchvision.models as models
from scipy import linalg
import numpy as np

# 1. Inception model setup
inception = models.inception_v3(weights='IMAGENET1K_V1', transform_input=False)
inception.fc = nn.Identity()   # feature output (2048-dim pool3)
inception.eval()

def get_inception_features(images):
    """images: [N, 3, 299, 299], normalized to [-1, 1]"""
    with torch.no_grad():
        return inception(images).cpu().numpy()

def get_inception_logits(images):
    """For IS — class probabilities"""
    incep_full = models.inception_v3(weights='IMAGENET1K_V1').eval()
    with torch.no_grad():
        return torch.softmax(incep_full(images), dim=-1).cpu().numpy()

# 2. Inception Score
def inception_score(images, splits=10):
    probs = get_inception_logits(images)
    N = len(probs); split_size = N // splits
    scores = []
    for i in range(splits):
        p = probs[i*split_size : (i+1)*split_size]
        py = p.mean(0)
        kl = p * (np.log(p + 1e-10) - np.log(py + 1e-10)[None, :])
        scores.append(np.exp(kl.sum(1).mean()))
    return np.mean(scores), np.std(scores)

# 3. Fréchet Inception Distance
def fid(real_images, fake_images):
    f_r = get_inception_features(real_images)
    f_g = get_inception_features(fake_images)
    mu_r, sigma_r = f_r.mean(0), np.cov(f_r, rowvar=False)
    mu_g, sigma_g = f_g.mean(0), np.cov(f_g, rowvar=False)
    diff = mu_r - mu_g
    covmean = linalg.sqrtm(sigma_r @ sigma_g, disp=False)[0]
    if np.iscomplexobj(covmean):
        covmean = covmean.real
    return diff @ diff + np.trace(sigma_r + sigma_g - 2 * covmean)

# 사용
# images = [N, 3, 299, 299], normalized
# is_mean, is_std = inception_score(generated)
# fid_val = fid(real, generated)
```

### 실험 2 — Precision/Recall 측정

```python
def precision_recall(features_r, features_g, k=3):
    """Kynkäänniemi 2019 — k-NN manifold based"""
    # 각 점의 k-NN 거리 (manifold radius)
    def knn_radii(features, k):
        N = len(features)
        d = np.linalg.norm(features[:, None] - features[None, :], axis=-1)
        d.sort(axis=-1)
        return d[:, k]   # k-th nearest

    r_real = knn_radii(features_r, k)
    r_fake = knn_radii(features_g, k)

    # cross-distance
    d_rg = np.linalg.norm(features_r[:, None] - features_g[None, :], axis=-1)

    # precision: 생성 sample 이 진짜 manifold 안에 있는가
    in_real = (d_rg <= r_real[:, None]).any(axis=0)
    precision = in_real.mean()

    # recall: 진짜 sample 이 생성 manifold 안에 있는가
    in_fake = (d_rg <= r_fake[None, :]).any(axis=1)
    recall = in_fake.mean()

    return precision, recall

# 일반적 결과 (CIFAR-10 기준):
#                 P     R    FID  IS
# StyleGAN2     0.65  0.45    9   8.5
# DDPM          0.62  0.55    3   9.5
# VAE           0.40  0.70   30   5.5
# Mode-collapsed GAN   0.80  0.10  60   2.0
```

### 실험 3 — NLL (bits per dimension) 측정

```python
def bits_per_dim(model, dataloader):
    """Explicit 또는 bounded explicit 모델만 가능"""
    total_nll = 0.0
    total_dims = 0
    with torch.no_grad():
        for x, _ in dataloader:
            log_p = model.log_p(x)   # [batch_size]
            total_nll += -log_p.sum().item()
            total_dims += np.prod(x.shape[1:]) * len(x)
    return total_nll / (total_dims * np.log(2))

# 일반적 CIFAR-10 결과:
# - PixelCNN: 3.03 bpd
# - Glow:     3.35 bpd
# - VAE (ELBO): 3.5 bpd (loose bound)
# - DDPM:     3.17 bpd
# - GAN:      N/A (implicit)
```

### 실험 4 — IS의 mode-collapse 둔감 증명

```python
# Mode-collapsed generator: 한 종류의 샘플만 생성 (예: 모두 "cat")
# 진짜 다양 generator: 다양한 클래스
# 두 경우의 IS 비교

# Mock: 모두 cat-like (class 281, IS classifier 의 cat)
mock_collapsed_probs = np.zeros((1000, 1000))
mock_collapsed_probs[:, 281] = 0.99
mock_collapsed_probs[:, :] += 0.01 / 1000

# Mock: 다양
mock_diverse_probs = np.eye(1000)[np.random.randint(0, 1000, 1000)] * 0.99 + 0.01 / 1000

def is_from_probs(probs):
    py = probs.mean(0)
    kl = probs * (np.log(probs + 1e-10) - np.log(py + 1e-10)[None, :])
    return np.exp(kl.sum(1).mean())

print(f"Mode-collapsed IS: {is_from_probs(mock_collapsed_probs):.2f}")  # 약 1
print(f"Diverse IS:       {is_from_probs(mock_diverse_probs):.2f}")    # 큰 값
# IS 는 mode collapse 를 어느 정도 감지하지만, 단 1 cluster 에 모이는 경우만
# 1 cluster + 그 안에서의 within-class diversity 부족은 감지 못함
```

---

## 🔗 이론과 실전의 간극

### 1. Metric 간 불일치

같은 모델이 metric 마다 다른 순위:
- StyleGAN2 vs DDPM: FID 는 비슷하지만 NLL 은 비교 불가 (StyleGAN 은 implicit)
- DALL-E vs Imagen: FID 는 다양하지만 human evaluation 은 다른 결과
- CLIP score 가 text-image alignment 평가에서는 FID 보다 우월

**시사점**: 단일 metric 으로 결정하지 말고, **multi-metric reporting**.

### 2. CLIP-based 평가의 부상

Text-to-image 시대에 FID 가 부족 — text 와의 alignment 측정 필요. CLIP score (cosine similarity in CLIP feature space):

$$\text{CLIPScore}(x, t) = \cos(\phi_\text{img}(x), \phi_\text{text}(t))$$

DALL-E 3, SD3 등의 evaluation 에서 표준이 됨.

### 3. Human Evaluation 의 필요성

Automated metric 은 모두 proxy. 최종 판단은 human:
- **2AFC** (2-alternative forced choice): "어느 게 더 진짜 같나?"
- **Crowdsourcing** (Amazon MTurk): 대규모 평가
- **Inter-annotator agreement**: Cohen's kappa 등으로 일관성 측정

비용이 비싸지만, 새 모델의 정당성 입증에 필수.

---

## ⚖️ 가정과 한계

| 가정 | 한계 |
|------|------|
| Inception classifier 가 universal | Domain mismatch (얼굴, 의료영상) 에서 부정확 |
| Gaussian feature 가정 (FID) | 실제 feature 분포 non-Gaussian 가능 |
| $k$-NN manifold (P/R) | $k$ hyperparameter 의존, computational cost |
| NLL = quality | Theis 2016: 약한 상관관계 |
| Implicit 모델은 NLL 불가 | IWAE bound, AIS 등으로 부분 추정 가능 |
| Single metric 충분 | Multi-metric, CLIP score, human eval 필요 |

---

## 📌 핵심 정리

$$\boxed{\text{IS}(p_\theta) = \exp(\mathbb{E}_x[\text{KL}(p(y|x) \| p(y))]) = \exp(I(X; Y))}$$

$$\boxed{\text{FID} = \|\mu_r - \mu_g\|^2 + \text{tr}(\Sigma_r + \Sigma_g - 2(\Sigma_r \Sigma_g)^{1/2})}$$

$$\boxed{\text{Precision: quality (sample ∈ real manifold), Recall: diversity (real ∈ sample manifold)}}$$

| Metric | 측정 | 모델 | 한계 |
|--------|------|------|------|
| **IS** | $I(X; Y)$, sharpness + class diversity | Image (ImageNet 의존) | Within-class diversity 무시 |
| **FID** | Distribution distance in feature | All families | Inception 의존, Gaussian 가정 |
| **P/R** | Quality 와 Diversity 분해 | All families | $k$-NN hyperparameter |
| **NLL (bpd)** | Likelihood fitting | Explicit only | Sample quality 와 약한 상관 |
| **CLIP score** | Text-image alignment | Conditional | CLIP 의존 |
| **Human eval** | Perceptual gold standard | All | 비용, 비일관성 |

---

## 🤔 생각해볼 문제

**문제 1** (기초): $\log \text{IS} = I(X; Y)$ 임을 보일 때, $H(Y) - H(Y|X)$ 의 두 항이 각각 어떤 측면을 측정하는지 직관적으로 설명하라. 어느 한 쪽만 좋아도 IS 가 높아질 수 있는가?

<details>
<summary>해설</summary>

- $H(Y) = -\sum_y p(y) \log p(y)$: marginal entropy of class predictions. **Diversity 측정** — 다양한 class 에 sample 분포 → 큰 $H(Y)$.
- $H(Y|X) = -\mathbb{E}_x \sum_y p(y|x) \log p(y|x)$: conditional entropy. **Sharpness 측정** — 각 sample 이 한 class 에 confident → 작은 $H(Y|X)$.

IS 는 둘의 차이의 지수, 따라서 **둘 다 좋아야** 큰 IS:
- High $H(Y)$ + low $H(Y|X)$: 다양한 sharp samples → high IS ✓
- Low $H(Y)$ + low $H(Y|X)$: 한 class confident (mode collapse) → low IS
- High $H(Y)$ + high $H(Y|X)$: 다양하지만 fuzzy (noise) → low IS
- Low $H(Y)$ + high $H(Y|X)$: 한 class 이지만 fuzzy → 매우 low IS

따라서 IS 가 둘 다 측정하지만, **within-class diversity** 는 무시 (한 class 의 모든 sample 이 똑같아도 sharpness 만 좋으면 OK).

</details>

**문제 2** (심화): FID = 0 이 두 분포가 정확히 일치함을 함의하는가? 반례를 제시하라.

<details>
<summary>해설</summary>

**반례**: FID 는 **Gaussian 가정** 하의 거리. 두 non-Gaussian 분포가 같은 평균과 공분산을 가지면 FID = 0 이지만 분포 자체는 다를 수 있음.

예: $p_r$ = mixture of 2 Gaussian (modes at $\pm 2$, var 1), $p_g$ = single Gaussian (mean 0, var 5). 두 분포의 평균과 second moment 일치하면 FID 작거나 0 이지만 mode 구조 완전히 다름.

**실전적 시사점**: FID = 0 이 "두 분포 동일" 을 의미하지 않음. Higher-order moments (skewness, kurtosis) 또는 mode 구조 차이는 잡지 못함. 그래서 **Precision/Recall 같은 set-based metric** 이 보완.

</details>

**문제 3** (논문 비평): Theis 2016 의 핵심 주장을 다시 본 후, "NLL 과 FID 둘 다 보고해야 한다" 의 정당성을 논하라. 어느 한 쪽만 좋은 모델의 예시를 제시하라.

<details>
<summary>해설</summary>

**좋은 NLL, 나쁜 FID 예**: PixelCNN — 데이터 분포의 모든 mode 를 cover 하지만, 각 sample 이 noisy/blurry 할 수 있음 (high-frequency content 학습 어려움). NLL 은 mode coverage 에 의해 좋지만, FID 는 perceptual quality 부족으로 나쁨.

**나쁜 NLL, 좋은 FID 예**: StyleGAN2 — Mode coverage 가 부분적이지만 cover 한 부분의 sample 이 매우 sharp. Perceptual quality 우수 (low FID), NLL 은 implicit 으로 측정 불가하지만 IWAE bound 등으로 추정하면 PixelCNN 보다 나쁠 가능성.

**왜 둘 다 보고해야 하는가**:
1. **다른 측면 측정** — NLL (mode coverage), FID (per-sample quality)
2. **사용 사례 의존** — Anomaly detection 이면 NLL 우선, image generation 이면 FID 우선
3. **모델 family 비교** — Implicit 만 비교하려면 FID 필요, explicit 만 비교하려면 NLL 가능
4. **연구 progress 의 정직한 표현** — 한 metric 만 좋게 cherry-pick 하는 publication bias 방지

**Diffusion 의 우월성**: 두 metric 모두 SOTA (low FID + competitive NLL), 따라서 trade-off 를 깬 family.

</details>

---

<div align="center">

[◀ 이전 (03. KL 최소화 통합)](./03-kl-minimization-unification.md) | [📚 README](../README.md) | [다음 ▶ (Ch2-01. Chain Rule Factorization)](../ch2-autoregressive/01-chain-rule-factorization.md)

</div>
