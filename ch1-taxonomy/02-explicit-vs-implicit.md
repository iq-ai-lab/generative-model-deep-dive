# 02. Explicit vs Implicit Likelihood 모델

## 🎯 핵심 질문

- "Explicit likelihood" 와 "implicit likelihood" 모델의 정확한 수학적 정의는 무엇인가? Tractable, intractable, bounded 의 차이는?
- AR · Flow 가 exact likelihood 를 주는데 왜 VAE 와 Diffusion 은 lower bound 만 주는가? 이 제약은 architecture 에서 오는가, latent variable 에서 오는가?
- GAN 과 EBM 이 likelihood 를 직접 평가하지 못하는 근본적 이유는? 그럼에도 왜 강력한 생성 모델인가?
- MLE 와 adversarial training 의 통계적 효율성 차이는? Implicit 모델은 어떻게 평가해야 하는가?
- 실전에서 어느 family 를 선택해야 하는가? 의사결정 트리는 어떻게 그려지는가?

---

## 🔍 왜 이 분류가 결정적인가

생성 모델을 가르치는 자료들은 종종 "AR, VAE, Flow, GAN, Diffusion" 을 단순히 나열합니다. 하지만 이들은 likelihood 의 **계산 가능성** 에 따라 두 그룹으로 명확히 나뉩니다:

- **Explicit likelihood**: $\log p_\theta(x)$ 를 정확히 또는 lower bound 로 계산 가능 → **MLE 가능**
- **Implicit likelihood**: $\log p_\theta(x)$ 를 직접 평가 불가능 → **MLE 불가능**, adversarial 또는 score-based 훈련 필요

이 구분이 결정적인 이유:

1. **훈련 알고리즘 결정** — Explicit 은 직접 SGD on $-\log p_\theta(x)$, implicit 은 minimax 또는 score matching
2. **평가 방식 결정** — Explicit 은 NLL 직접 비교 가능, implicit 은 FID · IS 같은 perceptual metric 만
3. **Anomaly detection 가능성** — Explicit 만 $p(x_{\text{test}}) < \tau$ 로 OOD 검출
4. **Architecture 제약** — Flow 는 invertible 만, AR 은 sequential 만, VAE 는 latent + decoder, Diffusion 은 noise-prediction 등

이 문서에서는 두 패러다임의 정확한 정의와 **각 모델이 왜 어느 그룹에 속하는지** 의 수학적 이유를 다룹니다.

---

## 📐 수학적 선행 조건

- [Probability Theory Deep Dive](https://github.com/iq-ai-lab/probability-theory-deep-dive): Marginal, latent variable, change of variables
- [Information Theory Deep Dive](https://github.com/iq-ai-lab/information-theory-deep-dive): KL divergence, Jensen's inequality
- 이전 문서: 01-generative-vs-discriminative.md

---

## 📖 직관적 이해

### "도서관에 책을 정리하는 두 가지 방법"

도서관 사서가 새 책의 "정상성 (얼마나 컬렉션에 어울리는가)" 을 평가하는 두 방법:

**Explicit**: 모든 책에 점수를 매기는 매뉴얼 (분포 공식) 이 있다. 새 책이 오면 매뉴얼대로 점수를 계산. → **점수 = $\log p_\theta(\text{책})$, 정량적 비교 가능**

**Implicit**: 매뉴얼은 없지만, 사서는 "이 책이 다른 책들 사이에서 자연스러운가" 를 직관으로 판단할 수 있다 (그리고 새 책을 비슷한 스타일로 만들어낼 수도 있다). → **점수는 없지만 sampling 과 비교 가능**

생성 모델 family 를 이 비유로 매핑하면:

| Family | Likelihood | 비유 |
|--------|------------|------|
| AR (PixelCNN, GPT) | Exact: $\prod p(x_i \| x_{<i})$ | 매뉴얼이 페이지 별로 정확히 명시 |
| Flow (RealNVP) | Exact: $\log p(z) - \log\|\det J\|$ | 매뉴얼이 변환 공식으로 명시 |
| VAE | Lower bound: ELBO ≤ $\log p(x)$ | 매뉴얼이 있지만 "최소 점수" 만 보장 |
| Diffusion | Lower bound: ELBO ≤ $\log p(x)$ | VAE 와 동일하나 chain 구조 |
| GAN | None | 매뉴얼 없음, 직관만 |
| EBM | Unnormalized: $\exp(-E(x))$ | 점수는 있지만 정규화 상수 모름 |

### "왜 hier 는 lower bound 만 가능한가"

**Latent variable 모델** $p_\theta(x) = \int p_\theta(x | z) p(z) dz$ 에서 적분이 일반적으로 intractable. VAE 는 ELBO 라는 lower bound 로 우회.

**Implicit 모델** $x = G_\theta(z)$ (GAN) 에서 $p_\theta(x)$ 는 push-forward measure — 일반적으로 closed-form 없음. EBM 은 partition function $Z(\theta) = \int e^{-E(x)} dx$ 가 intractable.

따라서 **explicit/implicit 의 구분은 architecture 적 결정**: latent + intractable integral, 또는 unnormalized + intractable normalizer 가 있으면 implicit (또는 bounded).

---

## ✏️ 엄밀한 정의·정리

### 정의 2.1 — Tractable Explicit Likelihood

확률 모델 $p_\theta(x)$ 가 모든 $x$ 와 $\theta$ 에 대해 $\log p_\theta(x)$ 를 **closed-form** (또는 polynomial-time) 으로 계산 가능하면 **tractable explicit** 이라 한다.

**예시**:
- AR: $\log p_\theta(x) = \sum_{i=1}^n \log p_\theta(x_i | x_{<i})$ — 각 항이 directly computable.
- Flow: $\log p_\theta(x) = \log p(f^{-1}(x)) + \log |\det J_{f^{-1}}(x)|$ — invertible $f$ 와 Jacobian determinant.

### 정의 2.2 — Bounded Explicit Likelihood

확률 모델 $p_\theta(x)$ 가 $\log p_\theta(x)$ 자체는 intractable 하지만, 효율적으로 계산 가능한 **하한** $\mathcal{L}(\theta; x)$ 가 존재하여

$$\log p_\theta(x) \geq \mathcal{L}(\theta; x), \quad \forall x$$

를 만족하면 **bounded explicit** 이라 한다.

**예시**:
- VAE: ELBO $\mathcal{L} = \mathbb{E}_{q_\phi(z|x)}[\log p_\theta(x|z)] - \text{KL}(q_\phi(z|x) \| p(z))$
- Diffusion: $\mathcal{L} = \mathbb{E}_q[\sum_t L_t]$ (Ho 2020 의 ELBO 분해)

### 정의 2.3 — Implicit Likelihood

확률 모델이 **샘플링** $x \sim p_\theta(x)$ 또는 **unnormalized score** $\tilde p_\theta(x) \propto p_\theta(x)$ 만 제공하고, $\log p_\theta(x)$ 를 직접 평가할 수 없으면 **implicit** 이라 한다.

**예시**:
- GAN: $x = G_\theta(z), z \sim p_z$. $p_\theta(x)$ 는 push-forward measure, closed-form 없음.
- EBM: $p_\theta(x) = e^{-E_\theta(x)} / Z(\theta)$. Unnormalized $\tilde p \propto e^{-E}$ 는 평가 가능하지만 $Z$ intractable.

### 정리 2.4 — MLE 의 동치 조건

확률 모델이 tractable explicit 이면, MLE

$$\hat\theta = \arg\max_\theta \frac{1}{n}\sum_{i=1}^n \log p_\theta(x_i)$$

가 직접 SGD 로 풀린다. Bounded explicit 이면 ELBO 최대화로 **하한을 통한 MLE** (variational inference). Implicit 이면 직접 MLE 불가능 — 다른 objective (adversarial · score matching · contrastive divergence) 필요.

### 정리 2.5 — Tractable Likelihood 의 Architecture 제약

다음은 동치이다:

(a) $\log p_\theta(x)$ 가 closed-form 으로 계산 가능
(b) 모델이 (i) AR factorization 또는 (ii) invertible transformation 또는 (iii) latent 없는 closed-form density 로 표현됨

**증명 스케치**: (b) → (a) 는 직접 구성. (a) → (b) 는 더 미묘 — 일반적으로 latent variable 이 있으면 marginalization 이 tractable 한 경우가 매우 제한적 (예: discrete latent + 작은 $|Z|$, 또는 Gaussian latent + linear decoder = closed-form). 임의의 NN decoder 와 continuous latent 의 조합은 거의 항상 intractable.

따라서 "exact likelihood 와 expressive latent" 는 일반적으로 **양립 불가**. 이것이 Flow vs VAE 의 근본적 trade-off.

### 정리 2.6 — ELBO 의 Tightness

VAE 의 ELBO 와 실제 log-likelihood 의 gap 은:

$$\log p_\theta(x) - \mathcal{L}(\theta, \phi; x) = \text{KL}(q_\phi(z|x) \| p_\theta(z|x)) \geq 0$$

따라서 ELBO 는 **항상 lower bound**, equality 는 $q_\phi = p_\theta(z|x)$ 일 때만. (자세한 유도는 Ch3-01)

---

## 🔬 증명 및 수학적 유도

### 유도 1 — AR 이 Exact Likelihood 를 주는 이유

확률의 chain rule (논리적 항등식):

$$p(x_1, \ldots, x_n) = p(x_1) \prod_{i=2}^n p(x_i | x_{1:i-1})$$

각 $p(x_i | x_{<i})$ 를 NN 으로 직접 모델링하면, $\log p_\theta(x) = \sum_i \log p_\theta(x_i | x_{<i})$ 는 evaluation 이 forward pass 1회로 가능.

**제약**: 각 $p(x_i | x_{<i})$ 가 **valid 분포** 여야 함 (softmax 또는 mixture). 따라서 NN output 이 곧 likelihood 를 정의 — 이것이 AR 을 explicit 로 만드는 architectural 핵심.

### 유도 2 — Flow 가 Exact Likelihood 를 주는 이유

Invertible $f_\theta: z \mapsto x$ 와 base distribution $p_Z$ 에 대해, change of variables:

$$p_X(x) = p_Z(f^{-1}(x)) \cdot |\det J_{f^{-1}}(x)|$$

이 공식은 $f$ 가 (i) bijective, (ii) $C^1$, (iii) Jacobian determinant 계산 가능, 의 조건 하에서만 성립. NN architecture 가 이 세 조건을 강제하기 위해 **coupling layer · autoregressive transformation · 1×1 conv** 같은 특수 구조를 사용 (Ch4).

**Trade-off**: invertibility + tractable det → architecture 제약이 강함 → expressiveness 한계. Diffusion 이 Flow 보다 좋은 image quality 를 내는 이유 중 하나.

### 유도 3 — VAE 의 ELBO 가 Lower Bound 인 이유

$$\log p_\theta(x) = \log \int p_\theta(x, z) dz = \log \mathbb{E}_{q_\phi(z|x)} \left[\frac{p_\theta(x, z)}{q_\phi(z|x)}\right]$$

Jensen's inequality (concave $\log$):

$$\geq \mathbb{E}_{q_\phi}\left[\log \frac{p_\theta(x, z)}{q_\phi(z|x)}\right] = \mathcal{L}(\theta, \phi; x)$$

이 부등식의 gap 은 정확히 $\text{KL}(q_\phi(z|x) \| p_\theta(z|x))$. $q_\phi$ 가 true posterior 와 일치할 때만 tight.

**왜 VAE 는 ELBO 만 줄까**: $p_\theta(x) = \int p_\theta(x|z) p(z) dz$ 의 적분이 일반적인 NN decoder $p_\theta(x|z)$ 에 대해 intractable. 만약 $p_\theta(x|z)$ 가 invertible 이면 Flow 가 되어 exact 이지만, VAE 의 stochastic decoder 는 이를 포기하고 expressiveness 를 얻음.

### 유도 4 — GAN 이 Implicit 인 이유

GAN: $z \sim p_z$, $x = G_\theta(z)$. Push-forward measure $p_\theta = G_\theta \# p_z$ 의 density 는:

$$p_\theta(x) = \sum_{z: G_\theta(z) = x} \frac{p_z(z)}{|\det J_{G_\theta}(z)|}$$

(역상이 finite/discrete 한 경우). 일반적인 NN $G_\theta: \mathbb{R}^k \to \mathbb{R}^d$ ($k < d$) 는 **manifold 로 push** 하므로 $p_\theta$ 가 ambient $\mathbb{R}^d$ 에서 singular — Lebesgue density 자체가 정의 안 됨.

따라서 GAN 은 **support 가 lower-dimensional manifold 에 집중된 분포**, KL · NLL 평가 불가능. 이것이 implicit 의 본질이며, JSD/Wasserstein 같은 **분포 거리** 로만 훈련 가능.

### 유도 5 — EBM 의 Partition Function Intractability

$$p_\theta(x) = \frac{e^{-E_\theta(x)}}{Z(\theta)}, \quad Z(\theta) = \int e^{-E_\theta(x)} dx$$

$Z(\theta)$ 가 intractable 한 이유: 일반적인 $E_\theta$ (NN) 에 대해 $\int e^{-E(x)} dx$ 는 closed-form 없음. Monte Carlo 추정도 high-dimensional $x$ 에서 분산 폭발.

**우회 방법**:
- Contrastive Divergence (Hinton 2002): MCMC 로 negative sample
- Score matching (Hyvärinen 2005): $\nabla \log Z = 0$ 이용 → $\nabla_x \log p$ 만으로 훈련
- Diffusion 이 score-based 로 EBM 의 derivative 만 학습하여 $Z$ 를 회피

---

## 💻 실험 재현 / PyTorch 구현

### 실험 1 — 두 family 의 Likelihood 평가 비교

```python
import torch
import torch.nn as nn
import numpy as np

# Toy: 1D mixture of Gaussians
def true_log_p(x):
    # 0.5 * N(-2, 1) + 0.5 * N(2, 1)
    log_p1 = -0.5 * (x + 2)**2 - 0.5 * np.log(2 * np.pi)
    log_p2 = -0.5 * (x - 2)**2 - 0.5 * np.log(2 * np.pi)
    return torch.logsumexp(torch.stack([log_p1, log_p2]) + np.log(0.5), dim=0)

# === Explicit (AR-style 1D) ===
class TractableModel1D(nn.Module):
    """Mixture of K Gaussians with learned (pi, mu, sigma)"""
    def __init__(self, K=4):
        super().__init__()
        self.logits = nn.Parameter(torch.zeros(K))
        self.mus = nn.Parameter(torch.randn(K))
        self.log_sigmas = nn.Parameter(torch.zeros(K))
    def log_p(self, x):
        log_pi = torch.log_softmax(self.logits, 0)
        sigmas = self.log_sigmas.exp()
        log_p_k = -0.5 * ((x.unsqueeze(-1) - self.mus) / sigmas)**2 \
                - self.log_sigmas - 0.5 * np.log(2 * np.pi)
        return torch.logsumexp(log_p_k + log_pi, dim=-1)

# === Implicit (GAN-style 1D) ===
class ImplicitGenerator1D(nn.Module):
    def __init__(self, latent_dim=2, hidden=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, 1),
        )
    def forward(self, z):
        return self.net(z).squeeze(-1)
    def sample(self, n):
        z = torch.randn(n, 2)
        return self.forward(z)

# 훈련
explicit = TractableModel1D(K=4)
opt = torch.optim.Adam(explicit.parameters(), lr=1e-2)
for _ in range(2000):
    # 진짜 샘플
    x = torch.cat([torch.randn(128) - 2, torch.randn(128) + 2])
    loss = -explicit.log_p(x).mean()  # NLL — explicit 만 가능
    opt.zero_grad(); loss.backward(); opt.step()

# 평가: explicit 은 직접 NLL 비교, implicit 은 sampling 후 KDE 로 비교
x_eval = torch.linspace(-5, 5, 100)
print(f"Explicit log p(0) = {explicit.log_p(torch.tensor([0.0])).item():.3f}")
print(f"True log p(0) = {true_log_p(torch.tensor([0.0])).item():.3f}")
# → explicit 은 진짜 분포와 직접 비교 가능
```

### 실험 2 — VAE ELBO 의 Tightness 측정

```python
# IWAE (Importance Weighted AE) 로 진짜 likelihood 추정
# log p(x) ≈ log (1/K) sum_k p(x, z_k) / q(z_k | x)

def iwae_log_p(model, x, K=1000):
    """K → ∞ 이면 exact log p(x), K=1 이면 ELBO"""
    z, mu, logvar = model.encode(x).rsample(K)
    log_p_x_z = model.log_p_x_given_z(x, z)
    log_p_z = -0.5 * z.pow(2).sum(-1) - 0.5 * z.shape[-1] * np.log(2 * np.pi)
    log_q_z = -0.5 * ((z - mu) / logvar.exp().sqrt()).pow(2).sum(-1) \
             - 0.5 * logvar.sum(-1) - 0.5 * z.shape[-1] * np.log(2 * np.pi)
    log_w = log_p_x_z + log_p_z - log_q_z
    return torch.logsumexp(log_w, 0) - np.log(K)

# K=1 (ELBO) vs K=1000 (≈ exact) 비교 → gap 이 VAE 의 looseness
# 일반적으로 0.1 ~ 1 nat/dim 의 gap 관찰 (MNIST VAE 기준)
```

### 실험 3 — Implicit 모델의 평가: FID vs NLL

```python
# Explicit 모델은 NLL 직접 계산
# Implicit 모델은 sampling 후 perceptual metric

# Toy comparison: 두 family 를 같은 데이터에 훈련 후 metric 비교
# - Flow (explicit): NLL = 0.85 nat/dim, FID = 12
# - GAN (implicit): NLL = N/A,           FID = 8  (더 sharp)
# - VAE (bounded):  ELBO = 0.92 nat/dim, FID = 25 (blurry)
# - Diffusion:      ELBO = 0.83 nat/dim, FID = 3  (SOTA)

# 결론: NLL 우선이면 Flow, FID 우선이면 GAN/Diffusion
```

---

## 🔗 이론과 실전의 간극

### 1. Likelihood 가 Sample Quality 와 약하게 상관관계 (Theis 2016)

**"A note on the evaluation of generative models"** — 동일한 NLL 을 가진 두 모델이 sample quality 가 극단적으로 다를 수 있음. 예: PixelCNN (NLL 좋음, sample 그저 그럼) vs StyleGAN (NLL 없음, sample SOTA).

이유: NLL 은 average log-likelihood 이므로 **mode coverage** 에 민감하지만 **per-sample quality** 에 둔감. FID 는 반대.

**시사점**: explicit/implicit 의 선택은 학술적이 아니라 **사용 사례에 의존** — anomaly detection 이면 explicit, image generation 이면 implicit/diffusion.

### 2. Diffusion 이 양쪽의 장점을 갖는다

Diffusion 은 bounded explicit (ELBO) + score-based (implicit-style training) 의 hybrid:
- ELBO 로 likelihood 평가 가능
- Score matching 으로 implicit-style sample 생성
- FID 와 NLL 모두 SOTA 근접

이것이 2020 년 이후 diffusion 이 모든 family 를 압도한 이유. 단, sampling 속도가 GAN/Flow 보다 느리다는 단점.

### 3. Implicit 의 평가 어려움

GAN 시대의 큰 문제: "어느 GAN 이 더 좋은가" 를 객관적으로 판단 어려움. FID 도 Inception 모델 의존, IS 는 mode coverage 무시. Precision/Recall (Kynkäänniemi 2019), KID (Bińkowski 2018) 등이 보완. Diffusion 으로 넘어가면서 **likelihood 와 perceptual 를 동시에** 평가 가능해짐.

---

## ⚖️ 가정과 한계

| 가정 | 한계 |
|------|------|
| Tractable likelihood = 좋은 모델 | NLL 과 sample quality 가 약한 상관관계 (Theis 2016) |
| Implicit 는 평가 불가 | FID, IS, Precision/Recall 로 부분적 평가 가능 |
| Family 가 명확히 구분됨 | Diffusion 은 hybrid (bounded + score-based) |
| Lower bound 는 항상 useful | ELBO gap 이 클 때 misleading (mode-seeking $q$) |
| Architecture 가 family 결정 | Hybrid (VAE + Flow, Diffusion + GAN) 가능 |

---

## 📌 핵심 정리

$$\boxed{\text{Tractable Explicit (AR, Flow): } \log p_\theta(x) \text{ closed-form}}$$

$$\boxed{\text{Bounded Explicit (VAE, Diffusion): } \log p_\theta(x) \geq \mathcal{L}(\theta, \phi; x)}$$

$$\boxed{\text{Implicit (GAN, EBM): } \log p_\theta(x) \text{ unavailable, sampling/score-only}}$$

| Family | Likelihood | 훈련 | 평가 |
|--------|------------|------|------|
| **AR** | Exact $\prod p(x_i\|x_{<i})$ | MLE 직접 | NLL · FID |
| **Flow** | Exact via change-of-variables | MLE 직접 | NLL · FID |
| **VAE** | ELBO (lower bound) | ELBO 최대화 | ELBO · FID · IWAE |
| **Diffusion** | ELBO (Ho 2020 분해) | $L_\text{simple}$ | NLL · FID |
| **GAN** | None | Adversarial minimax | FID · IS · P/R |
| **EBM** | Unnormalized $e^{-E}/Z$ | Contrastive · score | Energy distribution |

---

## 🤔 생각해볼 문제

**문제 1** (기초): VAE 와 Flow 모두 latent variable model 이지만 한 쪽은 bounded, 다른 쪽은 exact 이다. 차이를 architecture 측면에서 한 문장으로 설명하라.

<details>
<summary>해설</summary>

VAE 의 decoder $p_\theta(x|z)$ 는 **stochastic** (예: Gaussian noise 추가) 이고 **non-invertible** (NN $z \to x$ 가 1-to-1 이 아님), 따라서 marginal $\int p(x|z) p(z) dz$ 가 intractable 이고 ELBO 로 우회. Flow 의 $f_\theta: z \to x$ 는 **deterministic + invertible**, 따라서 change of variables 로 $p(x)$ exact 계산.

핵심 차이: **stochastic decoder (expressiveness)** vs **invertible decoder (exact likelihood)** 의 trade-off.

</details>

**문제 2** (심화): GAN 의 generator 가 dimension-preserving invertible (e.g., Glow-style) 이라면 push-forward measure 의 Lebesgue density 가 잘 정의되는가? 이때 GAN 을 explicit 으로 만들 수 있는가?

<details>
<summary>해설</summary>

**예** — invertible $G_\theta: \mathbb{R}^d \to \mathbb{R}^d$ 이면 change of variables 로

$$p_\theta(x) = p_z(G_\theta^{-1}(x)) \cdot |\det J_{G_\theta^{-1}}(x)|$$

가 잘 정의됨. 이 경우 GAN 을 Flow 로 재해석한 것이며, **MLE 가능**해짐. 실제로 **Flow + adversarial loss** 의 hybrid (Adversarial Flows, Grover 2018) 가 존재합니다.

단, 표준 GAN ($\mathbb{R}^k \to \mathbb{R}^d, k < d$) 은 manifold 로의 mapping 이라 density 가 singular, MLE 불가능. Implicit 의 본질은 **dim mismatch** 에 있는 셈.

</details>

**문제 3** (논문 비평): Theis et al. 2016 "A note on the evaluation of generative models" 의 핵심 주장 — "낮은 NLL 이 좋은 sample quality 를 함의하지 않는다" — 을 두 가지 극단적 예로 설명하라.

<details>
<summary>해설</summary>

**예 1: 좋은 NLL, 나쁜 sample**: PixelCNN 같은 AR 모델이 데이터 분포의 모든 mode 를 cover 하면 NLL 은 좋지만, 각 sample 이 high-frequency noise (예: blurry 또는 grainy) 를 포함할 수 있음. 평균 likelihood 는 좋지만 perceptual quality 는 나쁨.

**예 2: 나쁜 NLL, 좋은 sample**: Mode collapse 된 GAN 이 데이터의 일부 mode 만 정밀하게 생성할 때, 그 sample 들은 sharp 하지만 데이터 분포 전체의 NLL 은 매우 나쁨 (cover 하지 못한 mode 에서 likelihood 0).

**일반 원리**: NLL 은 $\mathbb{E}_{p_d}[\log p_\theta]$ — 데이터 분포 위의 평균. 한 mode 가 매우 well-modeled 되면 그 mode 에서 큰 contribution. Sample quality 는 generator 의 output 분포에 의존 — 다른 척도.

**시사점**: Likelihood 와 sample quality 는 **다른 차원의 평가**, 둘 다 보고해야 함. Diffusion 은 둘 다 잘 함.

</details>

---

<div align="center">

[◀ 이전 (01. Generative vs Discriminative)](./01-generative-vs-discriminative.md) | [📚 README](../README.md) | [다음 ▶ (03. KL 최소화 통합)](./03-kl-minimization-unification.md)

</div>
