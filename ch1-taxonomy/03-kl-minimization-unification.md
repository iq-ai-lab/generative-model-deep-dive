# 03. 통합 목표 — $\min_\theta \text{KL}(p_\text{data} \| p_\theta)$

## 🎯 핵심 질문

- MLE 와 $\min \text{KL}(p_\text{data} \| p_\theta)$ 가 정확히 동치임을 어떻게 증명하는가? 두 표현의 차이가 단순히 "상수" 라는 주장의 의미는?
- AR · VAE · Flow · GAN · Diffusion 의 **다른 훈련 objective** 들이 모두 어떻게 KL 최소화의 변형으로 환원되는가?
- Forward KL ($p_\text{data} \| p_\theta$) 과 Reverse KL ($p_\theta \| p_\text{data}$) 은 어떤 다른 동작을 만드는가? VAE 의 inference KL 은 어느 쪽인가?
- JSD 와 Wasserstein 은 KL 의 어떤 한계를 보완하는가? Support 가 겹치지 않을 때 KL 이 발산하는 문제는?
- "모든 생성 모델이 divergence 최소화이다" 라는 통합된 관점이 왜 실전적으로 유용한가?

---

## 🔍 왜 통합이 결정적인가

생성 모델 자료를 처음 접하면 5가지 모델이 5가지 완전히 다른 objective 를 사용하는 것처럼 보입니다:

- AR: $-\sum_i \log p(x_i | x_{<i})$
- VAE: $-\text{ELBO} = -\mathbb{E}_q[\log p(x|z)] + \text{KL}(q \| p(z))$
- Flow: $-\log p(z) - \log|\det J|$
- GAN: $\min_G \max_D V(D, G)$
- Diffusion: $\mathbb{E}_{t, x_0, \epsilon}[\|\epsilon - \epsilon_\theta(x_t, t)\|^2]$

하지만 이들은 모두 **KL divergence (또는 그 변형) 의 최소화** 입니다. 이를 통합적으로 이해하면:

1. **각 모델이 "어떤 근사" 를 하는지 명확** — exact KL (AR), lower bound KL (VAE), exact KL via change-of-variables (Flow), JSD ≈ symmetrized KL (GAN), weighted KL (Diffusion)
2. **모델 간 비교 가능** — 같은 metric 으로 평가
3. **새 모델 설계의 원리** — 어떤 divergence 를 최소화하는지 정하면 모델이 따라옴
4. **Hybrid 모델 설계** — VAE + Flow, Diffusion + GAN 등을 동일 framework 에서 정당화

이 문서는 **모든 생성 모델을 KL (또는 generalized divergence) 의 다른 근사** 로 보는 통합된 관점을 제공합니다.

---

## 📐 수학적 선행 조건

- [Probability Theory Deep Dive](https://github.com/iq-ai-lab/probability-theory-deep-dive): 기댓값, Law of Large Numbers
- [Information Theory Deep Dive](https://github.com/iq-ai-lab/information-theory-deep-dive): KL · JSD · Wasserstein · f-divergence
- 이전 문서: 02-explicit-vs-implicit.md

---

## 📖 직관적 이해

### "데이터 분포에 가까워지기 위한 다른 길"

생성 모델의 목표는 "모델 분포 $p_\theta$ 가 진짜 분포 $p_\text{data}$ 에 가까워지게 하는 것". "가깝다" 의 측정 방법이 KL, JSD, Wasserstein, score 거리 등 — 이들 사이의 차이가 곧 **모델 family 의 차이**.

| Divergence | 모델 | 특징 |
|------------|------|------|
| Forward KL $\text{KL}(p_d \| p_\theta)$ | MLE 기반 (AR, Flow, VAE, Diffusion) | **Mass-covering**: $p_d$ 가 큰 곳을 모두 cover |
| Reverse KL $\text{KL}(p_\theta \| p_d)$ | VAE 의 inference, Mode-seeking | **Mode-seeking**: $p_d$ 의 일부 mode 만 |
| JSD $\text{JSD}(p_d, p_\theta)$ | GAN | Symmetric, support 겹치지 않으면 상수 |
| Wasserstein $W(p_d, p_\theta)$ | WGAN | Support 겹치지 않아도 의미 있음 |
| Score difference | NCSN, Diffusion | $\nabla \log p_d$ vs $\nabla \log p_\theta$ |

### Forward vs Reverse KL 의 직관

$p_d$ 가 두 mode (왼쪽 봉우리 + 오른쪽 봉우리), $p_\theta$ 가 one Gaussian:

**Forward KL $\text{KL}(p_d \| p_\theta)$**: $p_d > 0$ 인 곳에서 $p_\theta$ 가 작으면 무한 페널티 → $p_\theta$ 가 두 mode 사이로 평균화 (mass-covering)

**Reverse KL $\text{KL}(p_\theta \| p_d)$**: $p_\theta > 0$ 인 곳에서 $p_d$ 가 작으면 무한 페널티 → $p_\theta$ 가 한 mode 만 정확히 cover (mode-seeking)

**시사점**:
- MLE = forward KL → blurry samples (예: VAE 가 averaged image)
- Reverse KL → mode collapse (예: GAN 의 일부 mode 집중)

### 통합된 관점의 그림

```
                    p_data (진짜 분포)
                         │
        ┌────────────────┼────────────────┐
        │                │                │
   AR (forward KL)  VAE (KL via ELBO)  Flow (forward KL via change-of-vars)
   직접 NLL          하한 최대화           정확 NLL
        │                │                │
        └────────────────┼────────────────┘
                         │
                         ▼
                   p_θ (모델)
                         ▲
        ┌────────────────┼────────────────┐
        │                │                │
   GAN (JSD)      Diffusion (weighted KL)  WGAN (Wasserstein)
   adversarial    score matching            Lipschitz
```

---

## ✏️ 엄밀한 정의·정리

### 정리 3.1 — MLE ≡ KL 최소화

데이터 $\mathcal{D} = \{x_i\}_{i=1}^n \overset{\text{iid}}{\sim} p_\text{data}$ 와 모델 family $\{p_\theta\}$ 에 대해:

$$\hat\theta_{\text{MLE}} = \arg\max_\theta \frac{1}{n}\sum_{i=1}^n \log p_\theta(x_i)$$

은 $n \to \infty$ 일 때 $\arg\min_\theta \text{KL}(p_\text{data} \| p_\theta)$ 와 동치이다.

**증명**: 큰 수의 법칙 (Law of Large Numbers):

$$\frac{1}{n}\sum_i \log p_\theta(x_i) \xrightarrow{n \to \infty} \mathbb{E}_{p_\text{data}}[\log p_\theta(x)]$$

그리고

$$\mathbb{E}_{p_\text{data}}[\log p_\theta(x)] = -H(p_\text{data}) - \text{KL}(p_\text{data} \| p_\theta)$$

여기서 $H(p_\text{data}) = -\mathbb{E}[\log p_\text{data}]$ 는 $\theta$ 와 무관한 상수. 따라서 $\theta$ 에 대한 maximization 은 KL minimization 과 동치. $\square$

### 정의 3.2 — Forward vs Reverse KL

$$\text{KL}(p \| q) = \mathbb{E}_p\left[\log \frac{p}{q}\right]$$

**Forward** (= mass-covering, M-projection): $\text{KL}(p_\text{data} \| p_\theta)$ — $p_\text{data} > 0$ 인 곳에 $p_\theta$ 가 mass 를 가져야 함.

**Reverse** (= mode-seeking, I-projection): $\text{KL}(p_\theta \| p_\text{data})$ — $p_\theta > 0$ 인 곳에 $p_\text{data}$ 가 mass 가 있어야 함.

**관계**: 둘 다 비대칭, 일반적으로 다른 최적해. KL 의 평균을 JSD 라 부름.

### 정의 3.3 — Jensen-Shannon Divergence

$$\text{JSD}(p, q) = \frac{1}{2}\text{KL}(p \| m) + \frac{1}{2}\text{KL}(q \| m), \quad m = \frac{p + q}{2}$$

**성질**:
- Symmetric: $\text{JSD}(p, q) = \text{JSD}(q, p)$
- Bounded: $0 \leq \text{JSD} \leq \log 2$
- $\text{JSD} = 0 \iff p = q$
- 제곱근 $\sqrt{\text{JSD}}$ 는 metric (거리 함수)

### 정의 3.4 — Wasserstein-1 Distance

$$W_1(p, q) = \inf_{\gamma \in \Pi(p, q)} \mathbb{E}_{(x, y) \sim \gamma}[\|x - y\|]$$

여기서 $\Pi(p, q)$ 는 marginals 가 $p, q$ 인 결합 분포의 집합 (couplings).

**핵심 성질**: support 가 겹치지 않아도 **연속**, **미분 가능** (KL/JSD 와 다름).

### 정리 3.5 — 5가지 모델의 KL 환원

각 family 의 훈련 objective 를 KL 또는 그 변형으로 표현하면:

| 모델 | Objective | KL 형태 |
|------|-----------|---------|
| AR | $-\mathbb{E}_{p_d}[\log p_\theta(x)]$ | Forward KL + const |
| Flow | 동일 (change-of-vars 사용) | Forward KL + const |
| VAE | $-\text{ELBO} = -\log p + \text{KL}(q \| p(z\|x))$ | Forward KL + posterior gap |
| Diffusion | $L_\text{simple}$ | Weighted forward KL (Ho 2020) |
| GAN | Minimax → JSD | $\text{JSD}(p_d, p_\theta) + \text{const}$ |
| WGAN | Wasserstein-1 | $W_1(p_d, p_\theta)$ (KL 아님) |
| Score-based | DSM | Implicit KL via score |

### 정리 3.6 — VAE 의 KL 분해

VAE 에서:

$$\mathbb{E}_{p_d}[\log p_\theta(x)] - \text{ELBO} = \mathbb{E}_{p_d}[\text{KL}(q_\phi(z|x) \| p_\theta(z|x))]$$

따라서 ELBO 를 최대화하면 두 가지가 동시에 일어남:

1. $\mathbb{E}_{p_d}[\log p_\theta(x)]$ 가 최대화 → forward KL minimization
2. $\mathbb{E}_{p_d}[\text{KL}(q_\phi \| p_\theta(z|x))]$ 가 최소화 → posterior 가 amortized inference 와 일치

이 이중 목적이 VAE 의 unique 한 특성. (자세한 분해는 Ch3-01)

---

## 🔬 증명 및 수학적 유도

### 유도 1 — 각 family 가 KL 을 어떻게 근사하는가

**AR**: $\log p_\theta(x) = \sum_i \log p_\theta(x_i | x_{<i})$ — exact, 따라서 $-\mathbb{E}_{p_d}[\log p_\theta] = H(p_d) + \text{KL}(p_d \| p_\theta)$ 의 정확한 추정 (LLN).

**Flow**: $\log p_\theta(x) = \log p(f^{-1}(x)) + \log|\det J_{f^{-1}}(x)|$ — exact, AR 과 동일.

**VAE**: $\log p_\theta(x) \geq \text{ELBO}$, gap = posterior KL. 따라서 ELBO 를 최대화하면 forward KL 을 **하한을 통해** 최소화.

**Diffusion**: ELBO 로부터 $L_\text{simple}$ 가 유도되며 (Ho 2020), 이는 weighted forward KL 형태:
$$L_\text{simple} \approx \mathbb{E}_t \mathbb{E}_{p_d, q}[w(t) \cdot \text{KL}(q(x_{t-1} | x_t, x_0) \| p_\theta(x_{t-1} | x_t))]$$

**GAN**: 최적 $D$ 에서 $V(D^*, G) = 2 \text{JSD}(p_d \| p_\theta) - \log 4$ (Ch5-02). 따라서 GAN minimax 는 JSD 최소화.

**WGAN**: Wasserstein-1 직접 추정 — KL 군이 아닌 transport-based metric.

### 유도 2 — Forward KL 의 Mass-Covering 성질

$\text{KL}(p_d \| p_\theta) = \int p_d \log(p_d / p_\theta) dx$. $p_d(x) > 0$ 이지만 $p_\theta(x) = 0$ 이면 적분 발산. 따라서 $p_\theta$ 는 $p_d$ 의 support 를 모두 cover 해야 finite KL.

**시사점**: AR · VAE · Flow · Diffusion 은 모두 mode coverage 를 강하게 강제. 이것이 "blurry samples" 의 원인 — 두 mode 사이를 cover 하기 위해 평균화.

### 유도 3 — Reverse KL 의 Mode-Seeking 성질

$\text{KL}(p_\theta \| p_d) = \int p_\theta \log(p_\theta / p_d) dx$. $p_\theta(x) > 0$ 이지만 $p_d(x) = 0$ 이면 발산. 따라서 $p_\theta$ 는 $p_d$ 의 support **안에만** 있어야 함 — 일부 mode 만 선택.

**예시**: GAN 의 mode collapse 는 reverse KL 적 행동의 극단적 경우. Mode 하나만 정확히 모델링하면 reverse KL 작아짐.

### 유도 4 — JSD 가 Support Mismatch 에 둔감한 이유

Disjoint support 인 $p, q$ ($p \cdot q = 0$ everywhere) 에 대해:

$$\text{KL}(p \| q) = \infty, \quad \text{KL}(q \| p) = \infty$$

하지만 $m = (p+q)/2$ 에서:

$$\text{KL}(p \| m) = \int p \log \frac{p}{(p+q)/2} = \int p \log 2 = \log 2$$

(왜냐하면 $p > 0$ 인 곳에 $q = 0$, 따라서 $m = p/2$, 비율 = 2)

마찬가지로 $\text{KL}(q \| m) = \log 2$. 그러면

$$\text{JSD}(p, q) = \frac{1}{2}\log 2 + \frac{1}{2}\log 2 = \log 2$$

**상수**! 따라서 disjoint support 에서 JSD 는 정보가 없음 → gradient 0 → GAN 훈련 실패. 이것이 WGAN (Wasserstein) 의 동기 (Ch5-04).

### 유도 5 — Diffusion 의 Weighted KL 환원

DDPM ELBO:

$$\text{ELBO} = \mathbb{E}_q[\log p(x_T)] + \sum_t \mathbb{E}_q[\log p_\theta(x_{t-1} | x_t) - \log q(x_t | x_{t-1})]$$

각 $t$ 항을 정리하면:

$$L_t = \mathbb{E}_q[\text{KL}(q(x_{t-1} | x_t, x_0) \| p_\theta(x_{t-1} | x_t))]$$

(자세한 유도는 Ch6-02). 이는 **각 noise level 에서 forward KL** 의 가중 합.

따라서 Diffusion 은 본질적으로 forward KL 최소화 (mass-covering), 단지 직접이 아니라 **chain 을 따라 분해**하여 안정적으로.

---

## 💻 실험 재현 / PyTorch 구현

### 실험 1 — Forward vs Reverse KL 의 mode-covering vs mode-seeking 시각화

```python
import torch
import numpy as np
import matplotlib.pyplot as plt

# Target: bimodal mixture
def p_data(x):
    return 0.5 * torch.exp(-0.5 * (x + 2)**2) / np.sqrt(2*np.pi) \
         + 0.5 * torch.exp(-0.5 * (x - 2)**2) / np.sqrt(2*np.pi)

# Model: single Gaussian (제약된 family)
class GaussianModel:
    def __init__(self, mu=0.0, log_sigma=0.0):
        self.mu = torch.tensor(mu, requires_grad=True)
        self.log_sigma = torch.tensor(log_sigma, requires_grad=True)
    def log_p(self, x):
        sigma = self.log_sigma.exp()
        return -0.5 * ((x - self.mu) / sigma)**2 - self.log_sigma - 0.5 * np.log(2*np.pi)
    def sample(self, n):
        return self.mu + self.log_sigma.exp() * torch.randn(n)

# Forward KL: minimize KL(p_data || p_model)
# = -E_{p_data}[log p_model] + const
def forward_kl_loss(model, n=1000):
    # 진짜 데이터 샘플
    x = torch.where(torch.rand(n) > 0.5, torch.randn(n) - 2, torch.randn(n) + 2)
    return -model.log_p(x).mean()

# Reverse KL: minimize KL(p_model || p_data)
# = E_{p_model}[log p_model - log p_data]
def reverse_kl_loss(model, n=1000):
    z = torch.randn(n)
    x = model.mu + model.log_sigma.exp() * z   # reparameterization
    log_p_model = model.log_p(x)
    log_p_d = torch.log(p_data(x) + 1e-10)
    return (log_p_model - log_p_d).mean()

# 두 모델 훈련
m_fwd = GaussianModel()
m_rev = GaussianModel()
opt_fwd = torch.optim.Adam([m_fwd.mu, m_fwd.log_sigma], lr=0.05)
opt_rev = torch.optim.Adam([m_rev.mu, m_rev.log_sigma], lr=0.05)

for step in range(2000):
    opt_fwd.zero_grad(); l_fwd = forward_kl_loss(m_fwd); l_fwd.backward(); opt_fwd.step()
    opt_rev.zero_grad(); l_rev = reverse_kl_loss(m_rev); l_rev.backward(); opt_rev.step()

print(f"Forward KL: μ={m_fwd.mu.item():.2f}, σ={m_fwd.log_sigma.exp().item():.2f}")
# 예상: μ ≈ 0, σ ≈ 2.x (두 mode 를 cover, mass-covering)

print(f"Reverse KL: μ={m_rev.mu.item():.2f}, σ={m_rev.log_sigma.exp().item():.2f}")
# 예상: μ ≈ +2 또는 -2, σ ≈ 1.0 (한 mode 만 선택, mode-seeking)

# Plot: target vs forward-fitted vs reverse-fitted
x_grid = torch.linspace(-6, 6, 200)
plt.plot(x_grid, p_data(x_grid), 'k', label='p_data (bimodal)')
plt.plot(x_grid, m_fwd.log_p(x_grid).exp().detach(), label='Forward KL fit (mass-covering)')
plt.plot(x_grid, m_rev.log_p(x_grid).exp().detach(), label='Reverse KL fit (mode-seeking)')
plt.legend(); plt.title('Forward vs Reverse KL'); plt.show()
```

### 실험 2 — Disjoint Support 에서 KL/JSD 가 발산/포화

```python
# 두 Gaussian 의 거리를 늘리며 KL · JSD · Wasserstein 측정
def compute_distances(mu1, mu2, sigma=0.1, n=10000):
    p = torch.distributions.Normal(mu1, sigma)
    q = torch.distributions.Normal(mu2, sigma)

    # KL via Monte Carlo
    x_p = p.sample((n,))
    kl_pq = (p.log_prob(x_p) - q.log_prob(x_p)).mean().item()

    # JSD via MC
    m_p = (p.log_prob(x_p).exp() + q.log_prob(x_p).exp()) / 2
    kl_pm = (p.log_prob(x_p) - torch.log(m_p + 1e-10)).mean().item()
    x_q = q.sample((n,))
    m_q = (p.log_prob(x_q).exp() + q.log_prob(x_q).exp()) / 2
    kl_qm = (q.log_prob(x_q) - torch.log(m_q + 1e-10)).mean().item()
    jsd = 0.5 * kl_pm + 0.5 * kl_qm

    # Wasserstein-1 (1D 에서 closed-form: |mu1 - mu2|)
    w1 = abs(mu1 - mu2)
    return kl_pq, jsd, w1

distances = np.linspace(0, 10, 20)
kls, jsds, w1s = [], [], []
for d in distances:
    kl, jsd, w1 = compute_distances(0.0, d, sigma=0.5)
    kls.append(kl); jsds.append(jsd); w1s.append(w1)

plt.plot(distances, kls, label='KL')
plt.plot(distances, jsds, label='JSD')
plt.plot(distances, w1s, label='W_1')
plt.xlabel('|mu_1 - mu_2|'); plt.legend(); plt.show()
# 관찰: 거리가 커지면 KL → ∞, JSD → log 2 (포화), W_1 은 선형 증가
# → WGAN 이 GAN 보다 안정적 훈련
```

---

## 🔗 이론과 실전의 간극

### 1. KL 외 metric 의 등장

KL 의 mass-covering 성향이 항상 좋은 것은 아닙니다. Image generation 에서 "blurry" 결과를 만드는 원인. 실전에서는 다른 metric 을 사용하는 경우가 많습니다:

- **WGAN**: Wasserstein-1 — support mismatch 에 robust
- **f-GAN** (Nowozin 2016): Generalized $f$-divergence (KL, JSD, $\chi^2$ 등 통합)
- **MMD** (Gretton 2012): Maximum Mean Discrepancy — kernel-based distance
- **Sinkhorn divergence**: Entropy-regularized Wasserstein

각 metric 은 다른 모델 family 를 motivate.

### 2. Forward KL 의 한계와 mode-collapse 의 trade-off

Forward KL 모델은 mode coverage 가 좋지만 sample sharpness 가 부족 (VAE 의 blurry image). Reverse KL / GAN 은 sharp 하지만 mode collapse 위험. **Diffusion** 이 SOTA 인 이유 중 하나는 두 성향의 균형 — chain 을 따라 분해된 forward KL 이 sharpness 와 coverage 를 모두 유지.

### 3. Implicit 모델의 KL 추정

GAN 같은 implicit 모델에서 $\text{KL}(p_d \| p_\theta)$ 를 직접 추정하는 방법 — density ratio estimation (Sugiyama 2012). $r(x) = p_d(x) / p_\theta(x)$ 를 NN 으로 학습 → $\text{KL} = \mathbb{E}_{p_d}[\log r(x)]$. GAN 의 discriminator 가 이 ratio 와 관련.

---

## ⚖️ 가정과 한계

| 가정 | 한계 |
|------|------|
| KL 이 항상 좋은 metric | Mode-covering 으로 blurry, support 겹치지 않으면 발산 |
| MLE = 최선의 훈련 | Likelihood 와 sample quality 의 약한 상관 (Theis 2016) |
| Forward / Reverse KL 만 고려 | f-divergence, IPM, Sinkhorn 등 일반화 가능 |
| 모든 family 가 KL 최소화 | WGAN 은 Wasserstein, MMD-GAN 은 MMD — KL 군 아님 |
| 통합 관점이 항상 유용 | 각 모델의 algorithmic detail 을 가릴 수 있음 |

---

## 📌 핵심 정리

$$\boxed{\hat\theta_{\text{MLE}} = \arg\max_\theta \mathbb{E}_{p_\text{data}}[\log p_\theta(x)] \equiv \arg\min_\theta \text{KL}(p_\text{data} \| p_\theta)}$$

$$\boxed{\text{Forward KL: mass-covering · Reverse KL: mode-seeking · JSD: symmetric · } W_1: \text{transport-based}}$$

| 모델 | 최소화 대상 | 비고 |
|------|------------|------|
| **AR** | Forward KL (exact) | 직접 NLL |
| **Flow** | Forward KL (exact) | Change-of-variables |
| **VAE** | Forward KL (lower bound) | ELBO, posterior gap |
| **Diffusion** | Weighted forward KL | Chain 분해, $L_\text{simple}$ |
| **GAN** | JSD | Adversarial minimax |
| **WGAN** | Wasserstein-1 | 1-Lipschitz constraint |
| **NCSN** | Score (Fisher divergence) | $\nabla \log p$ matching |

---

## 🤔 생각해볼 문제

**문제 1** (기초): MLE 가 KL 최소화와 동치임을 증명할 때, 왜 $H(p_\text{data})$ 가 "$\theta$ 와 무관한 상수" 인가? Cross-entropy loss 가 같은 형태인 이유는?

<details>
<summary>해설</summary>

$H(p_\text{data}) = -\mathbb{E}_{p_\text{data}}[\log p_\text{data}(x)]$ — 진짜 데이터 분포의 entropy. $\theta$ 는 모델 분포 $p_\theta$ 만 매개변수화하므로 $p_\text{data}$ (그리고 그 entropy) 에 의존하지 않음. 따라서 $\theta$ 에 대한 minimization 에서 상수.

**Cross-entropy**: $H(p, q) = -\mathbb{E}_p[\log q]$. 데이터에 대한 cross-entropy = $-\mathbb{E}_{p_d}[\log p_\theta] = H(p_d) + \text{KL}(p_d \| p_\theta)$. 따라서 cross-entropy minimization = MLE = forward KL minimization. NN 분류기 훈련의 기반.

</details>

**문제 2** (심화): VAE 의 ELBO 를 최대화하는 것이 forward KL $\text{KL}(p_d \| p_\theta)$ 의 lower bound 를 최대화함을 증명하라. 등호 조건은?

<details>
<summary>해설</summary>

ELBO $\mathcal{L}(\theta, \phi; x) \leq \log p_\theta(x)$.

$$\mathbb{E}_{p_d}[\mathcal{L}(\theta, \phi; x)] \leq \mathbb{E}_{p_d}[\log p_\theta(x)] = -H(p_d) - \text{KL}(p_d \| p_\theta)$$

좌변을 $\theta, \phi$ 에 대해 maximize 하면 우변의 lower bound 를 maximize 하는 것 → $\text{KL}(p_d \| p_\theta)$ 의 upper bound 를 minimize 하는 것 (mass-covering).

**등호 조건**: $\mathcal{L} = \log p$ ⟺ $\text{KL}(q_\phi(z|x) \| p_\theta(z|x)) = 0$ ⟺ amortized posterior $q_\phi$ 가 true posterior 와 일치. 일반적으로 NN 으로는 정확히 도달 불가, 따라서 항상 gap 존재 (= ELBO 의 looseness).

</details>

**문제 3** (논문 비평): GAN 의 objective 가 JSD 최소화로 환원되지만, 실전에서는 $-\log D$ (non-saturating) loss 를 사용하면 JSD 와 다른 objective 가 됨. 두 loss 의 차이를 KL 변형으로 표현하라 (Goodfellow 2014 Section 3 참조).

<details>
<summary>해설</summary>

**Saturating loss** (원래): $\min_G \mathbb{E}_{p_z}[\log(1 - D(G(z)))]$ → 최적 $D$ 에서 $V = 2 \text{JSD}(p_d \| p_g) - \log 4$.

**Non-saturating loss** (실전): $\min_G \mathbb{E}_{p_z}[-\log D(G(z))]$ — 동일한 $G$ optimum 이지만 gradient 가 다름.

Goodfellow 2014 Section 3 의 분석: non-saturating loss 가 사실은

$$\text{KL}(p_g \| p_d) - 2 \text{JSD}(p_d \| p_g) + \text{const}$$

형태를 minimize. 즉 **reverse KL + JSD** 의 조합. Reverse KL 부분이 mode-seeking 성향 → mode collapse 의 수학적 원인 중 하나.

**시사점**: 같은 모델의 다른 loss 가 다른 divergence 를 최소화 → 다른 동작. 이론과 실전 간 미묘한 차이.

</details>

---

<div align="center">

[◀ 이전 (02. Explicit vs Implicit)](./02-explicit-vs-implicit.md) | [📚 README](../README.md) | [다음 ▶ (04. Evaluation Metrics)](./04-evaluation-metrics.md)

</div>
