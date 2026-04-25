# 02. Reparameterization Trick

## 🎯 핵심 질문

- $\nabla_\phi \mathbb{E}_{q_\phi(z|x)}[f(z)]$ 를 계산하는데 왜 단순한 chain rule 이 작동하지 않는가? Stochastic node 의 gradient 가 왜 문제인가?
- Path-wise gradient (reparameterization) 와 score function (REINFORCE) gradient 의 정확한 차이는? Variance 가 왜 다른가?
- $z = \mu + \sigma \odot \epsilon, \epsilon \sim \mathcal{N}(0, I)$ 가 unbiased 한가? 왜 다른 분포에서도 작동하는가?
- Discrete latent 의 reparameterization — Gumbel-Softmax 가 어떻게 categorical 을 continuous relaxation 하는가?
- 실전에서 reparameterization 의 variance 감소가 어느 정도인가? REINFORCE 의 baseline 대비?

---

## 🔍 왜 Reparameterization Trick 이 결정적인가

VAE 의 ELBO 는

$$\mathcal{L}(\theta, \phi; x) = \mathbb{E}_{q_\phi(z|x)}[\log p_\theta(x|z)] - \text{KL}(q_\phi(z|x) \| p(z))$$

KL 항은 closed-form (Gaussian 가정) 이지만, **첫 항** $\mathbb{E}_{q_\phi}[\log p_\theta(x|z)]$ 의 $\phi$ 에 대한 gradient 가 어렵습니다. 이유: $q_\phi$ 의 parameter 가 expectation 의 분포에 들어감.

순진한 접근 (REINFORCE / score function) 은 high variance — 훈련 불안정. **Reparameterization trick** 은:

1. **$z \sim q_\phi$ 를 $z = g_\phi(x, \epsilon), \epsilon \sim p(\epsilon)$ 로 표현**
2. **Expectation 을 noise distribution 으로 옮김**: $\mathbb{E}_{q_\phi}[f(z)] = \mathbb{E}_\epsilon[f(g_\phi(x, \epsilon))]$
3. **Gradient 가 $\phi$ 까지 통과** — backprop 가능

이 단순한 trick 이 VAE 를 가능케 만들었고, 이후 모든 variational deep learning 의 표준이 됨.

---

## 📐 수학적 선행 조건

- 이전 문서: 01-elbo-derivation.md
- [Probability Theory Deep Dive](https://github.com/iq-ai-lab/probability-theory-deep-dive): Change of variables, expectation
- [Optimization Theory Deep Dive](https://github.com/iq-ai-lab/optimization-theory-deep-dive): Variance reduction, Monte Carlo

---

## 📖 직관적 이해

### "Stochastic node 의 gradient 문제"

NN 의 forward pass:

$$x \to \mu = f_\phi(x) \to z \sim \mathcal{N}(\mu, I) \to \log p(x|z)$$

문제: $z$ 가 sample — $\mu$ 에서 어떻게 gradient 가 흐르는가?

**Naive 접근 (잘못됨)**: $z$ 를 $\mu$ 로 취급하고 backprop. 그러면 stochasticity 가 사라짐.

**REINFORCE (correct but high variance)**: $\nabla_\phi \mathbb{E}_q[f(z)] = \mathbb{E}_q[f(z) \cdot \nabla_\phi \log q_\phi(z)]$. 작동하지만 variance 가 큼.

**Reparameterization (low variance)**: $z = \mu + \epsilon, \epsilon \sim \mathcal{N}(0, 1)$. 이제 $z$ 가 $\mu$ 의 deterministic function (random source $\epsilon$ 분리). Gradient 가 $\mu \to z$ 로 통과.

### "그림으로 본 reparameterization"

**이전**:
```
       x → μ → [z ~ N(μ, σ)] → loss
                  ↑
              stochastic, gradient 막힘
```

**이후**:
```
   ε ~ N(0, 1)
            ↓
   x → μ → z = μ + σ ε → loss
   x → σ ↗ (deterministic computation)
```

Stochasticity 가 input 으로 옮겨짐, 나머지는 deterministic — gradient 통과.

### Path-wise vs Score Function

두 gradient estimator:

**Path-wise** (reparameterization): $\nabla_\phi \mathbb{E}_\epsilon[f(g_\phi(\epsilon))] = \mathbb{E}_\epsilon[\nabla_\phi f(g_\phi(\epsilon))]$ — gradient 가 $f$ 와 $g_\phi$ 모두 통과.

**Score function** (REINFORCE): $\nabla_\phi \mathbb{E}_q[f(z)] = \mathbb{E}_q[f(z) \nabla_\phi \log q_\phi(z)]$ — gradient 가 $\log q$ 만 통과, $f$ 는 black-box.

**Variance 비교**: Path-wise 는 $f$ 의 smoothness 활용 → 작은 variance. Score function 은 $f$ 를 밖에서 weight 만 하므로 high variance.

---

## ✏️ 엄밀한 정의·정리

### 정의 2.1 — Reparameterization

확률 변수 $z \sim q_\phi$ 가 deterministic 함수 $g_\phi$ 와 noise variable $\epsilon \sim p_\epsilon$ 의 형태:

$$z = g_\phi(\epsilon, x), \quad \epsilon \sim p_\epsilon$$

로 표현 가능하면 $q_\phi$ 가 **reparameterizable** 이라 한다.

**예시 (Gaussian)**: $z \sim \mathcal{N}(\mu_\phi(x), \sigma_\phi^2(x) I)$. Reparameterize:

$$z = \mu_\phi(x) + \sigma_\phi(x) \odot \epsilon, \quad \epsilon \sim \mathcal{N}(0, I)$$

### 정리 2.2 — Path-wise Gradient

$q_\phi$ 가 reparameterizable 이면:

$$\nabla_\phi \mathbb{E}_{z \sim q_\phi}[f(z)] = \mathbb{E}_{\epsilon \sim p_\epsilon}[\nabla_\phi f(g_\phi(\epsilon, x))]$$

**증명**:

$$\mathbb{E}_{z \sim q_\phi}[f(z)] = \int f(z) q_\phi(z|x) dz$$

Change of variables $z = g_\phi(\epsilon, x)$:

$$= \int f(g_\phi(\epsilon, x)) p_\epsilon(\epsilon) d\epsilon = \mathbb{E}_\epsilon[f(g_\phi(\epsilon, x))]$$

$\nabla_\phi$: 적분 영역이 $\phi$ 와 무관하므로 적분 안으로 들어감:

$$\nabla_\phi \mathbb{E}_\epsilon[f(g_\phi(\epsilon, x))] = \mathbb{E}_\epsilon[\nabla_\phi f(g_\phi(\epsilon, x))] \quad \square$$

### 정리 2.3 — Score Function (REINFORCE) Gradient

일반적 (reparameterization 불필요):

$$\nabla_\phi \mathbb{E}_{z \sim q_\phi}[f(z)] = \mathbb{E}_{z \sim q_\phi}[f(z) \nabla_\phi \log q_\phi(z|x)]$$

**증명**:

$$\nabla_\phi \int f(z) q_\phi(z|x) dz = \int f(z) \nabla_\phi q_\phi(z|x) dz$$

$\nabla_\phi q_\phi = q_\phi \nabla_\phi \log q_\phi$ (log-derivative trick):

$$= \int f(z) q_\phi(z|x) \nabla_\phi \log q_\phi(z|x) dz = \mathbb{E}_q[f(z) \nabla_\phi \log q_\phi] \quad \square$$

### 정리 2.4 — Variance 비교

특정 가정 하에서 path-wise variance < score function variance.

**직관적 분석**: Score function 의 estimator $\hat g = f(z) \nabla_\phi \log q_\phi$ 의 variance:

$$\text{Var}(\hat g) = \mathbb{E}[(f(z))^2 \|\nabla_\phi \log q_\phi\|^2] - (\mathbb{E}[\hat g])^2$$

$f(z)$ 가 클수록 variance 폭발. 특히 high-dimensional $z$ 와 low-probability region 에서 nightmarish.

Path-wise $\hat g_\text{rep} = \nabla_\phi f(g_\phi(\epsilon))$ 는 chain rule 로 $f$ 의 smoothness 활용. Lipschitz $f$ 면 bounded variance.

**실증**: Rezende 2014 의 measurement 에서 path-wise 가 REINFORCE 대비 **$10^2 \sim 10^3$ 배 낮은 variance**.

### 정의 2.5 — Gumbel-Softmax (Discrete Latent)

Discrete $z \in \{1, \ldots, K\}$ with logits $\pi_\phi$ 의 reparameterization:

$$z = \arg\max_k (\log \pi_k + g_k), \quad g_k \sim \text{Gumbel}(0, 1)$$

Discrete argmax 는 미분 불가 → **soft relaxation**:

$$z_k = \frac{\exp((\log \pi_k + g_k) / \tau)}{\sum_j \exp((\log \pi_j + g_j) / \tau)}$$

Temperature $\tau \to 0$ 이면 argmax 에 수렴 (discrete), $\tau \to \infty$ 이면 uniform.

훈련 시 $\tau$ 를 anneal — 처음 큰 $\tau$ (continuous, 안정), 점차 작게 (discrete, 정확).

---

## 🔬 증명 및 수학적 유도

### 유도 1 — Gaussian Reparameterization 의 Unbiasedness

**주장**: $\mathbb{E}_\epsilon[f(\mu + \sigma \epsilon)] = \mathbb{E}_{z \sim \mathcal{N}(\mu, \sigma^2)}[f(z)]$.

**증명**: $z = \mu + \sigma \epsilon, \epsilon \sim \mathcal{N}(0, 1)$. Change of variables:

$$p(z) = p_\epsilon(\epsilon) \cdot |d\epsilon / dz| = \frac{1}{\sqrt{2\pi}} e^{-\epsilon^2/2} \cdot \frac{1}{\sigma} = \frac{1}{\sigma \sqrt{2\pi}} e^{-(z-\mu)^2 / 2\sigma^2}$$

= $\mathcal{N}(\mu, \sigma^2)$ density. 따라서 $z$ 는 정확히 $\mathcal{N}(\mu, \sigma^2)$ distribution. 두 expectation 이 동일.

### 유도 2 — 다양한 분포의 Reparameterization

**Gaussian** $\mathcal{N}(\mu, \Sigma)$: $z = \mu + L \epsilon, L L^\top = \Sigma, \epsilon \sim \mathcal{N}(0, I)$.

**Uniform** $\mathcal{U}(a, b)$: $z = a + (b - a) u, u \sim \mathcal{U}(0, 1)$.

**Exponential** $\text{Exp}(\lambda)$: $z = -\log(u) / \lambda, u \sim \mathcal{U}(0, 1)$.

**임의의 cumulative**: $z = F^{-1}(u), u \sim \mathcal{U}(0, 1)$ — inverse CDF method (단, $F^{-1}$ 가 $\phi$-differentiable 해야).

**Bernoulli (이산)**: 직접적 reparameterization 불가능 → Gumbel-Softmax 또는 Concrete.

### 유도 3 — REINFORCE 의 Variance Explosion

Score function gradient 의 variance 를 specific 예로:

$z \sim \mathcal{N}(\mu, 1), f(z) = z^2$. True gradient: $\nabla_\mu \mathbb{E}[z^2] = 2\mu$.

**REINFORCE estimator**: $\hat g = z^2 \cdot (z - \mu)$ ($\nabla_\mu \log q = z - \mu$).

$\mathbb{E}[\hat g] = \mathbb{E}[z^2(z - \mu)]$. Computation: 표현 $z = \mu + \epsilon$ 으로 $\mathbb{E}[(\mu + \epsilon)^2 \epsilon] = \mathbb{E}[2\mu \epsilon^2] = 2\mu$. ✓ unbiased.

$\text{Var}(\hat g) = \mathbb{E}[\hat g^2] - 4\mu^2 = \mathbb{E}[z^4 (z - \mu)^2] - 4\mu^2$. Heavy 6th-order moments → $O(\mu^4)$ 의 variance for large $\mu$.

**Path-wise estimator**: $\hat g = \partial_\mu (g_\phi(\epsilon))^2 = 2(\mu + \epsilon)$. $\mathbb{E}[\hat g] = 2\mu$. ✓
$\text{Var}(\hat g) = 4 \text{Var}(\epsilon) = 4$ — **constant**, $\mu$ 와 무관.

따라서 large $\mu$ 에서 REINFORCE 가 path-wise 대비 $O(\mu^4)$ 배 더 큰 variance.

### 유도 4 — Variance Reduction Techniques (REINFORCE 안에서)

REINFORCE 만 가능한 경우 (discrete, non-reparameterizable):

**Baseline subtraction**: $\hat g = (f(z) - b) \nabla \log q$. $b$ 가 $z$ 와 무관하면 unbiased, 적절한 $b$ 가 variance 감소. Optimal $b$ = $\mathbb{E}[f] / \text{const}$.

**Control variates**: $\hat g_\text{CV} = \hat g - c (\hat h - \mathbb{E}[h])$, $h$ 가 known mean 의 함수. 적절한 $c$ 로 variance 감소.

**Rao-Blackwellization**: 일부 randomness 를 marginalize.

이 기법들이 RL (policy gradient) 에서도 동일하게 적용.

### 유도 5 — Gumbel-Softmax 의 Concrete Distribution

Continuous relaxation 의 정확한 분포 (Maddison 2016, Jang 2016):

$$p(z; \pi, \tau) = \Gamma(K) \tau^{K-1} \prod_{k} \frac{\pi_k z_k^{-\tau - 1}}{\sum_j \pi_j z_j^{-\tau}}$$

$\tau \to 0$: support 가 simplex 의 vertex (one-hot). $\tau \to \infty$: uniform on simplex.

**훈련**:
- Forward: soft $z$ 사용
- Backward: gradient 가 $\pi_\phi, \tau$ 통과
- Annealing: $\tau$ 을 $1.0 \to 0.5$ 등 점차 감소

**Straight-Through estimator** (alternative): forward 는 hard argmax (one-hot), backward 는 soft. Bias 있지만 더 sharp.

---

## 💻 실험 재현 / PyTorch 구현

### 실험 1 — Path-wise vs REINFORCE Variance 측정

```python
import torch

# True problem: minimize E[z^2] where z ~ N(mu, 1), wrt mu
# Optimal: mu* = 0

mu = torch.tensor(2.0, requires_grad=True)
sigma = 1.0

def true_grad(mu):
    return 2 * mu   # 분석적

def reparam_grad(mu, n=100):
    eps = torch.randn(n)
    z = mu + sigma * eps
    f = z.pow(2)
    grad = torch.autograd.grad(f.mean(), mu, create_graph=False)[0]
    return grad.item()

def reinforce_grad(mu, n=100):
    z = mu + sigma * torch.randn(n)
    log_q = -0.5 * ((z - mu) / sigma).pow(2) - 0.5 * torch.log(torch.tensor(2*torch.pi))
    f = z.pow(2)
    grad = torch.autograd.grad((f.detach() * log_q).mean(), mu)[0]
    return grad.item()

# Variance over many seeds
import numpy as np
n_trials = 200
reparam_estimates = [reparam_grad(mu) for _ in range(n_trials)]
reinforce_estimates = [reinforce_grad(mu) for _ in range(n_trials)]

print(f"True gradient:           {true_grad(mu).item():.3f}")
print(f"Reparam — mean: {np.mean(reparam_estimates):.3f}, std: {np.std(reparam_estimates):.3f}")
print(f"REINFORCE — mean: {np.mean(reinforce_estimates):.3f}, std: {np.std(reinforce_estimates):.3f}")
print(f"Variance ratio: {np.var(reinforce_estimates) / np.var(reparam_estimates):.1f}x")
# 일반적으로 reinforce variance 가 reparam 의 100~1000 배
```

### 실험 2 — REINFORCE 의 Baseline 효과

```python
def reinforce_with_baseline(mu, baseline, n=100):
    z = mu + sigma * torch.randn(n)
    log_q = -0.5 * ((z - mu) / sigma).pow(2)
    f = z.pow(2)
    advantage = f.detach() - baseline   # baseline subtraction
    grad = torch.autograd.grad((advantage * log_q).mean(), mu)[0]
    return grad.item()

# baseline = E[f] = mu^2 + sigma^2 (closed-form for Gaussian)
optimal_baseline = mu.item()**2 + sigma**2

estimates = [reinforce_with_baseline(mu, optimal_baseline) for _ in range(200)]
print(f"REINFORCE + baseline — std: {np.std(estimates):.3f}")
# variance 가 reinforce 보다 작아짐 — 그래도 reparam 보다는 큼
```

### 실험 3 — Gumbel-Softmax for Discrete Latent VAE

```python
def gumbel_softmax(logits, tau=1.0, hard=False):
    # G = -log(-log(U))
    g = -torch.log(-torch.log(torch.rand_like(logits) + 1e-10) + 1e-10)
    y = torch.softmax((logits + g) / tau, dim=-1)
    if hard:
        # Straight-through: forward hard, backward soft
        y_hard = torch.zeros_like(y).scatter_(-1, y.argmax(-1, keepdim=True), 1.0)
        y = (y_hard - y).detach() + y
    return y

# Discrete VAE — categorical latent
class CategoricalVAE(nn.Module):
    def __init__(self, x_dim=784, K=20, h_dim=256):
        super().__init__()
        self.K = K
        self.enc = nn.Sequential(
            nn.Linear(x_dim, h_dim), nn.ReLU(),
            nn.Linear(h_dim, K),
        )
        self.dec = nn.Sequential(
            nn.Linear(K, h_dim), nn.ReLU(),
            nn.Linear(h_dim, x_dim),
        )
    def forward(self, x, tau=1.0):
        logits = self.enc(x)
        z = gumbel_softmax(logits, tau)
        x_recon = self.dec(z)
        return x_recon, logits

# 훈련: tau 를 1.0 → 0.5 로 anneal
# KL: KL(q | uniform prior) = sum_k pi_k log(pi_k * K)
```

### 실험 4 — VAE 훈련에서 Reparam vs REINFORCE 비교

```python
# 동일 VAE 구조에서 (a) reparam, (b) REINFORCE 로 ELBO 추정
# Loss curve 와 final NLL 비교

# (a) Reparam: 표준 VAE — 안정적 수렴
# (b) REINFORCE: high variance gradient → 훈련 매우 불안정, 수렴 어려움
# 일반적으로 VAE 는 reparam 없이 학습 거의 불가능 — 이것이 trick 의 결정적 가치
```

---

## 🔗 이론과 실전의 간극

### 1. Discrete Latent 의 어려움

VAE 에서 discrete latent (예: VQ-VAE 의 codebook) 를 위한 방법:
- **Gumbel-Softmax / Concrete**: continuous relaxation, biased but low variance
- **Straight-Through estimator**: identity gradient, biased
- **REINFORCE + baseline**: unbiased but high variance
- **REBAR, RELAX** (Tucker 2017, Grathwohl 2018): combined estimators

VQ-VAE (Ch3-05) 는 Straight-Through estimator + commitment loss 사용.

### 2. Multivariate Gaussian 의 경우

Full covariance $\Sigma$ 의 reparameterization: Cholesky $L$ such that $L L^\top = \Sigma$:

$$z = \mu + L \epsilon, \epsilon \sim \mathcal{N}(0, I)$$

NN 이 $\mu, L$ 을 출력. Diagonal $L$ 로 단순화 → mean-field, 표준 VAE.

Full covariance: 표현력 ↑, computational cost $O(d^2)$. 일반적으로 hierarchical VAE 또는 normalizing flow posterior 가 대안.

### 3. Implicit Reparameterization

$F^{-1}$ 이 closed-form 없는 분포 (e.g., Beta, Dirichlet) 의 경우, **implicit reparameterization** (Figurnov 2018):

$$\frac{\partial z}{\partial \phi} = -\frac{\partial F / \partial \phi}{\partial F / \partial z}$$

$F$ 의 partial derivative 만 필요, $F^{-1}$ 직접 계산 불필요.

---

## ⚖️ 가정과 한계

| 가정 | 한계 |
|------|------|
| $q_\phi$ 가 reparameterizable | Discrete 분포 직접 불가 |
| Closed-form $g_\phi$ | $F^{-1}$ 가 known 인 분포만 |
| Path-wise variance < REINFORCE | 항상은 아님 (특수 $f$ 에서 역전 가능) |
| Gradient estimator 가 unbiased | Straight-through 는 biased |
| Continuous latent | Discrete 는 Gumbel-Softmax 등 우회 |

---

## 📌 핵심 정리

$$\boxed{z = g_\phi(\epsilon, x), \epsilon \sim p_\epsilon \Rightarrow \nabla_\phi \mathbb{E}_q[f(z)] = \mathbb{E}_\epsilon[\nabla_\phi f(g_\phi(\epsilon, x))]}$$

$$\boxed{\text{Gaussian: } z = \mu_\phi(x) + \sigma_\phi(x) \odot \epsilon, \epsilon \sim \mathcal{N}(0, I)}$$

| 방법 | Variance | Bias | 적용 가능 |
|------|----------|------|-----------|
| **Path-wise** (reparameterization) | Low | Unbiased | Reparameterizable family |
| **REINFORCE** (score function) | High | Unbiased | Always (very general) |
| **REINFORCE + baseline** | Medium-high | Unbiased | General + variance reduction |
| **Gumbel-Softmax** | Low | Biased ($\tau > 0$) | Discrete (relaxed) |
| **Straight-Through** | Low | Biased | Discrete (hard forward) |

---

## 🤔 생각해볼 문제

**문제 1** (기초): $z \sim \mathcal{U}(a, b)$ 의 reparameterization 을 derive 하라. Lognormal $z \sim \text{LogNormal}(\mu, \sigma^2)$ 의 경우는?

<details>
<summary>해설</summary>

**Uniform**: $u \sim \mathcal{U}(0, 1)$, $z = a + (b - a) u$. Linear 이므로 trivially reparameterizable.

**LogNormal**: $\log z \sim \mathcal{N}(\mu, \sigma^2)$, 따라서 $z = \exp(\mu + \sigma \epsilon), \epsilon \sim \mathcal{N}(0, 1)$.

확인: $\partial z / \partial \mu = \exp(\mu + \sigma \epsilon) = z$. $\partial z / \partial \sigma = z \cdot \epsilon$. Gradient flow OK.

**일반 원칙**: 분포가 location-scale family $z = \mu + \sigma F^{-1}(u)$ 형태로 표현되면 reparameterizable. Many common distributions (Cauchy, Laplace, etc.) 이 그렇다.

</details>

**문제 2** (심화): REINFORCE estimator $\hat g_R = f(z) \nabla_\phi \log q_\phi(z)$ 의 variance 가 $f(z)$ 의 norm 에 의존. 따라서 **variance reduction 의 핵심은 $f$ 자체를 줄이는 것이 아니라, $f - b$ 의 norm 을 줄이는 것**. Optimal baseline $b^*$ 를 derive 하라.

<details>
<summary>해설</summary>

Variance:

$$\text{Var}(\hat g_R) = \mathbb{E}[(f(z) - b)^2 \|\nabla \log q\|^2] - (\mathbb{E}[\hat g_R])^2$$

(Baseline 이 unbiased gradient 보존 — $\mathbb{E}[b \nabla \log q] = b \cdot 0 = 0$).

$b$ 에 대해 미분:

$$\frac{\partial}{\partial b} \mathbb{E}[(f - b)^2 \|\nabla \log q\|^2] = -2 \mathbb{E}[(f - b) \|\nabla \log q\|^2]$$

= 0 ⟹ $b^* = \frac{\mathbb{E}[f \cdot \|\nabla \log q\|^2]}{\mathbb{E}[\|\nabla \log q\|^2]}$.

**해석**: optimal baseline 은 $\|\nabla \log q\|^2$-weighted average of $f$ — 단순히 $\mathbb{E}[f]$ 가 아님.

**실전 근사**: $b \approx \mathbb{E}[f]$ (running average), 또는 NN baseline (state-dependent in RL). Actor-Critic 의 critic 이 이 baseline 의 NN approximation.

</details>

**문제 3** (논문 비평): Gumbel-Softmax (Jang 2016, Maddison 2016) 와 Straight-Through (Bengio 2013) 의 trade-off 를 비교하라. VQ-VAE 가 후자를 선택한 이유는?

<details>
<summary>해설</summary>

**Gumbel-Softmax**:
- ✅ Continuous relaxation, gradient 정확 통과
- ✅ $\tau$ annealing 으로 점진적으로 discrete 에 가까워짐
- ❌ Forward 시 soft sample → discrete 사용처 (sequence generation) 에서 부정확
- ❌ $\tau$ tuning 이 까다로움

**Straight-Through**:
- ✅ Forward hard (one-hot, exact discrete)
- ✅ Backward 시 identity (gradient bias 단순)
- ❌ Biased gradient — convergence 보장 없음
- ❌ 실전에서는 잘 작동, 이유는 정확하지 않음

**VQ-VAE 의 선택**:
1. **Discrete codebook 의 정확한 사용** — soft 사용 시 codebook 의 "discreteness" 손실
2. **Decoder 가 정확한 codebook entry 만 받음** — Gumbel 의 soft mixture 는 의미 있는 codeword 가 아님
3. **단순함** — STE 가 코드 더 간단

**Trade-off 의 본질**: bias vs faithfulness. Gumbel 은 unbiased-ish (continuous gradient) 이지만 forward 가 fictional, STE 는 forward 정확하지만 biased gradient.

**현대적 동향**: Diffusion + token (e.g., MaskGIT) 으로 discrete generation 을 다른 방식으로. Gumbel-vs-STE 의 직접 비교는 점차 의미가 줄어듦.

</details>

---

<div align="center">

[◀ 이전 (01. ELBO)](./01-elbo-derivation.md) | [📚 README](../README.md) | [다음 ▶ (03. β-VAE)](./03-beta-vae-ib.md)

</div>
