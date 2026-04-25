# 01. VAE 의 유도 · ELBO (Kingma & Welling 2013)

## 🎯 핵심 질문

- Latent variable model $p(x) = \int p(x|z) p(z) dz$ 에서 적분이 왜 intractable 인가? 어떤 조건에서 tractable 해지는가?
- ELBO $\mathcal{L} = \mathbb{E}_{q(z|x)}[\log p(x|z)] - \text{KL}(q(z|x) \| p(z))$ 가 어떻게 정확히 reconstruction + regularization 으로 분해되는가?
- $\log p(x) = \mathcal{L} + \text{KL}(q_\phi(z|x) \| p_\theta(z|x))$ 의 정확한 등식은 무엇이며, gap 의 의미는?
- Amortized inference 에서 encoder $q_\phi(z|x)$ 가 왜 모든 $x$ 에 대해 공유되는가? Variational EM 과 어떻게 다른가?
- Jensen's inequality 와 KL split 의 두 가지 ELBO 유도가 어떻게 동등한 결론에 도달하는가?

---

## 🔍 왜 ELBO 가 결정적인가

VAE (Kingma & Welling 2013) 는 generative model 에 **deep learning + variational inference** 를 결합한 기념비적 작업. 핵심 기여:

1. **ELBO 의 amortized form** — encoder NN 으로 모든 데이터의 posterior 를 amortize, 매 데이터마다 최적화 불필요
2. **Reparameterization trick** — gradient 가 stochastic node 를 통과하게 함 (Ch3-02)
3. **Tractable training** — single SGD with backprop, no MCMC

ELBO 자체는 1990년대 mean-field VI 부터 사용된 lower bound 이지만, **NN 으로 amortize + backprop** 의 조합이 새로움. 이 문서는 ELBO 의 두 가지 유도 (Jensen, KL split), 그 분해의 의미, amortization 의 정당성을 다룹니다.

---

## 📐 수학적 선행 조건

- [Bayesian ML Deep Dive](https://github.com/iq-ai-lab/bayesian-ml-deep-dive): Variational inference, mean-field, amortization
- [Information Theory Deep Dive](https://github.com/iq-ai-lab/information-theory-deep-dive): KL divergence, Jensen's inequality
- [Probability Theory Deep Dive](https://github.com/iq-ai-lab/probability-theory-deep-dive): Marginal, posterior, conditional

---

## 📖 직관적 이해

### "관찰되는 것 vs 숨은 원인"

세상의 데이터 $x$ (이미지, 문장) 는 어떤 **숨은 원인** $z$ (semantic meaning, style) 로부터 생성된다고 가정. 예:

- $x$: 손글씨 숫자 이미지
- $z$: digit 종류 (0-9) + 글씨체 스타일 + 두께

수학적으로:

$$p(x) = \int p(x | z) p(z) dz$$

여기서 $p(z)$ 는 prior (예: standard Gaussian), $p(x | z)$ 는 decoder (예: NN that outputs Gaussian).

**문제**: 이 적분이 일반적으로 intractable. NN 으로 만든 $p(x|z)$ 에 대해 closed-form 적분 불가능.

### "역방향 추론의 어려움"

데이터 $x$ 가 주어졌을 때, 어떤 $z$ 가 생성했나? Posterior $p(z | x)$. Bayes:

$$p(z | x) = \frac{p(x | z) p(z)}{p(x)}$$

분모 $p(x)$ 가 intractable 이므로 posterior 도 intractable. 역방향 추론이 어려움.

### "Variational Approximation 으로 우회"

Intractable $p(z|x)$ 를 tractable family $q_\phi(z|x)$ (예: factored Gaussian) 로 approximate:

$$q_\phi(z|x) \approx p(z|x)$$

VAE 는 $q_\phi$ 를 NN encoder 로 표현 → **모든 $x$ 에 대해 같은 NN 가 다른 $\phi$-output 을 줌 — amortization**.

### ELBO 의 두 가지 직관

**유도 1 (Jensen)**: $\log \int = \log \mathbb{E} \geq \mathbb{E} \log$ (Jensen, log concave).

**유도 2 (KL split)**: $\log p(x) = \mathcal{L} + \text{KL}(q \| p(z|x))$. KL $\geq 0$ 이므로 $\log p(x) \geq \mathcal{L}$.

두 유도가 동일한 ELBO 를 줌 — 중요한 통찰.

---

## ✏️ 엄밀한 정의·정리

### 정의 1.1 — Latent Variable Model

확률 모델

$$p_\theta(x, z) = p_\theta(x | z) p(z)$$

여기서 $z \in \mathbb{R}^k$ 는 latent variable, $p(z)$ 는 prior (예: $\mathcal{N}(0, I)$), $p_\theta(x | z)$ 는 conditional likelihood (decoder).

**Marginal**: $p_\theta(x) = \int p_\theta(x | z) p(z) dz$.

### 정의 1.2 — ELBO (Evidence Lower BOund)

Variational distribution $q_\phi(z | x)$ 가 주어졌을 때:

$$\mathcal{L}(\theta, \phi; x) := \mathbb{E}_{q_\phi(z|x)}\left[\log \frac{p_\theta(x, z)}{q_\phi(z|x)}\right]$$

### 정리 1.3 — ELBO 가 Lower Bound (Jensen 유도)

$$\log p_\theta(x) \geq \mathcal{L}(\theta, \phi; x)$$

**증명** (Jensen):

$$\log p_\theta(x) = \log \int p_\theta(x, z) dz = \log \int q_\phi(z|x) \frac{p_\theta(x, z)}{q_\phi(z|x)} dz$$

$$= \log \mathbb{E}_{q_\phi(z|x)}\left[\frac{p_\theta(x, z)}{q_\phi(z|x)}\right]$$

$\log$ 가 concave 이므로 Jensen's inequality:

$$\geq \mathbb{E}_{q_\phi(z|x)}\left[\log \frac{p_\theta(x, z)}{q_\phi(z|x)}\right] = \mathcal{L} \quad \square$$

### 정리 1.4 — ELBO 와 Posterior KL (KL split 유도)

$$\log p_\theta(x) = \mathcal{L}(\theta, \phi; x) + \text{KL}(q_\phi(z|x) \| p_\theta(z|x))$$

**증명**:

$$\text{KL}(q_\phi(z|x) \| p_\theta(z|x)) = \mathbb{E}_{q_\phi}\left[\log \frac{q_\phi(z|x)}{p_\theta(z|x)}\right]$$

Bayes: $p_\theta(z|x) = p_\theta(x, z) / p_\theta(x)$. 대입:

$$= \mathbb{E}_{q_\phi}\left[\log q_\phi(z|x) - \log p_\theta(x, z) + \log p_\theta(x)\right]$$

$$= -\mathcal{L} + \log p_\theta(x)$$

따라서 $\log p_\theta(x) = \mathcal{L} + \text{KL}$. $\text{KL} \geq 0$ 이므로 $\log p_\theta(x) \geq \mathcal{L}$, 등호는 $q_\phi = p_\theta(z|x)$ 일 때. $\square$

### 정리 1.5 — ELBO 의 Reconstruction + Regularization 분해

$$\mathcal{L}(\theta, \phi; x) = \underbrace{\mathbb{E}_{q_\phi(z|x)}[\log p_\theta(x | z)]}_{\text{Reconstruction}} - \underbrace{\text{KL}(q_\phi(z|x) \| p(z))}_{\text{Regularization}}$$

**증명**:

$$\mathcal{L} = \mathbb{E}_{q_\phi}\left[\log \frac{p_\theta(x, z)}{q_\phi(z|x)}\right] = \mathbb{E}_{q_\phi}\left[\log \frac{p_\theta(x|z) p(z)}{q_\phi(z|x)}\right]$$

$$= \mathbb{E}_{q_\phi}[\log p_\theta(x|z)] + \mathbb{E}_{q_\phi}[\log p(z)] - \mathbb{E}_{q_\phi}[\log q_\phi(z|x)]$$

마지막 두 항 = $-\text{KL}(q_\phi(z|x) \| p(z))$. $\square$

### 정의 1.6 — Amortized Inference

표준 VI: 각 $x_i$ 마다 분리된 variational parameter $\phi_i$ 최적화 — $n$ 데이터마다 $n$ 개 파라미터.

**Amortized**: $\phi_i = \text{Encoder}_\phi(x_i)$ — 모든 $x$ 에 대해 같은 NN, 출력만 다름. 파라미터 $\phi$ 는 모든 데이터 공유.

**장점**:
- Test 시 새 $x$ 의 $z$ 도 forward pass 로 즉시 inference
- Generalization across data points
- Optimization 안정 (single $\phi$, joint training)

### 정리 1.7 — VAE Loss 의 SGD 가능성

$$-\mathcal{L} = -\mathbb{E}_{q_\phi(z|x)}[\log p_\theta(x|z)] + \text{KL}(q_\phi(z|x) \| p(z))$$

$x$ 에 대한 Monte Carlo 추정 (mini-batch SGD), $z$ 에 대한 reparameterization trick 으로 모두 SGD 가능.

---

## 🔬 증명 및 수학적 유도

### 유도 1 — Two ELBO Derivations 의 동등성

**Jensen 유도**: $\log \int q (p/q) \geq \int q \log(p/q)$. ELBO 는 RHS.

**KL split**: $\log p(x) = \int q \log(p \cdot p/p) - \int q \log(q/p(z|x))$. ELBO 는 첫 항.

두 결과의 동등성: 직접 산수.

$$\mathcal{L}_{\text{Jensen}} = \mathbb{E}_q[\log p(x, z) - \log q] = \mathbb{E}_q[\log p(x|z) + \log p(z) - \log q] = \mathbb{E}_q[\log p(x|z)] - \text{KL}(q \| p(z))$$

$$\mathcal{L}_{\text{KL split}} = \log p(x) - \text{KL}(q \| p(z|x))$$

이 둘이 같음을 보이려면:

$$\log p(x) - \text{KL}(q \| p(z|x)) = \log p(x) - \mathbb{E}_q[\log q - \log p(z|x)]$$
$$= \log p(x) - \mathbb{E}_q[\log q] + \mathbb{E}_q[\log p(x, z) - \log p(x)]$$
$$= \log p(x) - \mathbb{E}_q[\log q] + \mathbb{E}_q[\log p(x, z)] - \log p(x)$$
$$= \mathbb{E}_q[\log p(x, z) - \log q] = \mathcal{L}_{\text{Jensen}} \quad \square$$

### 유도 2 — Gaussian VAE 의 Closed-Form KL

VAE 의 일반적 선택: $p(z) = \mathcal{N}(0, I)$, $q_\phi(z|x) = \mathcal{N}(\mu_\phi(x), \text{diag}(\sigma_\phi^2(x)))$.

$$\text{KL}(q_\phi \| p) = \frac{1}{2} \sum_{i=1}^k \left(\mu_i^2 + \sigma_i^2 - 1 - \log \sigma_i^2\right)$$

**증명** (1D 의 경우):

$$\text{KL}(\mathcal{N}(\mu, \sigma^2) \| \mathcal{N}(0, 1)) = \mathbb{E}_{q}\left[\log \frac{q(z)}{p(z)}\right]$$

$$= \mathbb{E}_q[-\log \sigma - 0.5 (z - \mu)^2 / \sigma^2 + 0.5 z^2]$$

$\mathbb{E}_q[(z - \mu)^2] = \sigma^2$, $\mathbb{E}_q[z^2] = \mu^2 + \sigma^2$:

$$= -\log \sigma - 0.5 + 0.5(\mu^2 + \sigma^2) = \frac{1}{2}(\mu^2 + \sigma^2 - 1 - \log \sigma^2)$$

다차원 factored 는 합. $\square$

### 유도 3 — ELBO 의 두 항이 Trade-off

**Reconstruction**: $z$ 가 $x$ 의 정보를 많이 가질수록 큼 (좋음).

**KL Regularization**: $q_\phi(z|x)$ 가 $p(z)$ 와 다를수록 큼 (페널티).

**Trade-off**: 강한 reconstruction 위해 $z$ 가 정보 많이 → $q$ 가 prior 에서 멀어짐 → KL 큼.

**Information-theoretic 해석**: KL = mutual information $I(X; Z)$ 의 upper bound (Alemi 2017). β-VAE (다음 문서) 가 이 trade-off 를 명시적으로 조절.

### 유도 4 — Posterior 와 Encoder 의 일치 조건

**ELBO Tight 조건**: $\text{KL}(q_\phi(z|x) \| p_\theta(z|x)) = 0$ ⟺ $q_\phi(z|x) = p_\theta(z|x)$ everywhere.

**일반적으로 도달 불가**: NN encoder $q_\phi$ 가 factored Gaussian, 진짜 posterior $p_\theta(z|x)$ 는 더 복잡 (multi-modal, correlated).

**해결책**:
- Normalizing flow posterior (Rezende 2015): $q$ 의 family 확장
- IAF posterior (Kingma 2016)
- Hierarchical VAE: multi-layer $z$
- IWAE (Burda 2015): Importance weighted ELBO

### 유도 5 — Amortization Gap

Two gap 분해:
- **Approximation gap**: $\inf_q \text{KL}(q \| p_\theta(z|x))$ — variational family 의 한계
- **Amortization gap**: amortized $q_\phi$ 가 optimal $q^*$ 에 못 미치는 정도

**증거** (Cremer 2018): VAE 의 ELBO loss 의 약 50% 가 amortization gap. 각 데이터 별로 추가 optimization (test-time fine-tuning) 이 ELBO 향상.

---

## 💻 실험 재현 / PyTorch 구현

### 실험 1 — VAE 구현 (MNIST)

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision

class VAE(nn.Module):
    def __init__(self, x_dim=784, z_dim=32, h_dim=400):
        super().__init__()
        # Encoder
        self.enc = nn.Sequential(
            nn.Linear(x_dim, h_dim), nn.ReLU(),
            nn.Linear(h_dim, h_dim), nn.ReLU(),
        )
        self.mu = nn.Linear(h_dim, z_dim)
        self.logvar = nn.Linear(h_dim, z_dim)
        # Decoder
        self.dec = nn.Sequential(
            nn.Linear(z_dim, h_dim), nn.ReLU(),
            nn.Linear(h_dim, h_dim), nn.ReLU(),
            nn.Linear(h_dim, x_dim),
        )

    def encode(self, x):
        h = self.enc(x)
        return self.mu(h), self.logvar(h)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        return self.dec(z)

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        x_recon = self.decode(z)
        return x_recon, mu, logvar

def vae_loss(x, x_recon, mu, logvar):
    # Bernoulli decoder for binary MNIST
    recon = F.binary_cross_entropy_with_logits(x_recon, x, reduction='sum')
    # Closed-form KL for Gaussian q vs N(0,I)
    kl = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return recon, kl

# 훈련
loader = torch.utils.data.DataLoader(
    torchvision.datasets.MNIST('~/data', download=True,
        transform=torchvision.transforms.ToTensor()),
    batch_size=128, shuffle=True
)
model = VAE().cuda()
opt = torch.optim.Adam(model.parameters(), lr=1e-3)

for epoch in range(20):
    total_recon, total_kl = 0, 0
    for x, _ in loader:
        x = x.view(-1, 784).cuda()
        x_recon, mu, logvar = model(x)
        recon, kl = vae_loss(x, x_recon, mu, logvar)
        loss = recon + kl
        opt.zero_grad(); loss.backward(); opt.step()
        total_recon += recon.item(); total_kl += kl.item()
    print(f"Epoch {epoch}: recon = {total_recon:.0f}, KL = {total_kl:.0f}")
```

### 실험 2 — IWAE 로 ELBO 의 Tightness 측정

```python
@torch.no_grad()
def iwae_log_p(model, x, K=100):
    """Importance Weighted ELBO — K → ∞ 이면 exact log p(x)"""
    mu, logvar = model.encode(x)
    std = (0.5 * logvar).exp()
    # K samples per data point
    z = mu.unsqueeze(0) + std.unsqueeze(0) * torch.randn(K, *mu.shape, device=x.device)
    x_recon = model.decode(z)
    # Bernoulli log p(x|z)
    log_p_x_z = -F.binary_cross_entropy_with_logits(
        x_recon, x.unsqueeze(0).expand(K, -1, -1), reduction='none').sum(-1)
    # log p(z) and log q(z|x) — Gaussians
    log_p_z = -0.5 * (z.pow(2).sum(-1) + z.shape[-1] * torch.log(torch.tensor(2*torch.pi)))
    log_q_z = -0.5 * (((z - mu) / std).pow(2).sum(-1)
                     + logvar.sum(-1) + z.shape[-1] * torch.log(torch.tensor(2*torch.pi)))
    log_w = log_p_x_z + log_p_z - log_q_z   # [K, batch]
    # IWAE estimator
    return torch.logsumexp(log_w, 0) - torch.log(torch.tensor(K, dtype=torch.float))

# K=1: ELBO, K=large: closer to true log p
x = next(iter(loader))[0][:64].view(-1, 784).cuda()
elbo = iwae_log_p(model, x, K=1).mean().item()
log_p_approx = iwae_log_p(model, x, K=1000).mean().item()
print(f"ELBO (K=1):       {elbo:.2f}")
print(f"log p (K=1000):   {log_p_approx:.2f}")
print(f"Gap (looseness):  {log_p_approx - elbo:.2f}")
# 일반적으로 0.1 ~ 1 nat/dim gap 관찰
```

### 실험 3 — Latent Space 의 Interpolation

```python
@torch.no_grad()
def latent_interp(model, x1, x2, n=10):
    mu1, _ = model.encode(x1.view(1, -1))
    mu2, _ = model.encode(x2.view(1, -1))
    alphas = torch.linspace(0, 1, n).to(x1.device)
    interp = []
    for a in alphas:
        z = (1 - a) * mu1 + a * mu2
        x_decoded = torch.sigmoid(model.decode(z))
        interp.append(x_decoded)
    return torch.cat(interp, dim=0)

# 두 MNIST 이미지 (예: 3 과 8) 사이 latent interpolation
# → 부드러운 morphing 관찰 — VAE latent 의 의미 있는 manifold
```

---

## 🔗 이론과 실전의 간극

### 1. ELBO 의 Looseness

ELBO 가 lower bound 라는 것은 명확하지만, gap 의 크기는 모델/데이터에 따라 다름. MNIST VAE 의 일반적 gap: 0.1~0.5 nats/dim. 이는:

- **Mode-covering**: 진짜 posterior 의 일부 mode 만 cover
- **Posterior collapse** (Ch3-04): 더 심각한 가능성
- **IWAE, MoG q, Flow posterior** 로 부분 해결

### 2. Reconstruction 의 종류

VAE 의 decoder $p(x|z)$:
- **Gaussian**: $\mathcal{N}(\mu_\theta(z), \sigma^2 I)$ — MSE reconstruction (blurry)
- **Bernoulli**: binary $x$ — BCE
- **Discretized Logistic** (PixelCNN++): natural for image
- **Diffusion as decoder** (Diffusion-VAE): 최신 hybrid

Gaussian decoder 가 blur 의 원인 — VQ-VAE, β-VAE 등 후속 작업이 보완.

### 3. Posterior Collapse 의 위험

Decoder 가 너무 강력 (예: PixelCNN as decoder) 하면 $z$ 무시 가능 → $q(z|x) \approx p(z)$ 로 붕괴. 자세한 내용은 Ch3-04.

---

## ⚖️ 가정과 한계

| 가정 | 한계 |
|------|------|
| Latent variable model | 적분 intractable, ELBO 로 우회 |
| Variational $q_\phi$ family | Mean-field Gaussian 은 진짜 posterior 와 차이 |
| Amortized inference | Approximation + amortization gap |
| Reparameterization 가능 | Continuous latent 만 (discrete 는 Gumbel-Softmax 등) |
| Closed-form KL | Gaussian $q$, $p$ 가정 — 다른 family 에서는 MC |
| Stochastic decoder | Blurry samples — Flow 또는 Diffusion 결합 필요 |

---

## 📌 핵심 정리

$$\boxed{\log p_\theta(x) = \mathcal{L}(\theta, \phi; x) + \text{KL}(q_\phi(z|x) \| p_\theta(z|x))}$$

$$\boxed{\mathcal{L} = \underbrace{\mathbb{E}_{q_\phi}[\log p_\theta(x|z)]}_{\text{Reconstruction}} - \underbrace{\text{KL}(q_\phi(z|x) \| p(z))}_{\text{Regularization}}}$$

| 객체 | 의미 |
|------|------|
| **$p(z)$** | Prior (e.g., $\mathcal{N}(0, I)$) |
| **$p_\theta(x\|z)$** | Decoder (NN) — $z$ 에서 $x$ 의 likelihood |
| **$q_\phi(z\|x)$** | Encoder (NN) — amortized variational posterior |
| **ELBO** | $\log p(x)$ 의 lower bound, 최적화 대상 |
| **Posterior gap** | $\text{KL}(q_\phi \| p_\theta(z\|x))$ — bound 의 looseness |
| **Reparameterization** | $z = \mu + \sigma \odot \epsilon$ 로 gradient 통과 |

---

## 🤔 생각해볼 문제

**문제 1** (기초): $q(z|x) = \mathcal{N}(\mu, \sigma^2)$, $p(z) = \mathcal{N}(0, 1)$ 일 때 closed-form KL = $\frac{1}{2}(\mu^2 + \sigma^2 - 1 - \log \sigma^2)$ 임을 보였다. $\mu = 0, \sigma = 1$ 일 때 KL = 0 임을 확인하고, $\sigma \to 0$ 일 때 KL → ∞ 임의 의미를 직관적으로 설명하라.

<details>
<summary>해설</summary>

$\mu = 0, \sigma = 1$: $\frac{1}{2}(0 + 1 - 1 - \log 1) = 0$. ✓ — encoder 가 prior 와 일치 시 KL = 0.

$\sigma \to 0$: $-\log \sigma^2 \to \infty$ → KL → ∞.

**직관**: $\sigma \to 0$ 은 $q(z|x)$ 가 delta function 으로 수렴 — encoder 가 deterministic mapping. Prior $\mathcal{N}(0, 1)$ 는 분산 있음, delta 와 매우 다름 → 큰 KL.

**시사점**: VAE 가 $\sigma \to 0$ 으로 가는 것을 KL 항이 막음 → encoder 가 stochastic 유지. 너무 deterministic 하면 latent space 에 hole 이 생기고, sampling 시 hole 에서 sampling 하면 reconstruction 실패.

</details>

**문제 2** (심화): ELBO 의 두 유도 (Jensen, KL split) 가 동등한 결과를 줌을 명시적으로 보였다. 두 유도 중 어느 것이 (i) computational implementation 에, (ii) theoretical insight 에 더 적합한가?

<details>
<summary>해설</summary>

**(i) Computational**: Jensen 유도 직접적. $\mathcal{L} = \mathbb{E}_q[\log p(x, z) - \log q(z|x)]$ — Monte Carlo (sample $z \sim q$) 로 추정, 각 항을 NN forward 로 계산. 이 형태가 코드 구현에 직접 매핑.

**(ii) Theoretical**: KL split 이 깊은 통찰. $\log p(x) = \mathcal{L} + \text{KL}(q \| p(z|x))$ — bound 의 **gap 이 정확히 posterior approximation 오차** 임을 보임. 이로부터:
- Bound tight ⟺ $q = p(z|x)$
- VAE 가 사실 posterior 도 학습 (variational E-step)
- Amortization, IWAE, Flow posterior 의 motivation

**결론**:
- **Jensen** 으로 implementation, **KL split** 으로 understanding
- VAE 자료가 이 두 유도를 모두 다루는 이유
- 새 모델 설계 시 KL split 의 관점이 더 generative

</details>

**문제 3** (논문 비평): Cremer et al. 2018 "Inference Suboptimality in Variational Autoencoders" 에서 amortization gap 이 VAE ELBO 의 약 50% 라고 측정했다. 이 발견이 VAE 의 한계와 어떤 관련이 있는가?

<details>
<summary>해설</summary>

**Cremer 2018 의 측정**: ELBO loss $L_{\text{ELBO}}$ 와 per-sample optimal $L^*$ (각 데이터마다 분리된 $\phi^*$ 최적화) 의 비교. **Amortization gap = $L_{\text{ELBO}} - L^*$ ≈ 50% of total gap**.

**시사점**:
1. **Amortization 이 free lunch 가 아님**: 단일 NN 가 모든 데이터의 inference 를 처리 → 평균적 성능이지만 per-sample 은 sub-optimal
2. **Test-time fine-tuning** 이 효과적 — 각 새 $x$ 에 대해 $\phi$ 를 추가 SGD 하면 ELBO 향상
3. **Approximation gap (variational family 의 한계) 도 50%** — Mean-field Gaussian 의 한계

**해결 방향**:
- **More expressive $q$**: Normalizing flow posterior, IAF posterior
- **Hybrid**: amortized + per-sample fine-tuning (semi-amortized VI)
- **IWAE**: K-sample importance weighted, tighter bound

**현대적 관점**: VAE 는 근본적으로 limited bound. Diffusion model 이 chain 의 분해로 더 tight bound (single-step KL 의 합), 따라서 sample quality 우월. 이것이 Diffusion 이 VAE 를 압도한 이유 중 하나.

</details>

---

<div align="center">

[◀ 이전 (Ch2-04. GPT)](../ch2-autoregressive/04-gpt-as-generative.md) | [📚 README](../README.md) | [다음 ▶ (02. Reparameterization)](./02-reparameterization.md)

</div>
