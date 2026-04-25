# 03. β-VAE 와 Information Bottleneck (Higgins 2017)

## 🎯 핵심 질문

- $\mathcal{L}_\beta = \mathbb{E}_q[\log p(x|z)] - \beta \cdot \text{KL}(q \| p(z))$ 의 $\beta$ 가 무엇을 조절하는가? $\beta = 1$ 이 표준 VAE 인 이유는?
- Information Bottleneck (IB) 관점에서 $\beta$ 가 어떻게 mutual information $I(X; Z)$ 의 channel capacity 로 해석되는가?
- $\beta > 1$ 이 왜 disentangled representation 을 만드는가? dSprites, Faces 같은 데이터에서의 실증 결과는?
- Rate-Distortion trade-off (Alemi 2018) 에서 β-VAE 가 곡선의 어느 지점인가?
- $\beta < 1$ 가 sharp samples (high reconstruction) 을 주는데 왜 일반적이지 않은가?

---

## 🔍 왜 β-VAE 가 결정적인가

표준 VAE ($\beta = 1$) 는 latent 가 disentangled 되지 않습니다 — 한 latent dimension 이 여러 factor 를 mix 한 representation. β-VAE (Higgins 2017) 는 단순히 KL 항에 $\beta$ 를 곱하는 한 줄 변경으로 **disentanglement** 을 induce.

핵심 통찰:
- $\beta > 1$: KL 페널티 강화 → encoder 가 prior $\mathcal{N}(0, I)$ 에 더 가까이 → 각 latent dimension 이 **독립** (Gaussian의 axis-aligned)
- 이 압력 + reconstruction loss 가 데이터의 **factor of variation** (예: 모양, 크기, 회전) 을 따로 따로 latent dim 에 매핑

이 단순한 modification 이 representation learning 의 큰 영역을 열어줌:
- **Disentanglement metrics** (β-VAE, FactorVAE, MIG)
- **Information Bottleneck framework** 으로의 통합
- **β annealing** 같은 schedule 기법

이 문서에서는 β-VAE 의 수학과 IB 해석, 그리고 disentanglement 의 정확한 의미를 다룹니다.

---

## 📐 수학적 선행 조건

- 이전 문서: 01-elbo-derivation, 02-reparameterization
- [Information Theory Deep Dive](https://github.com/iq-ai-lab/information-theory-deep-dive): Mutual information, channel capacity, rate-distortion
- [Bayesian ML Deep Dive](https://github.com/iq-ai-lab/bayesian-ml-deep-dive): Variational inference

---

## 📖 직관적 이해

### "Disentanglement: 각 latent 가 다른 factor"

이상적인 generative model: 데이터의 각 **변동 요인** (factor of variation) 이 latent 의 **다른 dimension** 에 매핑.

**예시 (dSprites)**: 64×64 흑백 도형 데이터셋. 5개 factor: (1) shape, (2) scale, (3) rotation, (4) x-position, (5) y-position.

**Disentangled VAE**: 5개 latent dim 이 각각 한 factor 를 encode. $z_1$ 만 바꾸면 shape 만 변화, scale/rotation/position 은 동일.

**Entangled VAE**: 한 latent dim 이 여러 factor 를 mix. $z_1$ 바꾸면 shape + rotation + position 모두 동시 변화.

β-VAE 의 발견: **$\beta > 1$ 로 KL 압력 강화 → disentangled 자연스럽게 발생**.

### KL 압력의 메커니즘

$\text{KL}(q(z|x) \| p(z))$ 가 **각 latent dim 을 prior 와 가깝게** 강제. Prior $\mathcal{N}(0, I)$ 는 axis-aligned Gaussian → 각 dim 이 **독립**.

$\beta > 1$: KL 강화 → encoder 가 latent 를 **더 axis-aligned 분포** 로 표현 → 각 dim 이 데이터의 한 factor 만 capture (가능한 경우).

**문제점**: $\beta$ 너무 크면 reconstruction 희생 → blurry. $\beta$ tuning 이 dataset-dependent.

### Rate-Distortion 관점

Information Theory 의 rate-distortion: 정보량 (rate) 을 줄이면 왜곡 (distortion) 증가.

$$\beta \cdot \text{KL} - \mathbb{E}[\log p(x|z)]: \quad \beta = \text{rate cost}, \text{rec loss} = \text{distortion}$$

β-VAE 는 rate-distortion frontier 의 다른 점들을 explore:
- $\beta \to 0$: 모든 정보 사용 (no rate cost), perfect reconstruction (low distortion)
- $\beta \to \infty$: rate = 0, $z \approx \mathcal{N}(0, I)$ (no info), distortion 최대
- $\beta = 1$: 표준 VAE, ELBO

---

## ✏️ 엄밀한 정의·정리

### 정의 3.1 — β-VAE Objective

$$\mathcal{L}_\beta(\theta, \phi; x) := \mathbb{E}_{q_\phi(z|x)}[\log p_\theta(x|z)] - \beta \cdot \text{KL}(q_\phi(z|x) \| p(z))$$

$\beta = 1$: 표준 ELBO. $\beta > 1$: 강화된 KL 페널티. $\beta < 1$: 약화.

### 정의 3.2 — Disentanglement (informal)

Latent representation $z = (z_1, \ldots, z_d)$ 이 **disentangled** 라 함은: 데이터의 generative factor $f_1, \ldots, f_k$ ($k \leq d$) 에 대해 각 $z_i$ 가 정확히 한 factor 만 capture, 다른 factor 와 independent.

**정량적 측정**:
- **β-VAE metric** (Higgins 2017): pair-wise factor 변경 시 latent 의 변화 패턴
- **MIG** (Chen 2018): Mutual Information Gap
- **FactorVAE score** (Kim & Mnih 2018)
- **DCI** (Eastwood 2018): Disentanglement, Completeness, Informativeness

### 정의 3.3 — Information Bottleneck (Tishby 2015)

확률 변수 $X$ (입력), $Y$ (목표), $Z$ (representation) 에 대해:

$$\min_Z \beta \cdot I(X; Z) - I(Y; Z)$$

$\beta$: rate-distortion trade-off. $\beta$ 작으면 $Z$ 가 $X$ 의 정보 많이 보존, $\beta$ 크면 $Z$ 가 compressed (only $Y$-relevant info).

### 정리 3.4 — β-VAE 와 IB 의 연결 (Alemi 2017)

Unsupervised IB ($Y = X$): minimize $\beta I(X; Z) - I(X; Z) = (\beta - 1) I(X; Z)$? — no, 더 careful 하게.

실제 연결: β-VAE objective 를 expectation under $p_d(x)$ 한 것:

$$\mathcal{L}_\beta = -\beta \cdot R + D$$

where $R = \mathbb{E}_{p_d}[\text{KL}(q(z|x) \| p(z))]$ ≈ upper bound on $I(X; Z)$, $D = \mathbb{E}_{p_d, q}[\log p(x|z)]$ ≈ lower bound on reconstruction quality.

따라서 β-VAE 가 IB 의 upper-lower bound version. $\beta$ 가 직접 channel capacity bound.

### 정리 3.5 — KL 항의 Aggregate Decomposition

$\mathbb{E}_{p_d}[\text{KL}(q(z|x) \| p(z))]$ 의 분해 (Hoffman & Johnson 2016):

$$\mathbb{E}_{p_d}[\text{KL}(q(z|x) \| p(z))] = I(X; Z) + \text{KL}(q(z) \| p(z))$$

여기서 $q(z) = \mathbb{E}_{p_d}[q(z|x)]$ 는 aggregate posterior.

**해석**:
- $I(X; Z)$: $z$ 가 $x$ 정보 얼마나 가짐 (representation 의 informativeness)
- $\text{KL}(q(z) \| p(z))$: aggregate posterior 와 prior 의 mismatch ("hole" 또는 prior misuse)

β > 1 이 두 항 모두 페널티 → less information + closer to prior.

### 정리 3.6 — Disentanglement 의 한계 (Locatello 2019)

**"Challenging Common Assumptions in Unsupervised Learning of Disentangled Representations"**: Pure unsupervised setting 에서 **identifiability 가 보장되지 않음** — 같은 데이터에 대한 다른 disentangled representation 이 모두 valid.

**시사점**:
- β-VAE 의 disentanglement 는 **inductive bias** (axis-aligned prior + factored encoder) 에 의존
- 진짜 disentanglement 위해서는 **추가 supervision** 또는 specific architecture 필요
- 그럼에도 β-VAE 는 useful baseline

---

## 🔬 증명 및 수학적 유도

### 유도 1 — KL 항이 axis-aligned 압력

$q_\phi(z|x) = \mathcal{N}(\mu(x), \text{diag}(\sigma^2(x)))$ (factored Gaussian encoder).

$\text{KL}(q \| p) = \frac{1}{2} \sum_i (\mu_i^2 + \sigma_i^2 - 1 - \log \sigma_i^2)$.

이 KL 이 작아지려면 각 $i$ 에서 $\mu_i \approx 0, \sigma_i \approx 1$ — 즉 $q$ 가 prior 에 가까이.

**KL = 0 도달**: $q = p$, 모든 $x$ 에 대해 같은 분포 → posterior collapse (Ch3-04). 즉 latent 무력화.

**중간**: $\mu_i$ 가 작은 magnitude, 일부 dim 이 0 (KL 작음, info 적음), 일부 dim 이 informative (KL 큼). $\beta > 1$ 로 압력 → 적은 dim 만 active, axis-aligned.

### 유도 2 — Disentanglement 의 발생 조건

데이터 $x$ 가 generative factor $f_1, \ldots, f_k$ 의 함수: $x = G(f_1, \ldots, f_k)$. Factors 가 independent ($p(f) = \prod p(f_j)$).

**Optimal factored encoder**: 각 $z_i$ 가 한 $f_j$ 와 1-to-1 mapping 이면:
- $q(z|x) = \prod \delta(z_i - f_{j(i)})$
- $q(z) = \prod p(f_{j(i)})$ — independent

만약 $p(z) = \mathcal{N}(0, I)$ (factored), $q(z)$ 가 factored 와 일치하려면 위와 같은 disentangled mapping 이 자연.

**$\beta > 1$ 의 역할**: factored prior 와 일치 시키는 압력 강화 → entangled 표현이 더 큰 KL 페널티.

(이 argument 은 informal — Locatello 2019 의 reservation 참고)

### 유도 3 — Rate-Distortion Curve (Alemi 2018)

각 $\beta$ 에 대한 trained model 의 (R, D) plot:

- **Rate** $R = \mathbb{E}_{p_d}[\text{KL}(q(z|x) \| p(z))]$ ≈ $I(X; Z)$ upper bound
- **Distortion** $D = -\mathbb{E}_{p_d, q}[\log p(x|z)]$ ≈ reconstruction error

$\beta$ 변화로 (R, D) 가 곡선 따라 이동 — **rate-distortion frontier** 의 모양 reveal.

**시사점**:
- $\beta = 1$: ELBO 의 한 점만, 전체 곡선 아님
- 다양한 $\beta$ 로 학습하면 frontier 전체 explore
- 실전 응용에 따라 $\beta$ 선택 — 압축이 중요하면 large $\beta$, fidelity 중요하면 small

### 유도 4 — β와 Posterior Collapse 의 관계

$\beta \to \infty$ 의 극단: KL = 0 강제 → $q(z|x) = p(z)$ for all $x$ → latent ignored, decoder 만으로 reconstruction. Posterior collapse.

**Mode of failure**: $\beta$ 너무 크면 latent 가 unused. $\beta$ 너무 작으면 latent 가 overuse, prior 와 멀어져 sampling quality 손상.

**Sweet spot**: dataset-dependent. dSprites: $\beta \approx 4$. Faces: $\beta \approx 250$. CelebA: $\beta \approx 100$. (원 논문 보고)

### 유도 5 — β Annealing Schedule

훈련 초기에 $\beta = 0$ (KL 끔), 점차 $\beta = \beta_\text{target}$ 로 증가:

$$\beta(t) = \beta_\text{target} \cdot \min(1, t / T_\text{warmup})$$

**효과**:
- 초기: pure reconstruction → encoder 가 informative latent 학습
- 점차 KL 추가: 학습된 latent 위에 disentanglement 압력
- Posterior collapse 방지

대안: cyclic annealing (Fu 2019) — $\beta$ 를 cyclically 0 ↔ target 변동.

---

## 💻 실험 재현 / PyTorch 구현

### 실험 1 — β-VAE on dSprites

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.utils import make_grid

class BetaVAE(nn.Module):
    def __init__(self, x_shape=(1, 64, 64), z_dim=10):
        super().__init__()
        self.z_dim = z_dim
        # CNN encoder
        self.enc = nn.Sequential(
            nn.Conv2d(1, 32, 4, 2, 1), nn.ReLU(),
            nn.Conv2d(32, 32, 4, 2, 1), nn.ReLU(),
            nn.Conv2d(32, 64, 4, 2, 1), nn.ReLU(),
            nn.Conv2d(64, 64, 4, 2, 1), nn.ReLU(),
            nn.Flatten(),
            nn.Linear(64 * 4 * 4, 256), nn.ReLU(),
        )
        self.mu = nn.Linear(256, z_dim)
        self.logvar = nn.Linear(256, z_dim)
        # CNN decoder
        self.dec_fc = nn.Sequential(
            nn.Linear(z_dim, 256), nn.ReLU(),
            nn.Linear(256, 64 * 4 * 4), nn.ReLU(),
        )
        self.dec_conv = nn.Sequential(
            nn.ConvTranspose2d(64, 64, 4, 2, 1), nn.ReLU(),
            nn.ConvTranspose2d(64, 32, 4, 2, 1), nn.ReLU(),
            nn.ConvTranspose2d(32, 32, 4, 2, 1), nn.ReLU(),
            nn.ConvTranspose2d(32, 1, 4, 2, 1),
        )

    def encode(self, x):
        h = self.enc(x)
        return self.mu(h), self.logvar(h)
    def decode(self, z):
        h = self.dec_fc(z).view(-1, 64, 4, 4)
        return self.dec_conv(h)
    def forward(self, x):
        mu, logvar = self.encode(x)
        std = (0.5 * logvar).exp()
        z = mu + std * torch.randn_like(std)
        return self.decode(z), mu, logvar

def beta_vae_loss(x, x_recon, mu, logvar, beta=4.0):
    recon = F.binary_cross_entropy_with_logits(x_recon, x, reduction='sum')
    kl = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp()).sum()
    return recon + beta * kl, recon, kl

# 다양한 β 로 학습
betas = [0.5, 1.0, 4.0, 10.0]
models = {}
for beta in betas:
    model = BetaVAE(z_dim=10).cuda()
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    # ... training loop ...
    models[beta] = model

# Disentanglement 측정: latent traversal
@torch.no_grad()
def latent_traversal(model, x, dim_idx, range_=(-3, 3), n=10):
    mu, _ = model.encode(x.unsqueeze(0))
    traversals = []
    for v in torch.linspace(range_[0], range_[1], n):
        z = mu.clone()
        z[0, dim_idx] = v
        traversals.append(torch.sigmoid(model.decode(z)))
    return torch.cat(traversals)

# β=4 모델: 각 dim traversal → 한 factor 만 변화 (disentangled)
# β=1 모델: traversal 시 여러 factor 동시 변화 (entangled)
```

### 실험 2 — Rate-Distortion Curve

```python
# 다양한 β 로 학습 → (R, D) plot
import matplotlib.pyplot as plt

R_list, D_list = [], []
for beta in [0.1, 0.5, 1.0, 2.0, 4.0, 8.0, 16.0, 32.0]:
    model = BetaVAE().cuda()
    # train ...
    with torch.no_grad():
        R, D = 0, 0
        for x, _ in test_loader:
            x_recon, mu, logvar = model(x.cuda())
            R += -0.5 * (1 + logvar - mu.pow(2) - logvar.exp()).sum().item()
            D += F.binary_cross_entropy_with_logits(
                x_recon, x.cuda(), reduction='sum').item()
        R_list.append(R / len(test_loader.dataset))
        D_list.append(D / len(test_loader.dataset))

plt.plot(R_list, D_list, 'o-')
plt.xlabel('Rate (KL nats)'); plt.ylabel('Distortion (recon nats)')
plt.title('β-VAE Rate-Distortion Curve')
# 예상: 단조 감소 — rate 증가 시 distortion 감소
```

### 실험 3 — β Annealing

```python
# β 를 0 → 4 로 점진 증가
def train_with_annealing(model, opt, loader, T_warmup=10000):
    step = 0
    for epoch in range(50):
        for x, _ in loader:
            beta = 4.0 * min(1.0, step / T_warmup)   # linear warmup
            x_recon, mu, logvar = model(x.cuda())
            loss, recon, kl = beta_vae_loss(x.cuda(), x_recon, mu, logvar, beta)
            opt.zero_grad(); loss.backward(); opt.step()
            step += 1
            if step % 100 == 0:
                print(f"Step {step}, β = {beta:.2f}, recon = {recon:.0f}, kl = {kl:.0f}")

# 효과: 초기에 reconstruction 학습, 점차 disentanglement 학습
# Posterior collapse 방지
```

### 실험 4 — Mutual Information Gap (MIG) Metric

```python
def mig_score(model, dataset_with_factors, n_samples=10000):
    """MIG: max - 2nd-max MI per factor, average across factors"""
    # 1. Sample (x, f) pairs
    # 2. Encode → z
    # 3. For each factor f_j:
    #    Compute MI(f_j; z_i) for each i
    #    MIG_j = (max_i MI - 2nd-max_i MI) / H(f_j)
    # 4. Average over j
    # 자세한 구현은 Chen 2018 의 official repo 참조
    pass

# 일반 결과 (dSprites):
# β = 1.0: MIG ≈ 0.05
# β = 4.0: MIG ≈ 0.20 (more disentangled)
# β = 16.0: MIG ≈ 0.18 (degraded due to posterior collapse)
```

---

## 🔗 이론과 실전의 간극

### 1. β tuning 의 어려움

Optimal $\beta$ 가 dataset-dependent. 새 데이터셋에 적용 시 grid search 또는:
- **Capacity-controlled β-VAE** (Burgess 2018): KL 의 target 값 ($C$) 직접 지정, $\beta$ 자동 조절
- **CCI-VAE**: $C$ 를 schedule (점진적 증가)
- **FactorVAE** (Kim & Mnih 2018): Total Correlation 만 페널티 → reconstruction 손해 감소

### 2. Disentanglement 의 측정과 해석

**Locatello 2019** 의 결과: 같은 dataset 에서도 다른 random seed 가 다른 disentanglement metric 을 줌 — **identifiability 보장 없음**.

해결 방향:
- **Weak supervision** (Locatello 2020): 일부 factor label 사용
- **Group VAE** (Khemakhem 2020): Identifiable VAE with auxiliary labels
- **Contrastive disentanglement** (Tian 2021)

### 3. Beta-VAE 의 후속 영향

β-VAE 의 단순한 modification 이 큰 영향:
- **InfoVAE** (Zhao 2017): MMD 기반 regularization
- **WAE** (Wasserstein Autoencoder, Tolstikhin 2017): Wasserstein-based
- **Discrete β-VAE** (e.g., VQ-VAE 의 β-similar regularization)
- **CCI-VAE, FactorVAE, β-TC-VAE** 등 disentanglement 의 다양한 변형

---

## ⚖️ 가정과 한계

| 가정 | 한계 |
|------|------|
| $\beta > 1$ 이 disentanglement 를 induce | Locatello 2019: 데이터/seed dependence, 보장 없음 |
| Factored Gaussian encoder + prior | 진짜 generative factor 가 axis-aligned 라는 가정 |
| Reconstruction quality 와 disentanglement trade-off | $\beta$ 너무 크면 둘 다 손상 (collapse) |
| 단일 $\beta$ 가 모든 latent dim 에 동일 | Per-dimension capacity 가 더 좋을 수 있음 |
| Synthetic dataset (dSprites) 에서의 evaluation | Real-world 일반화 한계 |

---

## 📌 핵심 정리

$$\boxed{\mathcal{L}_\beta = \mathbb{E}_{q_\phi(z|x)}[\log p_\theta(x|z)] - \beta \cdot \text{KL}(q_\phi(z|x) \| p(z))}$$

$$\boxed{\mathbb{E}_{p_d}[\text{KL}(q(z|x) \| p(z))] = I(X; Z) + \text{KL}(q(z) \| p(z))}$$

| $\beta$ | 효과 | 결과 |
|---------|------|------|
| $\beta = 0$ | KL 무시 | Pure AE, no regularization, hole 가능 |
| $\beta < 1$ | 약한 KL | High info, sharp recon, less disentangled |
| $\beta = 1$ | 표준 ELBO | Standard VAE |
| $\beta > 1$ | 강한 KL | Disentanglement, but blurry recon |
| $\beta \to \infty$ | KL = 0 강제 | Posterior collapse |

| 관련 metric | 의미 |
|-----------|------|
| **Rate** $R$ | $\approx I(X; Z)$ upper bound |
| **Distortion** $D$ | Reconstruction error |
| **MIG** | Mutual Information Gap, disentanglement |
| **β-VAE metric** | Pair-wise factor classification |
| **FactorVAE score** | TC-based disentanglement |

---

## 🤔 생각해볼 문제

**문제 1** (기초): β-VAE 의 KL 항을 분해 $\mathbb{E}_{p_d}[\text{KL}(q(z|x) \| p(z))] = I(X; Z) + \text{KL}(q(z) \| p(z))$ 를 증명하라.

<details>
<summary>해설</summary>

$$\mathbb{E}_{p_d(x)}[\text{KL}(q(z|x) \| p(z))] = \mathbb{E}_{p_d(x)} \mathbb{E}_{q(z|x)}[\log q(z|x) - \log p(z)]$$

분리: $\log q(z|x) = \log q(z|x) - \log q(z) + \log q(z)$. 대입:

$$= \mathbb{E}\left[\log \frac{q(z|x)}{q(z)}\right] + \mathbb{E}[\log q(z) - \log p(z)]$$

첫 항: $\mathbb{E}_{p_d(x), q(z|x)}\left[\log \frac{q(z|x)}{q(z)}\right] = I(X; Z)$ — definition of MI.

둘째 항: $\mathbb{E}_{q(z)}[\log q(z) - \log p(z)] = \text{KL}(q(z) \| p(z))$.

**시사점**: KL 페널티 = $I(X; Z)$ + aggregate-prior mismatch. β > 1 이 둘 다 페널티.

</details>

**문제 2** (심화): β-VAE 가 sufficiently large $\beta$ 에서 posterior collapse (KL = 0 for all $x$) 로 수렴함을 보여라. 이때 reconstruction loss 는 어떻게 되는가?

<details>
<summary>해설</summary>

**Optimization argument**: Loss $\mathcal{L}_\beta = D + \beta R$, $R = \text{KL} \geq 0$. $\beta \to \infty$ 일 때 $R = 0$ 이 강제 → $q(z|x) = p(z)$ for all $x$ — encoder 가 unconditional, latent 무력화.

**Reconstruction**: $z \sim p(z) = \mathcal{N}(0, I)$, decoder 가 $z$ 만 보고 $x$ 를 reconstruct. $z$ 가 무작위 sample 이므로 (no info from $x$), decoder 는 결국 데이터의 평균만 예측 — 매우 blurry.

**왜 모델이 이 mode 로 가는가**: optimization 의 local minimum. KL 항이 너무 크면, **어떤 informative encoding 도 KL 페널티가 reconstruction 이득을 능가** → 모델이 latent 사용 포기, $q(z|x) = p(z)$ 로 가는 것이 더 나은 loss.

**해결**: β annealing, free bits, 또는 hierarchical VAE.

**시사점**: posterior collapse 는 β-VAE 의 fundamental limitation. 적절한 $\beta$ 와 schedule 필수.

</details>

**문제 3** (논문 비평): Locatello 2019 의 "Challenging Common Assumptions in Unsupervised Learning of Disentangled Representations" 의 핵심 발견 — "pure unsupervised disentanglement 가 보장되지 않음" — 의 의미를 설명하라. β-VAE 가 여전히 useful 한가?

<details>
<summary>해설</summary>

**Locatello 2019 의 발견**:
1. **Identifiability 부재**: 같은 데이터, 같은 architecture 도 random seed 에 따라 다른 representation
2. **Disentanglement 와 downstream task 성능 의 약한 상관**: "더 disentangled" 가 항상 더 좋은 representation 아님
3. **Inductive bias 의존**: β-VAE 의 disentanglement 는 axis-aligned prior + factored encoder 의 bias 결과 — 어떤 데이터에서는 valid, 다른 데이터에서는 부적절

**시사점**:
- "Disentanglement" 자체가 ill-defined — generative factor 가 무엇인지 의존
- 데이터의 internal structure 와 prior 가 일치할 때만 작동
- Pure unsupervised 로는 한계, **약간의 supervision 필요**

**β-VAE 의 가치**:
1. **Useful baseline**: simple, 한 줄 변경, 많은 dataset 에서 reasonable disentanglement
2. **IB framework 의 instantiation**: rate-distortion frontier 의 한 점들
3. **β annealing, capacity control 의 starting point**
4. **Theoretical insight**: KL 의 information bottleneck 해석

**현대적 동향**: Disentanglement 자체보다는 **causal representation learning, contrastive learning** 으로 이동. 단, β-VAE 의 단순함과 분석 가능성은 여전히 valuable.

**결론**: β-VAE 가 "magical disentanglement solution" 은 아니지만 (Locatello 의 critique 정당), **useful tool with clear theoretical foundation** 으로 여전히 유효.

</details>

---

<div align="center">

[◀ 이전 (02. Reparameterization)](./02-reparameterization.md) | [📚 README](../README.md) | [다음 ▶ (04. Posterior Collapse)](./04-posterior-collapse.md)

</div>
