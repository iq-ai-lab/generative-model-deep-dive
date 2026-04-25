# 04. Wasserstein GAN (Arjovsky 2017)

## 🎯 핵심 질문

- Wasserstein-1 distance $W_1(p, q) = \inf_\gamma \mathbb{E}[\|X - Y\|]$ 의 정확한 정의와 의미는?
- Kantorovich-Rubinstein duality $W_1 = \sup_{\|f\|_L \leq 1} \mathbb{E}_p[f] - \mathbb{E}_q[f]$ 의 derivation?
- 1-Lipschitz constraint $\|f\|_L \leq 1$ 가 NN 에서 어떻게 enforced 되는가?
- Weight clipping (WGAN) vs Gradient Penalty (WGAN-GP) 의 trade-off?
- Wasserstein 이 JSD 보다 우월한 수학적 이유 — support 가 disjoint 일 때도 informative gradient?

---

## 🔍 왜 WGAN 이 결정적인가

Arjovsky 2017 "Wasserstein GAN" 은 GAN 의 fundamental 문제 — JSD saturation 과 mode collapse — 를 distance metric 의 변경으로 해결:

1. **Wasserstein-1**: support disjoint 에서도 informative gradient
2. **Kantorovich-Rubinstein dual**: 1-Lipschitz $f$ 로 효율적 추정
3. **Stable training**: loss 가 monotone, training progress measurable
4. **Mode collapse 감소**: transport-based metric, mode missing 직접 penalty

WGAN 의 후속 발전 — WGAN-GP (Gulrajani 2017), Spectral Norm (Miyato 2018) — 모두 Lipschitz 제약의 다른 구현. 이 문서에서는 Wasserstein distance 의 수학과 dual formulation, 그리고 WGAN/WGAN-GP 의 정확한 algorithm 을 다룹니다.

---

## 📐 수학적 선행 조건

- 이전 문서들: 01-minimax, 02-jsd, 03-mode-collapse
- [Convex Optimization Deep Dive](https://github.com/iq-ai-lab/convex-optimization-deep-dive): Duality, Lagrangian
- [Probability Theory Deep Dive](https://github.com/iq-ai-lab/probability-theory-deep-dive): Coupling, marginal

---

## 📖 직관적 이해

### "흙더미를 옮기는 비용"

Wasserstein-1 의 직관: 분포 $p$ 의 mass 를 분포 $q$ 의 위치로 옮기는 **최소 비용** (Earth Mover's Distance).

- $p$ = "오늘 흙이 쌓여 있는 분포"
- $q$ = "내일 옮기고 싶은 분포"
- 비용 = "각 흙더미를 옮기는 거리의 가중 합"

수학적: coupling $\gamma(x, y)$ = "위치 $x$ 의 mass 를 $y$ 로 옮기는 양". Marginals constrains $\gamma \to p, \gamma \to q$.

$$W_1 = \inf_\gamma \mathbb{E}_{(x, y) \sim \gamma}[\|x - y\|]$$

### "JSD 와의 결정적 차이"

두 점 분포 $p = \delta_a, q = \delta_b$ (disjoint support):

- $JSD(p, q) = \log 2$ (constant, 거리 무관)
- $W_1(p, q) = \|a - b\|$ (선형, 거리 비례)

**시사점**: 두 분포가 떨어질수록 $W_1$ 이 비례적으로 증가 — gradient 가 informative. JSD 는 saturated.

### Kantorovich-Rubinstein Dual 의 마법

Primal $\inf_\gamma$ 는 high-dim integral over couplings — direct 계산 불가능.

**Dual** (Kantorovich-Rubinstein, Villani 2003):

$$W_1 = \sup_{\|f\|_L \leq 1} \mathbb{E}_p[f] - \mathbb{E}_q[f]$$

여기서 $\|f\|_L$ 는 Lipschitz norm. **NN $f_\theta$ 로 sup 추정** — WGAN 의 "critic".

### 1-Lipschitz Constraint 의 enforcement

NN $f_\theta$ 가 1-Lipschitz: $\|f_\theta(x) - f_\theta(y)\| \leq \|x - y\|$.

**방법들**:
1. **Weight clipping** (WGAN): $\theta \leftarrow \text{clip}(\theta, -c, c)$ — crude, biased
2. **Gradient penalty** (WGAN-GP): loss 에 $\lambda (\|\nabla_x f\| - 1)^2$ 항 — soft
3. **Spectral Norm** (Ch5-05): each weight matrix 의 spectral norm = 1

각각의 trade-off — 다음 문서들에서 더.

---

## ✏️ 엄밀한 정의·정리

### 정의 4.1 — Wasserstein-1 Distance

분포 $p, q$ on $\mathcal{X}$, ground distance $d(x, y) = \|x - y\|$:

$$W_1(p, q) = \inf_{\gamma \in \Pi(p, q)} \mathbb{E}_{(x, y) \sim \gamma}[\|x - y\|]$$

여기서 $\Pi(p, q) = \{\gamma : \pi_1 \gamma = p, \pi_2 \gamma = q\}$ (marginals 가 $p, q$ 인 결합 분포의 집합).

**Properties**:
- $W_1 \geq 0$, $W_1 = 0 \iff p = q$
- Symmetric: $W_1(p, q) = W_1(q, p)$
- Triangle inequality
- **Metric** on probability distributions

### 정리 4.2 — Kantorovich-Rubinstein Duality

$$W_1(p, q) = \sup_{f: \|f\|_L \leq 1} \mathbb{E}_{p}[f] - \mathbb{E}_q[f]$$

여기서 $\|f\|_L = \sup_{x \neq y} |f(x) - f(y)| / \|x - y\|$ 는 Lipschitz constant.

**Sketch of proof** (full proof in Villani 2003):

Primal: $\min_\gamma \mathbb{E}_\gamma[\|x - y\|]$ subject to marginals.

Lagrangian:
$$L(\gamma, \phi, \psi) = \mathbb{E}_\gamma[\|x - y\|] + \int \phi (p - \int \gamma dy) + \int \psi (q - \int \gamma dx)$$

Dual (after manipulation):
$$\sup_{\phi, \psi} \mathbb{E}_p[\phi] - \mathbb{E}_q[\psi] \text{ s.t. } \phi(x) - \psi(y) \leq \|x - y\|$$

Choosing $f = \phi = \psi$: 1-Lipschitz constraint, $W_1 = \sup_f \mathbb{E}_p[f] - \mathbb{E}_q[f]$. $\square$

### 정의 4.3 — WGAN Loss

**Critic** (1-Lipschitz NN): $f_\phi$.

**Generator**: $G_\theta(z)$ as before.

$$L_\text{WGAN} = \min_G \max_{\|f\|_L \leq 1} \mathbb{E}_{p_d}[f(x)] - \mathbb{E}_{p_z}[f(G(z))]$$

이는 $\min_G W_1(p_d, p_g)$ 와 동치 (Lipschitz constraint 만족 시).

### 정의 4.4 — Weight Clipping (WGAN)

각 update 후:

$$\phi \leftarrow \text{clip}(\phi, -c, c) \quad \text{element-wise, } c = 0.01 \text{ typical}$$

이로 $\|f\|_L$ 의 upper bound 가 $\theta$-dependent 한 constant — **approximately 1-Lipschitz** if NN 의 architecture 고정.

**Issue**:
- Capacity reduction: clip 이 강하면 NN 표현력 감소
- Pathological behavior: weight 가 $\pm c$ 의 두 극단에 모임
- Bias: Lipschitz 조건의 정확한 enforcement 아님

### 정의 4.5 — Gradient Penalty (WGAN-GP, Gulrajani 2017)

Soft constraint via loss term:

$$L_\text{WGAN-GP} = \mathbb{E}_{p_d}[f] - \mathbb{E}_{p_g}[f] + \lambda \mathbb{E}_{\hat x}[(\|\nabla_{\hat x} f(\hat x)\|_2 - 1)^2]$$

여기서 $\hat x = \alpha x + (1 - \alpha) G(z)$, $\alpha \sim \mathcal{U}(0, 1)$ — interpolation between real and fake.

**$\lambda = 10$ 표준**.

**Justification**: optimal $f$ in $W_1$ has $\|\nabla f\| = 1$ on geodesic between $p_d$ and $p_g$. Penalty enforces 이 condition on the interpolated samples.

### 정리 4.6 — Wasserstein 이 JSD 보다 Smooth in $\theta$

$p_{g, \theta} = \mathcal{N}(\theta, \sigma^2 I)$ 의 simple example.

- $W_1(p_d, p_g) = \|\mu_d - \theta\|$ — linear in $\theta$, smooth
- $JSD(p_d, p_g)$: $\theta \to \infty$ 에서 $\log 2$ saturated — non-informative gradient

따라서 WGAN 의 generator gradient 가 항상 informative.

---

## 🔬 증명 및 수학적 유도

### 유도 1 — 1D Wasserstein 의 Closed Form

1D 분포 $p, q$ 의 CDF $F_p, F_q$. Then:

$$W_1(p, q) = \int_0^1 |F_p^{-1}(t) - F_q^{-1}(t)| dt = \int_{-\infty}^\infty |F_p(x) - F_q(x)| dx$$

(quantile-based 또는 CDF-based 형태)

이는 1D 에서 $W_1$ 가 closed-form 으로 계산 가능. High-dim 에서는 일반적으로 계산 어려움.

### 유도 2 — Two-Point Distribution 의 $W_1$ 와 JSD 비교

$p = \delta_a, q = \delta_b$ (Dirac measures at $a, b$).

**$W_1$**: 유일 coupling $\gamma = \delta_{(a, b)}$ → $W_1 = \|a - b\|$.

**JSD**: $m = (1/2)(\delta_a + \delta_b)$. $\text{KL}(p \| m) = \log 2$ (since $p(a) = 1, m(a) = 1/2$). 마찬가지 $\text{KL}(q \| m) = \log 2$. $JSD = \log 2$.

**비교**: $\|a - b\|$ varies, $JSD = \log 2$ constant.

**시사점**: Wasserstein 이 거리 정보 보존, JSD 는 binary (overlap or not).

### 유도 3 — KR Dual 의 Dual Variable Interpretation

KR dual: $\sup_{f \text{ 1-Lipschitz}} \mathbb{E}_p[f] - \mathbb{E}_q[f]$.

Optimal $f^*$ = "$p$ 와 $q$ 를 가장 잘 구별하는 1-Lipschitz function". 의미:
- $f^*(x_p)$ 가 큼, $f^*(x_q)$ 가 작음 — $p$ vs $q$ 의 separator
- 1-Lipschitz constraint 가 "smooth gradient of separation"

$\nabla f^*$ 가 "transport direction" — $q$ mass 를 $p$ 로 가져가는 vector field.

**Empirical**: critic 의 logit 을 시각화하면 $p_d$ 영역에서 큰 값, $p_g$ 영역에서 작은 값. Transport direction 명확.

### 유도 4 — WGAN-GP 의 $\|\nabla f\| = 1$ Justification

**Theorem** (Gulrajani 2017): $W_1$ 의 optimal $f^*$ 는 $p_d, p_g$ 사이 transport plan 의 geodesic 위에서 $\|\nabla f^*\| = 1$ a.e.

**Idea**: $f^*$ 가 1-Lipschitz, $W_1$ 의 sup 도달 → "saturated" Lipschitz constraint. $\|\nabla f^*\|$ 가 maximal value 1.

**WGAN-GP 의 enforcement**: penalty $(\|\nabla f\| - 1)^2$ on interpolated points — soft, 약간의 bias 있지만 weight clipping 보다 우월.

### 유도 5 — Stability Comparison

**Standard GAN training**:
- Loss curve: oscillating, non-monotone
- $D$ 가 "너무 강해지면" $G$ gradient 약함
- Mode collapse 흔함

**WGAN training**:
- Critic 이 더 학습할수록 $W_1$ 추정 정확도 증가 → no over-training problem
- $W_1(p_d, p_g)$ 가 monotone (convergent training 의 indicator)
- Mode collapse 거의 없음

**5x more critic updates per generator** (WGAN 표준): critic 이 1-Lipschitz family 에서 잘 학습되도록.

---

## 💻 실험 재현 / PyTorch 구현

### 실험 1 — WGAN with Weight Clipping

```python
import torch
import torch.nn as nn
import torch.optim as optim

class Critic(nn.Module):
    def __init__(self, dim=2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, 128), nn.LeakyReLU(0.2),
            nn.Linear(128, 128), nn.LeakyReLU(0.2),
            nn.Linear(128, 1),
        )
    def forward(self, x):
        return self.net(x)   # no sigmoid (return scalar)

G = Generator()      # from previous chapter
critic = Critic()
opt_G = optim.RMSprop(G.parameters(), lr=5e-5)
opt_C = optim.RMSprop(critic.parameters(), lr=5e-5)

n_critic = 5         # update critic 5x per G update
clip_value = 0.01

for step in range(20000):
    for _ in range(n_critic):
        x_real = sample_8gauss(256)
        z = torch.randn(256, 2)
        x_fake = G(z).detach()
        # Maximize E[critic(real)] - E[critic(fake)]
        loss_C = -(critic(x_real).mean() - critic(x_fake).mean())
        opt_C.zero_grad(); loss_C.backward(); opt_C.step()
        # Weight clip
        for p in critic.parameters():
            p.data.clamp_(-clip_value, clip_value)

    # Train G
    z = torch.randn(256, 2)
    loss_G = -critic(G(z)).mean()
    opt_G.zero_grad(); loss_G.backward(); opt_G.step()

    if step % 500 == 0:
        # W_1 estimate
        with torch.no_grad():
            x_real = sample_8gauss(1000)
            z = torch.randn(1000, 2)
            x_fake = G(z)
            W1_estimate = (critic(x_real).mean() - critic(x_fake).mean()).item()
        print(f"Step {step}: W1 ≈ {W1_estimate:.3f}")

# WGAN: mode coverage 가 표준 GAN 보다 보통 우월
```

### 실험 2 — WGAN-GP 구현

```python
def gradient_penalty(critic, x_real, x_fake, lambda_gp=10):
    """WGAN-GP 의 gradient penalty term"""
    batch_size = x_real.size(0)
    alpha = torch.rand(batch_size, 1, device=x_real.device)
    x_hat = alpha * x_real + (1 - alpha) * x_fake
    x_hat.requires_grad_(True)
    f_hat = critic(x_hat)
    grads = torch.autograd.grad(
        outputs=f_hat.sum(), inputs=x_hat,
        create_graph=True, retain_graph=True,
    )[0]
    gp = ((grads.norm(2, dim=-1) - 1) ** 2).mean()
    return lambda_gp * gp

# WGAN-GP training (replace weight clipping)
for step in range(20000):
    for _ in range(n_critic):
        x_real = sample_8gauss(256)
        z = torch.randn(256, 2)
        x_fake = G(z).detach()
        loss_critic = critic(x_fake).mean() - critic(x_real).mean()
        gp = gradient_penalty(critic, x_real, x_fake)
        total = loss_critic + gp
        opt_C.zero_grad(); total.backward(); opt_C.step()

    # Train G (same as before)
    z = torch.randn(256, 2)
    loss_G = -critic(G(z)).mean()
    opt_G.zero_grad(); loss_G.backward(); opt_G.step()

# WGAN-GP: weight clipping 제거, gradient penalty 추가
# 일반적으로 더 stable + better quality
```

### 실험 3 — Loss 의 Monotonicity 비교

```python
# Standard GAN vs WGAN 의 loss curve
plt.plot(losses_G_standard, label='Standard GAN G loss')
plt.plot(losses_C_standard, label='Standard D loss')
# Standard: oscillating, non-monotone

plt.figure()
plt.plot(W1_estimates, label='WGAN W1 estimate')
# WGAN: monotone decreasing — training progress measurable
```

### 실험 4 — Mode Coverage 비교

```python
# 동일 8-Gaussians data, 다른 GAN 으로 학습 후 mode coverage 측정
# Standard GAN: 종종 1-3 modes
# WGAN: 보통 6-8 modes
# WGAN-GP: 보통 8 modes (full coverage)
# Spectral Norm: WGAN-GP 와 비슷
```

---

## 🔗 이론과 실전의 간극

### 1. Weight Clipping 의 한계

Arjovsky 2017 의 weight clipping 은 simple but problematic:
- $c = 0.01$ 이 critic capacity 강하게 제한
- Weight 가 $\pm c$ 의 boundary 에 모이는 pathology
- Lipschitz constant 가 architecture 에 의존 — exact 1-Lipschitz 아님

WGAN-GP, Spectral Norm 이 일반적으로 우월.

### 2. Gradient Penalty 의 Cost

WGAN-GP 의 GP term: 매 step 에 추가 backward (gradient of gradient) → **2x compute** vs standard WGAN.

이를 회피하려는 alternatives:
- **Spectral Norm** (Miyato 2018): per-weight, no GP
- **Consistency regularization** (Wei 2018): GP 의 변형
- **R1 regularization** (Mescheder 2018): real-only gradient penalty

### 3. WGAN 의 Modern Status

WGAN-GP 와 Spectral Norm 이 modern GAN 의 표준. 대부분의 production GAN (StyleGAN, BigGAN) 이 spectral norm 사용. Wasserstein 의 통찰 (transport-based, no saturation) 이 design philosophy 의 일부.

**Diffusion 시대**: GAN 의 mainstream usage 감소. 단, real-time 용 (HiFi-GAN vocoder, GAN-based super-resolution) 에서 여전히 활용.

---

## ⚖️ 가정과 한계

| 가정 | 한계 |
|------|------|
| 1-Lipschitz constraint 정확 | Weight clipping 부정확, GP 가 soft, Spectral Norm 이 approximate |
| KR duality 가 NN 으로 추정 | Critic 의 capacity 부족 시 부정확 |
| $W_1$ 이 perceptual distance 와 align | Always 그렇지 않음, FID 는 다른 metric |
| 5x critic updates 가 균형 | Dataset/architecture dependent |
| Wasserstein 이 항상 better than JSD | High-dim 에서 difference 감소 가능 |

---

## 📌 핵심 정리

$$\boxed{W_1(p, q) = \inf_{\gamma \in \Pi(p, q)} \mathbb{E}[\|x - y\|]}$$

$$\boxed{W_1 = \sup_{\|f\|_L \leq 1} \mathbb{E}_p[f] - \mathbb{E}_q[f] \quad \text{(Kantorovich-Rubinstein dual)}}$$

| WGAN component | 설명 |
|---------------|------|
| **Critic** | 1-Lipschitz NN $f_\phi$ (no sigmoid) |
| **Critic loss** | $-\mathbb{E}_p[f] + \mathbb{E}_q[f]$ (negate for maximization) |
| **Generator loss** | $-\mathbb{E}_z[f(G(z))]$ |
| **Lipschitz** | Weight clipping (WGAN), GP (WGAN-GP), Spectral Norm |
| **# critic updates** | 5x per generator (typically) |

| Metric | JSD | $W_1$ |
|--------|-----|-------|
| Disjoint support | Saturated at $\log 2$ | Linear in distance |
| Symmetric | Yes | Yes |
| Metric | $\sqrt{JSD}$ is | $W_1$ is |
| GAN vanishing gradient | Yes | No |

---

## 🤔 생각해볼 문제

**문제 1** (기초): 1D 점 분포 $p = \delta_0, q = \delta_5$ 의 $W_1$ 와 KR dual sup 의 optimal $f^*$ 를 구하라.

<details>
<summary>해설</summary>

**$W_1$**: 유일 coupling, $W_1 = |0 - 5| = 5$.

**KR dual**: $\sup_{|f|_L \leq 1} f(0) - f(5)$.

1-Lipschitz: $|f(0) - f(5)| \leq |0 - 5| = 5$. 따라서 $f(0) - f(5) \leq 5$.

**Optimal**: $f^*(x) = -x + c$ (linear with slope -1, $|f^*|_L = 1$). $f^*(0) - f^*(5) = c - (c - 5) = 5$. ✓

(또는 $f^*(x) = -x$ — $c = 0$ choice)

**시사점**: optimal critic 이 "ground distance" 를 따라가는 linear function. $\nabla f^* = -1$ — transport direction.

</details>

**문제 2** (심화): WGAN-GP 의 gradient penalty term 이 $\hat x = \alpha x + (1-\alpha) G(z)$ 의 interpolated point 에 적용되는 이유를 KR dual 의 saturation 측면에서 설명하라.

<details>
<summary>해설</summary>

**KR dual optimal**: $f^*$ 가 $p_d, p_g$ 사이의 **optimal transport plan 의 geodesic** 위에서 $\|\nabla f^*\| = 1$ a.e.

**Geodesic interpretation**:
- Optimal coupling $\gamma^*$: 각 $x \sim p_d$ 와 $y \sim p_g$ 를 transport
- Geodesic $\hat x = \alpha x + (1 - \alpha) y$ 가 transport path
- 이 path 위에서 $f^*$ 의 gradient = unit vector along path

**Why interpolation?**:
- $f^*$ 의 1-Lipschitz constraint 가 $p_d$ 와 $p_g$ 의 supports 만이 아니라 **그 사이의 path** 에서도 saturated
- WGAN-GP 의 GP term: $(\|\nabla f(\hat x)\| - 1)^2$ on $\hat x$ — interpolated points 에서 unit gradient 강제

**Alternative**: $\hat x = x$ (real only) 또는 $\hat x = G(z)$ (fake only) 도 가능 — 하지만 interpolation 이 path 전체를 cover.

**Practical**:
- $\alpha \sim \mathcal{U}(0, 1)$ random sampling
- 매 step 의 random $\hat x$ 가 sufficient enforcement
- $\lambda = 10$ default — too large 면 critic capacity 손해, too small 면 enforcement 약함

**미묘한 점**:
- Optimal $f^*$ 가 모든 곳에서 $\|\nabla\| = 1$ 아님 — geodesic 위에서만
- GP 가 over-constraint 가능 — bias 있음
- 실전에서는 잘 작동, 이는 GP 의 robustness

**시사점**: GP term 은 KR optimal 의 **수학적 property** 의 enforcement. Soft 이지만 effective. Spectral Norm 의 hard constraint (per-weight) 와 다른 접근.

</details>

**문제 3** (논문 비평): WGAN-GP vs Spectral Norm 의 비교 — 두 접근의 architectural difference, computational cost, empirical performance 를 분석하라.

<details>
<summary>해설</summary>

**WGAN-GP**:
- Loss-level constraint: $\lambda(\|\nabla f\| - 1)^2$
- Soft (allows occasional violation)
- Sample-based: interpolated points 에서 enforcement
- Cost: 추가 backward (gradient of gradient) — **2x backward**
- Hyperparameter: $\lambda$, $n_\text{critic}$

**Spectral Norm** (Miyato 2018, Ch5-05):
- Architecture-level constraint: $\sigma(W_l) = 1$ for each weight matrix
- Hard (exact constraint)
- Power iteration: $u^{(t+1)} = W v^{(t)} / \|W v^{(t)}\|$, $v^{(t+1)} = W^\top u^{(t+1)} / \|...\|$ — 1-2 iterations per forward
- Cost: 약간 추가 (per-layer, but cheap)
- Hyperparameter: minimal

**Architectural Difference**:
- WGAN-GP: $\|f\|_L \leq 1$ globally
- Spectral Norm: $\|f\|_L \leq \prod_l \sigma(W_l) = 1$ — product of layer Lipschitz

**Performance**:
- 작은 dataset (CIFAR-10, etc.): 비슷 performance
- 큰 dataset (ImageNet): Spectral Norm 약간 우월 (BigGAN 의 선택)
- Stability: 둘 다 standard GAN 보다 우월
- Mode coverage: WGAN-GP 가 약간 우월 (geodesic enforcement 의 직접성)

**Computational**:
- Memory: WGAN-GP 가 추가 (gradient of gradient)
- Speed: Spectral Norm 이 빠름 (per-layer cheap operation)

**Modern usage**:
- BigGAN, StyleGAN: Spectral Norm
- 일부 research: WGAN-GP (다양한 변형 탐색)
- Diffusion 시대 이후: GAN 자체의 사용이 줄어 어느 쪽이든 less critical

**선택 기준**:
- Compute budget 적음: Spectral Norm
- Mode coverage critical: WGAN-GP
- Production stability: Spectral Norm (간단, robust)
- Research / customization: WGAN-GP (flexible)

**시사점**: 두 접근이 같은 목표 (1-Lipschitz) 의 다른 enforcement. Trade-off 는 strict vs flexible, cheap vs expensive. Modern best practice 는 Spectral Norm + 약간의 GP-style regularization (e.g., R1).

</details>

---

<div align="center">

[◀ 이전 (03. Mode Collapse)](./03-mode-collapse.md) | [📚 README](../README.md) | [다음 ▶ (05. Spectral Norm)](./05-spectral-norm.md)

</div>
