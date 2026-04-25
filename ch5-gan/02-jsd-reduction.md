# 02. 최적 D 에서 V 의 JSD 환원 증명

## 🎯 핵심 질문

- 최적 $D^*(x) = p_d(x) / (p_d(x) + p_g(x))$ 를 $V(D, G)$ 에 대입하면 정확히 $2 \cdot JSD(p_d \| p_g) - \log 4$ 가 되는가?
- 이 환원 증명의 각 step 의 의미는? 왜 JSD 가 자연스럽게 등장하는가?
- $V(D^*, G^*) = -\log 4$ 가 GAN 의 minimum value 인 이유? $G^*$ 에서 $p_g = p_d$?
- JSD 가 0 ⟺ $p_g = p_d$ 의 함의 — generator 가 정확히 데이터 분포 학습?
- 이 환원 결과가 mode collapse, vanishing gradient 의 수학적 원인을 어떻게 설명하는가?

---

## 🔍 왜 이 환원이 결정적인가

Goodfellow 2014 의 핵심 정리: optimal $D$ 에서 GAN minimax 가 **JSD divergence 최소화** 와 동치. 이 환원이 우리에게 주는 통찰:

1. **Theoretical foundation** — GAN 이 well-defined divergence 를 최소화함을 증명
2. **Unique optimum** — JSD 가 0 ⟺ $p_g = p_d$, 따라서 minimax 의 해가 unique
3. **Mode collapse 의 원인** — JSD 의 non-overlapping support 에서 gradient 0 (Ch5-03)
4. **WGAN 의 motivation** — JSD 의 한계가 Wasserstein 으로 대체

이 단순한 대입과 산수 — 그러나 그 결과가 GAN 의 모든 후속 분석의 기반. 이 문서에서는 JSD 환원의 정확한 증명을 단계별로 다룹니다.

---

## 📐 수학적 선행 조건

- 이전 문서: 01-minimax-formulation.md
- [Information Theory Deep Dive](https://github.com/iq-ai-lab/information-theory-deep-dive): KL divergence, JSD definition
- Ch1-03: KL minimization unification

---

## 📖 직관적 이해

### "JSD 가 자연스럽게 등장하는 이유"

JSD 의 정의:

$$JSD(p, q) = \frac{1}{2} \text{KL}(p \| m) + \frac{1}{2} \text{KL}(q \| m), \quad m = (p + q)/2$$

JSD = symmetric "평균과의 KL" 의 average. Optimal $D^*$ 가 $p_d / (p_d + p_g) = p_d / (2m)$ 형태 — 정확히 mixture $m$ 에 대한 normalized ratio.

따라서 $V(D^*, G)$ 의 두 expectation 이 자연스럽게 KL with $m$ 로 환원 → JSD.

### 환원 증명의 단계

1. $D^*$ 대입: $\log D^* = \log(p_d / (p_d + p_g))$, $\log(1 - D^*) = \log(p_g / (p_d + p_g))$
2. Expectation 의 합: $\mathbb{E}_{p_d}[\log(2 p_d / 2m)] + \mathbb{E}_{p_g}[\log(2 p_g / 2m)]$
3. 분리: $\log(p_d / m) - \log 2$ 형태로
4. KL 인식: $\mathbb{E}_{p_d}[\log(p_d / m)] = \text{KL}(p_d \| m)$
5. 합치기: $\text{KL}(p_d \| m) + \text{KL}(p_g \| m) - 2 \log 2 = 2 \text{JSD} - \log 4$

각 step 이 산수 - 핵심은 "**$D^*$ 의 형태가 mixture $m$ 의 ratio 라는 것**".

### Minimum Value 의 의미

$V(D^*, G)$ 가 $G$ 의 함수. JSD $\geq 0$, JSD = 0 ⟺ $p_g = p_d$. 따라서:

$$\min_G V(D^*, G) = 0 - \log 4 = -\log 4 \approx -1.386$$

이 minimum 은 $p_g = p_d$ 일 때만 도달. **유일** 한 optimal generator (분포 측면).

---

## ✏️ 엄밀한 정의·정리

### 정리 2.1 — JSD 환원 (Goodfellow 2014)

Fixed $G$ 에 대해 optimal $D^* = p_d / (p_d + p_g)$ 를 $V(D, G)$ 에 대입하면:

$$V(D^*, G) = 2 \cdot JSD(p_d \| p_g) - \log 4$$

여기서 $JSD(p, q) = \frac{1}{2} \text{KL}(p \| m) + \frac{1}{2} \text{KL}(q \| m)$, $m = (p + q)/2$.

### 증명 (정리 2.1)

$D^* = p_d / (p_d + p_g)$. 그러면 $1 - D^* = p_g / (p_d + p_g)$.

$V$ 의 두 항 대입:

$$V(D^*, G) = \mathbb{E}_{p_d}[\log D^*] + \mathbb{E}_{p_g}[\log(1 - D^*)]$$

$$= \mathbb{E}_{p_d}\left[\log \frac{p_d}{p_d + p_g}\right] + \mathbb{E}_{p_g}\left[\log \frac{p_g}{p_d + p_g}\right]$$

분모를 $2m$ ($m = (p_d + p_g)/2$) 로 표기:

$$= \mathbb{E}_{p_d}\left[\log \frac{p_d}{2m}\right] + \mathbb{E}_{p_g}\left[\log \frac{p_g}{2m}\right]$$

$$= \mathbb{E}_{p_d}\left[\log \frac{p_d}{m}\right] - \mathbb{E}_{p_d}[\log 2] + \mathbb{E}_{p_g}\left[\log \frac{p_g}{m}\right] - \mathbb{E}_{p_g}[\log 2]$$

$\mathbb{E}_{p}[\log 2] = \log 2$ (constant). 두 KL 항:

$$\mathbb{E}_{p_d}[\log(p_d / m)] = \text{KL}(p_d \| m)$$

$$\mathbb{E}_{p_g}[\log(p_g / m)] = \text{KL}(p_g \| m)$$

따라서:

$$V(D^*, G) = \text{KL}(p_d \| m) + \text{KL}(p_g \| m) - 2 \log 2$$

JSD 의 정의 $JSD(p_d \| p_g) = \frac{1}{2}[\text{KL}(p_d \| m) + \text{KL}(p_g \| m)]$ 사용:

$$V(D^*, G) = 2 \cdot JSD(p_d \| p_g) - 2 \log 2 = 2 \cdot JSD(p_d \| p_g) - \log 4 \quad \square$$

### 정리 2.2 — Minimum of $V(D^*, G)$

$$\min_G V(D^*, G) = -\log 4$$

도달 시 $p_g = p_d$.

**증명**: $JSD \geq 0$, $JSD = 0 \iff p_d = p_g$ (information theory 의 표준 결과). 따라서 $V(D^*, G) \geq 2 \cdot 0 - \log 4 = -\log 4$, equality iff $p_g = p_d$. $\square$

### 정리 2.3 — Optimal Strategy 의 Existence and Uniqueness

GAN minimax problem $\min_G \max_D V(D, G)$ 의 **distribution-level** optimum:

- **Optimal $D$**: $D^*(x) = p_d / (p_d + p_g)$ — 각 $G$ 에 대해 unique (a.e.)
- **Optimal $G$**: $p_g = p_d$ — distribution 으로 unique

(NN parameter level 에서는 서로 다른 $\theta$ 가 같은 $p_g$ 줄 수 있음 — non-uniqueness)

### 정리 2.4 — Saturation of JSD

$p_d, p_g$ 의 support 가 disjoint 이면:

$$JSD(p_d \| p_g) = \log 2 \quad (\text{constant})$$

**증명**: disjoint support 에서 $p_d \cdot p_g = 0$ everywhere. $m = (p_d + p_g)/2$. $p_d > 0$ 인 곳: $m = p_d / 2$, $p_d / m = 2$. $p_g > 0$ 인 곳: $p_g / m = 2$.

$$\text{KL}(p_d \| m) = \int_{p_d > 0} p_d \log 2 \, dx = \log 2$$

마찬가지로 $\text{KL}(p_g \| m) = \log 2$. 따라서 $JSD = (1/2)(\log 2 + \log 2) = \log 2$. $\square$

### 함의 — Vanishing Gradient

JSD = $\log 2$ (constant) ⟹ $V(D^*, G) = 2 \log 2 - \log 4 = 0$ (constant in $G$) ⟹ $\nabla_G V = 0$.

**Generator 학습 불가능** — gradient 0. 이것이 GAN training 의 fundamental 문제, WGAN 의 motivation (Ch5-04).

---

## 🔬 증명 및 수학적 유도

### 유도 1 — JSD 의 다른 표현

JSD 는 Shannon entropy 로도 표현 가능:

$$JSD(p, q) = H\left(\frac{p + q}{2}\right) - \frac{1}{2}[H(p) + H(q)]$$

여기서 $H$ 는 entropy.

**증명**:

$$JSD = \frac{1}{2}[\text{KL}(p \| m) + \text{KL}(q \| m)]$$

$$= \frac{1}{2}\left[\int p \log p - \int p \log m + \int q \log q - \int q \log m\right]$$

$$= \frac{1}{2}\left[-H(p) - \int p \log m - H(q) - \int q \log m\right]$$

$\int p \log m + \int q \log m = \int (p + q) \log m = 2 \int m \log m = -2 H(m)$:

$$JSD = \frac{1}{2}[-H(p) - H(q)] + H(m) = H(m) - \frac{1}{2}[H(p) + H(q)] \quad \square$$

이 형태는 JSD 의 "mixture entropy 와 average entropy 의 차이" 라는 직관 — mixing 으로 information 손실 (entropy 증가).

### 유도 2 — JSD ≤ log 2 의 Bound

$$JSD(p, q) \leq \log 2$$

**증명**: $H(m) \leq \log |\text{support}(m)|$ (max entropy bound). $H(p), H(q) \geq 0$.

또는 alternatively:

$\text{KL}(p \| m) = \int p \log(p / m) \leq \int p \log(p / (p/2)) = \int p \log 2 = \log 2$ (since $m \geq p/2$).

같은 방법으로 $\text{KL}(q \| m) \leq \log 2$. 따라서 $JSD \leq \log 2$. $\square$

**Saturation**: equality 는 disjoint support 에서.

### 유도 3 — $V(D^*, G^*) = -\log 4$ 의 직접 검증

$p_g = p_d \Rightarrow D^* = p_d / (p_d + p_d) = 1/2$ everywhere.

$V(D^*, G^*) = \mathbb{E}_{p_d}[\log(1/2)] + \mathbb{E}_{p_d}[\log(1/2)] = 2 \log(1/2) = -2 \log 2 = -\log 4$. ✓

### 유도 4 — JSD 의 Distance Property

$\sqrt{JSD}$ 는 metric (distance):
- $\sqrt{JSD}(p, p) = 0$
- $\sqrt{JSD}(p, q) = \sqrt{JSD}(q, p)$
- Triangle inequality

이 property 가 JSD 를 "useful divergence" 로 만듦. KL 은 metric 아님 (asymmetric).

**시사점**: GAN 이 JSD 최소화 → "데이터와 모델 분포 사이의 distance" 를 줄이는 well-defined 의미.

### 유도 5 — Two Mode 분포의 JSD

예: $p_d = $ bimodal Gaussian (modes at $\pm 1$), $p_g = $ unimodal (mode at $-1$).

- Mode at $-1$: $p_d > 0, p_g > 0$, both have mass — $m \approx (p_d + p_g)/2$
- Mode at $+1$: $p_d > 0, p_g \approx 0$ — $m \approx p_d / 2$, $\text{KL}(p_d | m) \approx \log 2$ (locally)
- 따라서 JSD $\approx (\log 2)/2$ at the missing mode

**해석**: $p_g$ 가 mode 하나 빠뜨리면 JSD 가 그 부분에서 $\log 2$ contribute. $p_g$ 가 모든 mode cover 해야 JSD 작아짐 → mode coverage 의 incentive.

---

## 💻 실험 재현 / PyTorch 구현

### 실험 1 — JSD 환원 수치 검증

```python
import numpy as np

def kl_divergence(p, q):
    """KL(p || q) for discrete distributions"""
    eps = 1e-12
    return np.sum(p * np.log((p + eps) / (q + eps)))

def jsd(p, q):
    m = 0.5 * (p + q)
    return 0.5 * kl_divergence(p, m) + 0.5 * kl_divergence(q, m)

# Setup: 1D discrete (binned)
x_grid = np.linspace(-5, 5, 200)
p_d = np.exp(-0.5 * (x_grid - 1) ** 2); p_d /= p_d.sum()
p_g = np.exp(-0.5 * (x_grid + 0.5) ** 2); p_g /= p_g.sum()

# 1. Direct V(D*, G)
D_star = p_d / (p_d + p_g + 1e-12)
V_direct = (p_d * np.log(D_star + 1e-12)).sum() \
         + (p_g * np.log(1 - D_star + 1e-12)).sum()

# 2. Via JSD
V_jsd = 2 * jsd(p_d, p_g) - np.log(4)

print(f"V(D*, G) direct:  {V_direct:.6f}")
print(f"2*JSD - log4:    {V_jsd:.6f}")
# 두 값이 일치 — 정리 검증
```

### 실험 2 — 분포의 거리에 따른 JSD 변화

```python
import matplotlib.pyplot as plt

distances = np.linspace(0, 10, 50)
jsds = []
kls = []
v_optimal = []

for d in distances:
    p_d = np.exp(-0.5 * x_grid ** 2); p_d /= p_d.sum()
    p_g = np.exp(-0.5 * (x_grid - d) ** 2); p_g /= p_g.sum()

    jsds.append(jsd(p_d, p_g))
    kls.append(kl_divergence(p_d, p_g))
    v_optimal.append(2 * jsd(p_d, p_g) - np.log(4))

plt.subplot(1, 2, 1)
plt.plot(distances, jsds, label='JSD')
plt.plot(distances, [np.log(2)] * len(distances), 'k--', label='log 2 (saturation)')
plt.xlabel('Distance between modes')
plt.ylabel('Divergence')
plt.legend(); plt.title('JSD saturates at log 2')

plt.subplot(1, 2, 2)
plt.plot(distances, kls, label='KL')
plt.xlabel('Distance')
plt.ylabel('KL')
plt.title('KL diverges to ∞')
plt.legend()
plt.show()
# JSD: 거리 커지면 log 2 에 saturation
# KL: 거리 커지면 무한대로 발산
```

### 실험 3 — Disjoint Support 에서 JSD 의 saturation

```python
# 두 점 분포 (Dirac 의 discrete approximation)
def disjoint_supports(d, sigma=0.01, n=1000):
    x = np.linspace(-2, 2 + d, n)
    p_d = np.exp(-0.5 * (x ** 2) / sigma ** 2); p_d /= p_d.sum()
    p_g = np.exp(-0.5 * ((x - 1 - d) ** 2) / sigma ** 2); p_g /= p_g.sum()
    return p_d, p_g

distances = [0.1, 1.0, 5.0, 10.0]
for d in distances:
    p_d, p_g = disjoint_supports(d)
    j = jsd(p_d, p_g)
    print(f"Distance d = {d}: JSD = {j:.4f} (log 2 = {np.log(2):.4f})")
# d 가 충분히 크면 JSD ≈ log 2 (saturation)
```

### 실험 4 — Optimal G 도달 검증

```python
# Trainable: μ_g, σ_g of Gaussian generator
import torch
import torch.optim as optim

class SimpleG:
    def __init__(self, mu=2.0, sigma=2.0):
        self.mu = torch.tensor(mu, requires_grad=True)
        self.log_sigma = torch.tensor(np.log(sigma), requires_grad=True)
    def density(self, x):
        sigma = self.log_sigma.exp()
        return (-0.5 * ((x - self.mu) / sigma) ** 2 - self.log_sigma - 0.5 * np.log(2*np.pi)).exp()

# Target
mu_d, sigma_d = 0.0, 1.0   # ideal G at this
G = SimpleG(mu=2.0, sigma=2.0)

x = torch.linspace(-5, 5, 200)
p_d = (-0.5 * (x - mu_d).pow(2) - 0.5 * np.log(2*np.pi)).exp()
p_d /= p_d.sum()

opt = optim.Adam([G.mu, G.log_sigma], lr=0.01)
for step in range(2000):
    p_g = G.density(x); p_g /= p_g.sum()
    m = 0.5 * (p_d + p_g)
    j = 0.5 * (p_d * (p_d / (m + 1e-12)).log()).sum() \
      + 0.5 * (p_g * (p_g / (m + 1e-12)).log()).sum()
    loss = j   # minimize JSD
    opt.zero_grad(); loss.backward(); opt.step()

print(f"Final μ_g: {G.mu.item():.3f} (target: {mu_d})")
print(f"Final σ_g: {G.log_sigma.exp().item():.3f} (target: {sigma_d})")
print(f"Final JSD: {j.item():.6f}")
# 수렴: μ_g → 0, σ_g → 1, JSD → 0
```

---

## 🔗 이론과 실전의 간극

### 1. Theoretical Optimum vs Practical Result

이론: $\min_G V(D^*, G) = -\log 4$ at $p_g = p_d$.

실전:
- $D$ 가 정확한 $D^*$ 도달 못 함 (capacity, optimization)
- $G$ 의 NN parameter 가 모든 $p_g$ 표현 불가
- Gradient 가 ideal minimax 와 다른 dynamics
- Mode collapse 등 문제

### 2. JSD Saturation 의 실전적 영향

$p_d, p_g$ 가 거의 disjoint 일 때 (훈련 초기 random $G$):
- $V(D^*, G) \approx 2 \log 2 - \log 4 = 0$
- $\nabla_G V \approx 0$

**현실 mitigations**:
- Non-saturating loss (Ch5-01)
- Spectral normalization, gradient penalty
- Wasserstein distance (Ch5-04)

### 3. Mode Collapse 의 수학적 시그널

$p_g$ 가 일부 mode 만 cover 하면:
- 그 mode 에서 JSD 작은 contribution
- 안 cover 한 mode 에서 큰 contribution (JSD 의 disjoint support 부분)

따라서 JSD 가 "일부 mode 만 우월" 한 generator 에 강하게 페널티. 그러나 **gradient signal** 이 약함 (JSD saturation, vanishing).

---

## ⚖️ 가정과 한계

| 가정 | 한계 |
|------|------|
| Optimal $D^*$ 도달 가능 | 실제 NN $D$ 는 capacity + optimization limit |
| JSD 최소화 가 GAN 의 학습 | Non-saturating loss 는 다른 divergence |
| $p_g = p_d$ 가 unique optimum | NN parameter level 에서 non-unique |
| Distribution overlap 충분 | Disjoint 면 JSD saturated, gradient 없음 |
| Continuous distribution | Discrete 에서도 비슷, manifold 구조 다름 |

---

## 📌 핵심 정리

$$\boxed{V(D^*, G) = 2 \cdot JSD(p_d \| p_g) - \log 4}$$

$$\boxed{\min_G V(D^*, G) = -\log 4 \Leftrightarrow p_g = p_d \Leftrightarrow JSD = 0}$$

| 양 | 값 |
|----|------|
| $D^*$ | $p_d / (p_d + p_g)$ |
| $V(D^*, G)$ | $2 JSD - \log 4 \in [-\log 4, 0]$ |
| Minimum | $-\log 4$ at $p_g = p_d$ |
| Maximum (disjoint) | $0$ (JSD = log 2 saturation) |

| 결과 | 의미 |
|------|------|
| **JSD ≥ 0**, $= 0 \iff p_d = p_g$ | Unique optimum, distance-like |
| **JSD ≤ $\log 2$** | Saturation in disjoint support |
| **$\sqrt{JSD}$** = metric | Symmetric distance |
| **Mode collapse** | $p_g$ 가 일부 mode 만 → JSD large at missing modes |
| **Vanishing gradient** | Disjoint support → $\nabla_G V \approx 0$ |

---

## 🤔 생각해볼 문제

**문제 1** (기초): $D^*$ 를 $V$ 에 직접 대입하는 위 증명 의 각 step 을 다시 보고, 각 step 에서 사용된 정의/항등식이 무엇인지 명시하라.

<details>
<summary>해설</summary>

**Step 1**: $D^* = p_d / (p_d + p_g)$, $1 - D^* = p_g / (p_d + p_g)$. — Definition of optimal D (이전 챕터).

**Step 2**: $\log D^* = \log p_d - \log(p_d + p_g) = \log p_d - \log(2m) = \log(p_d / m) - \log 2$. — Logarithm 의 분배 + $m$ 의 정의.

**Step 3**: 같은 방식으로 $\log(1 - D^*) = \log(p_g / m) - \log 2$.

**Step 4**: 두 expectation 합:
$$\mathbb{E}_{p_d}[\log(p_d / m)] - \log 2 + \mathbb{E}_{p_g}[\log(p_g / m)] - \log 2$$

**Step 5**: KL definition: $\mathbb{E}_{p}[\log(p / m)] = \text{KL}(p \| m)$.

**Step 6**: $\text{KL}(p_d \| m) + \text{KL}(p_g \| m) - 2 \log 2$.

**Step 7**: JSD definition: $JSD(p, q) = \frac{1}{2}[\text{KL}(p \| m) + \text{KL}(q \| m)]$. → $\text{KL} + \text{KL} = 2 \cdot JSD$.

**Step 8**: $V(D^*, G) = 2 \cdot JSD - 2 \log 2 = 2 \cdot JSD - \log 4$.

**핵심 항등식**:
- Optimal D 의 형태
- $m = (p + q)/2$ 의 mixture 정의
- KL 과 expectation 의 정의
- JSD 가 KL 의 합의 절반이라는 정의

</details>

**문제 2** (심화): $JSD(p, q) \leq \log 2$ 의 saturation bound 가 GAN training 의 vanishing gradient 와 어떻게 연결되는지 설명하라.

<details>
<summary>해설</summary>

**Saturation**: $JSD = \log 2$ 일 때 (disjoint support).

**$V$ 의 값**: $V(D^*, G) = 2 \log 2 - \log 4 = 0$ (saturated).

**Generator gradient**:
$$\nabla_\theta V(D^*, G_\theta) = \nabla_\theta [2 \cdot JSD(p_d \| p_{g_\theta}) - \log 4]$$

만약 $p_g$ 가 작은 perturbation 받아도 still disjoint with $p_d$ → JSD remains $\log 2$ → $\nabla = 0$.

**실전적 시나리오**:
1. **훈련 초기**: $G$ 가 random — $p_g$ support 가 noise (주로 데이터 manifold 와 disjoint)
2. **$D$ 가 너무 강함**: $D$ 가 거의 perfect classifier ($D \approx 0$ for fake, $D \approx 1$ for real) → 사실상 disjoint 처럼 동작
3. **High-dim manifold mismatch**: image data 의 manifold 가 매우 thin, generator manifold 와 보통 not overlapping in high-dim

각 경우 JSD saturated → gradient 약함 → 학습 정체.

**해결책**:
1. **Non-saturating loss**: 다른 divergence 형태 사용 (Ch5-01)
2. **Inject noise**: $D$ input 에 noise 추가하여 distribution support 확장
3. **Spectral norm**: $D$ 의 Lipschitz 제약 → too-strong $D$ 막음
4. **Wasserstein** (Ch5-04): support mismatch 에 robust 한 Wasserstein-1

**시사점**: JSD-based GAN 의 **fundamental limitation**. 이론적으로 좋은 objective 이지만 high-dim NN training 에서 gradient signal 부족. WGAN 의 motivation.

</details>

**문제 3** (논문 비평): JSD 환원 결과는 "GAN 이 well-defined divergence 를 최소화한다" 는 GAN 의 정당성을 증명. 하지만 실전에서 GAN 의 generator 가 종종 JSD optimum 도달 못 함. 어떤 architectural 또는 training 한계가 이 gap 을 만드는가?

<details>
<summary>해설</summary>

**Theoretical guarantee**: minimax 의 unique optimum 이 $p_g = p_d$, JSD = 0.

**Practical gaps**:

**1. Discriminator 의 Capacity Limit**:
- 이론: $D^*(x) = p_d / (p_d + p_g)$ — exact posterior
- 실전: NN $D$ 가 finite capacity, $D^*$ 를 정확히 표현 못 함
- 결과: $V(D, G)$ 가 정확히 $V(D^*, G)$ 가 아님 — JSD 와의 환원 부정확

**2. Optimization 의 Local Issues**:
- Minimax 가 saddle point — local 이 아닌 global
- Alternating SGD 가 항상 minimax 의 정확한 해로 수렴하지 않음
- Mode collapse 는 local minimum 의 manifestation

**3. Generator 의 Architectural Constraint**:
- $G_\theta$ 의 family 가 임의 $p_g$ 표현 못 함
- $z \sim p_z$ (Gaussian) 에서 $G(z)$ 는 manifold 에 push-forward
- High-dim image 의 모든 mode 를 단일 NN 으로 cover 어려움

**4. Mini-batch + Stochastic Gradients**:
- 이론은 expectation 위에서, 실전은 mini-batch
- Variance 가 mini-batch size 에 의존
- BatchNorm 의 영향 (training/inference mode 차이)

**5. Vanishing/Exploding Gradient**:
- JSD saturation 시 gradient 0
- $D$ 가 너무 강하면 generator gradient 약함
- $D$ 가 너무 약하면 informative signal 부족

**6. Non-Identifiability**:
- 다른 NN parameter 가 같은 $p_g$ 줄 수 있음
- 학습 trajectory 가 다양한 equivalent solutions 사이 jumping 가능

**해결 방향** (후속 챕터):
- **Spectral Norm**: $D$ 의 Lipschitz 제약 (Ch5-05)
- **WGAN**: Wasserstein distance 사용 (Ch5-04)
- **Progressive growing**: 단계적 학습 (Ch5-06)
- **Self-attention**: long-range dependencies for global coherence

**시사점**: theoretical optimum 의 존재는 GAN 의 well-defined-ness 를 보장. 그러나 그 optimum 도달은 별도의 architectural + training 의 도전. 후속 GAN family 의 모든 발전이 이 gap 을 줄이는 시도.

</details>

---

<div align="center">

[◀ 이전 (01. Minimax)](./01-minimax-formulation.md) | [📚 README](../README.md) | [다음 ▶ (03. Mode Collapse)](./03-mode-collapse.md)

</div>
