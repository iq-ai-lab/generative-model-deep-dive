# 02. Consistency Model · Rectified Flow

## 🎯 핵심 질문

- Consistency Model (Song 2023): $f_\theta(x_t, t) \to x_0$ (모든 $t$ 에 대해 일관성) 이 어떻게 one-step generation 을 가능케 하는가?
- Consistency 의 self-consistency objective $\|f_\theta(x_t, t) - f_{\theta^-}(x_{t-1}, t-1)\|^2$ 의 의미?
- Rectified Flow (Liu 2022): $dx/dt = v_\theta(x, t)$ 의 straight trajectory 가 ODE 적분 오차를 어떻게 최소화하는가?
- Consistency Distillation vs Isolation 의 차이 — pre-trained diffusion 활용 vs 처음부터 학습?
- SD3 가 Rectified Flow 를 채택한 이유와 그 구조적 advantage?

---

## 🔍 왜 Consistency 와 Rectified Flow 가 결정적 발전인가

Diffusion 의 sampling 속도 문제 (50-1000 steps) 의 해결을 위한 두 새 패러다임 (2022-2023):

1. **Consistency Models** (Song 2023): one-step or few-step generation, distillation 또는 isolated training
2. **Rectified Flow** (Liu 2022): straight ODE trajectory, fewer NFE for same quality
3. **SD3 의 채택**: Stable Diffusion 3 (2024) 가 Rectified Flow 기반
4. **Production-ready speed**: real-time generation 가능성

이로 diffusion 의 speed bottleneck 이 GAN 수준 (single step) 으로 해소. 이 문서에서는 두 접근의 수학과 modern application 을 다룹니다.

---

## 📐 수학적 선행 조건

- 이전 챕터들 (Ch6 전체)
- [Stochastic Differential Equations Deep Dive](https://github.com/iq-ai-lab/sde-deep-dive): ODE solvers, trajectories
- Ch4-05: Neural ODE

---

## 📖 직관적 이해

### "Diffusion 의 50 step 을 1 step 으로"

Diffusion sampling: noise $x_T$ 에서 $T$ steps 거쳐 $x_0$. 각 step 이 small denoising.

**Consistency Model**: NN 이 어떤 $t$ 에서도 직접 $x_0$ 추정. $f(x_t, t) = x_0$ for all $t$.

**효과**: $T = 1$ step 으로 generation 가능. Even multi-step ($T = 4$) 으로 quality 더 향상.

### Self-Consistency Objective

NN $f_\theta(x_t, t)$ 가 모든 $t$ 에 대해 같은 $x_0$ output 해야:

$$f_\theta(x_t, t) = f_\theta(x_{t-1}, t-1)$$

(같은 trajectory 의 다른 시점에서 같은 $x_0$ 예측)

이를 **self-consistency loss** 로:

$$\mathcal{L} = \mathbb{E}\left[\|f_\theta(x_t, t) - f_{\theta^-}(\hat x_{t-1}, t-1)\|^2\right]$$

$\theta^-$ = EMA of $\theta$ (target network).

### Rectified Flow 의 직관

**Standard Diffusion ODE**:
$$\frac{dx}{dt} = f(x, t) - \frac{1}{2} g(t)^2 \nabla \log p_t(x)$$

Trajectory 가 curved — ODE 적분이 어려움 (많은 NFE 필요).

**Rectified Flow**:
$$\frac{dx}{dt} = v_\theta(x, t)$$

여기서 $v_\theta$ 가 **straight line interpolation** 의 vector field:

$$x(t) = (1 - t) x_0 + t \epsilon, \quad t \in [0, 1]$$

(linear interpolation between data $x_0$ and noise $\epsilon$).

**Vector field**: $dx/dt = \epsilon - x_0$ (constant along trajectory).

NN 이 이 constant 를 학습 — straight trajectory 의 ODE 가 single Euler step 으로 정확.

### Why Straight is Better

ODE solver 의 step error: $O(\Delta t^k)$ for $k$-th order method.

**Curved trajectory**: 작은 $\Delta t$ 에서만 정확 — 많은 step 필요.

**Straight trajectory**: 임의 $\Delta t$ 에서 정확 (Euler step 충분) — single step possible.

이론상 limit: rectified flow 가 perfect straight 면 1 step generation.

---

## ✏️ 엄밀한 정의·정리

### 정의 2.1 — Consistency Model

NN $f_\theta(x, t): \mathbb{R}^d \times [0, T] \to \mathbb{R}^d$ — "어떤 $(x_t, t)$ 에서도 $x_0$ 직접 예측".

**Self-consistency**:

$$f_\theta(x_t, t) = f_\theta(x_{t-1}, t-1) \quad \text{for } t > 0$$

(같은 trajectory point pairs 에서)

**Boundary**: $f_\theta(x_0, 0) = x_0$ (identity at $t = 0$).

### 정의 2.2 — Consistency Training (Isolation)

Pre-trained teacher 없이 처음부터 학습. Self-consistency loss:

$$\mathcal{L}_\text{CT} = \mathbb{E}_{n, x_0, \epsilon}\left[d(f_\theta(x_{t_{n+1}}, t_{n+1}), f_{\theta^-}(\hat x_{t_n}, t_n))\right]$$

여기서:
- $x_{t_{n+1}} = x_0 + t_{n+1} \epsilon$ (noisy)
- $\hat x_{t_n} = x_{t_{n+1}} - (t_{n+1} - t_n) \epsilon$ (one Euler step toward $x_0$)
- $\theta^-$ = EMA of $\theta$
- $d$ = distance (e.g., L2 또는 LPIPS)

### 정의 2.3 — Consistency Distillation

Pre-trained teacher (DDPM) 가 있을 때:

$$\hat x_{t_n} = \text{ODESolver}(x_{t_{n+1}}, t_{n+1} \to t_n)$$

(teacher 의 ODE solver 로 한 step 진행, 더 정확한 target)

Student $f_\theta$ 가 teacher 의 trajectory 를 일관성 있게 학습.

### 정의 2.4 — Sampling

**One-step**: $x_T \sim \text{prior}$, $\hat x_0 = f_\theta(x_T, T)$.

**Multi-step** (better quality): $x_T \to \hat x_0 \to$ noise back to $x_{t'} \to \hat x_0' \to ...$ — alternate refinement.

Typical: 1, 4, 8 steps.

### 정의 2.5 — Rectified Flow (Liu 2022)

데이터 $x_0 \sim p_d$ 와 noise $\epsilon \sim \mathcal{N}(0, I)$ 의 **linear interpolation**:

$$x_t = (1 - t) x_0 + t \epsilon, \quad t \in [0, 1]$$

ODE:

$$\frac{dx}{dt} = v_\theta(x, t)$$

학습:

$$\mathcal{L} = \mathbb{E}_{t, x_0, \epsilon}[\|v_\theta(x_t, t) - (\epsilon - x_0)\|^2]$$

NN 이 vector field $\epsilon - x_0$ 학습.

### 정리 2.6 — Rectified Flow 의 Straightness

이상적 case ($v_\theta = \epsilon - x_0$ exactly):

- $dx/dt = \epsilon - x_0$
- $x(t) = x(0) + t(\epsilon - x_0) = (1 - t) x_0 + t \epsilon$ ✓

**Trajectory 가 straight line** between $x_0$ and $\epsilon$ — Euler step with $\Delta t = 1$ 이 정확:

$$x_1 = x_0 + 1 \cdot (\epsilon - x_0) = \epsilon \quad ✓$$

**Reverse**: $x_0 = \epsilon - (\epsilon - x_0) = x_0$ — single step.

### 정리 2.7 — Reflow Iteration (Liu 2022)

NN learned $v_\theta$ 가 정확한 $\epsilon - x_0$ 와 약간 다름 → trajectory 가 slightly curved.

**Reflow** procedure:
1. Train $v_\theta^{(0)}$ on $(x_0, \epsilon)$ pairs
2. Generate paired data: $\epsilon \to v_\theta^{(0)}$ ODE → $x_0$
3. Re-train $v_\theta^{(1)}$ on **straight pairs** $(\epsilon, x_0)$
4. Iterate

각 reflow 가 trajectory 더 straight → fewer NFE.

---

## 🔬 증명 및 수학적 유도

### 유도 1 — Consistency 의 Boundary Condition

NN parameterization 으로 boundary $f_\theta(x_0, 0) = x_0$ 강제:

$$f_\theta(x, t) = c_\text{skip}(t) \cdot x + c_\text{out}(t) \cdot F_\theta(x, t)$$

with $c_\text{skip}(0) = 1, c_\text{out}(0) = 0$. 그러면 $f_\theta(x, 0) = x$.

EDM-style:
- $c_\text{skip}(t) = \sigma_d^2 / (\sigma(t)^2 + \sigma_d^2)$
- $c_\text{out}(t) = \sigma(t) \sigma_d / \sqrt{\sigma(t)^2 + \sigma_d^2}$

Consistency 가 다른 $t$ 에서 self-consistent 학습.

### 유도 2 — Consistency Distillation 의 정당성

Teacher trajectory: $x_t \to x_{t-\Delta t}$ (one ODE step). 만약 teacher 가 정확한 score 학습했다면, $x_{t-\Delta t}$ 도 같은 trajectory 의 점.

$x_0 = $ trajectory 의 endpoint. 따라서 student 가:

- $f_\theta(x_t, t) = x_0$
- $f_\theta(x_{t-\Delta t}, t - \Delta t) = x_0$

두 prediction 이 일치 → self-consistency 자연스럽게.

**Empirical** (Song 2023): 1-step distilled student 가 50-step teacher 와 비슷 quality.

### 유도 3 — Reflow 의 Straightness Argument

Initial $v_\theta^{(0)}$: trained on $(x_0, \epsilon)$ random pairs. 학습 가능한 ODE vector field — 일반적으로 NN 에 따라 trajectory 가 curved.

After ODE integration: $\epsilon \to x_0^{\text{est}}$. **Pair $(x_0^{\text{est}}, \epsilon)$ 가 trajectory 의 endpoint** — straight line interpolation 가능.

**Reflow** with these pairs:

$\tilde x_t = (1 - t) x_0^{\text{est}} + t \epsilon$ — known straight interpolation.

NN 가 이 straight trajectory 학습 → $v_\theta^{(1)}$ 의 trajectory 가 더 straight.

**Iterations**: convergence to actually straight pairs (maximum coupling).

### 유도 4 — Why Straight Reduces NFE

ODE Euler step: $x_{t + \Delta t} = x_t + \Delta t \cdot v(x_t, t)$.

**Curved trajectory**: $v$ 가 $x, t$ 에 따라 변화. 큰 $\Delta t$ 면 $v$ 의 변화 무시 → error.

**Straight trajectory**: $v(x, t) = $ const along trajectory → 임의 $\Delta t$ 정확.

**Empirical NFE**:
- Standard Diffusion: 50-1000 NFE
- Rectified Flow (1 reflow): 5-10 NFE
- Rectified Flow (2 reflows): 1-4 NFE
- Consistency Model: 1-4 NFE

### 유도 5 — SD3 의 Rectified Flow Choice

Stable Diffusion 3 (Esser 2024) 의 design:
1. **Latent space**: VAE compression (같은 SD 1/2)
2. **MMDiT** (Multi-Modal DiT): Transformer-based, text + image tokens
3. **Rectified Flow loss**: $\|v_\theta - (\epsilon - x_0)\|^2$
4. **No reflow** (initial 학습만으로도 충분)

**Why Rectified Flow over DDPM**:
- Fewer sampling steps (28 default vs SDXL 의 30-50)
- Better quality at same compute
- Theoretical justification (straight trajectory)

**Performance**: SD3 가 SDXL 대비 25% better human preference, similar compute.

---

## 💻 실험 재현 / PyTorch 구현

### 실험 1 — Rectified Flow 구현 (Toy 1D)

```python
import torch
import torch.nn as nn

class RectifiedFlowNet(nn.Module):
    def __init__(self, hidden=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2, hidden), nn.SiLU(),
            nn.Linear(hidden, hidden), nn.SiLU(),
            nn.Linear(hidden, 1),
        )
    def forward(self, x, t):
        t_ = t.unsqueeze(-1) if t.dim() < x.dim() else t
        return self.net(torch.cat([x, t_], dim=-1)).squeeze(-1)

v_theta = RectifiedFlowNet()
opt = torch.optim.Adam(v_theta.parameters(), lr=1e-3)

# Training: vector field = epsilon - x_0
for step in range(3000):
    x_0 = sample_target(256)   # bimodal mixture
    eps = torch.randn(256)
    t = torch.rand(256)
    x_t = (1 - t) * x_0 + t * eps
    target = eps - x_0
    pred = v_theta(x_t, t)
    loss = ((pred - target) ** 2).mean()
    opt.zero_grad(); loss.backward(); opt.step()

# 1-step sampling
def one_step_sample(v_theta, n=500):
    eps = torch.randn(n)
    t_full = torch.ones(n)
    v = v_theta(eps, t_full)
    x_0_pred = eps - v   # straight Euler from t=1 to t=0
    return x_0_pred

samples = one_step_sample(v_theta).numpy()
# Compare with bimodal target
```

### 실험 2 — Multi-Step ODE Sampling

```python
def multi_step_sample(v_theta, n=500, n_steps=10):
    x = torch.randn(n)
    dt = 1.0 / n_steps
    t = 1.0
    for _ in range(n_steps):
        v = v_theta(x, torch.full((n,), t))
        x = x - dt * v   # backward Euler
        t -= dt
    return x

# 10-step vs 1-step quality 비교
samples_1 = one_step_sample(v_theta).numpy()
samples_10 = multi_step_sample(v_theta, n_steps=10).numpy()
# Rectified Flow: 1-step 도 OK, 10-step 더 정확
```

### 실험 3 — Consistency Model (Distillation)

```python
class ConsistencyNet(nn.Module):
    """f_theta(x, t) = c_skip * x + c_out * F(x, t)"""
    def __init__(self, hidden=64):
        super().__init__()
        self.F = nn.Sequential(
            nn.Linear(2, hidden), nn.SiLU(),
            nn.Linear(hidden, hidden), nn.SiLU(),
            nn.Linear(hidden, 1),
        )
        # EDM-like c functions
        self.sigma_data = 0.5

    def c_skip(self, t):
        return self.sigma_data**2 / (t**2 + self.sigma_data**2)

    def c_out(self, t):
        return t * self.sigma_data / (t**2 + self.sigma_data**2).sqrt()

    def forward(self, x, t):
        F_out = self.F(torch.cat([x, t.unsqueeze(-1)], dim=-1)).squeeze(-1)
        return self.c_skip(t) * x + self.c_out(t) * F_out

# Distillation from teacher (assume pre-trained DDPM)
def distill(teacher_ddpm, student, opt, T=80, n_iter=10000):
    student_target = ConsistencyNet()
    # EMA target
    for step in range(n_iter):
        x_0 = sample_target(256)
        eps = torch.randn(256)
        n = torch.randint(1, T, (256,))
        t_n = sigmas(n)   # noise schedule
        t_n_1 = sigmas(n + 1)
        x_t_n_1 = x_0 + t_n_1.unsqueeze(-1) * eps
        # Teacher: one ODE step from t_{n+1} to t_n
        with torch.no_grad():
            x_t_n = teacher_ddpm.ode_step(x_t_n_1, t_n_1, t_n)
        # Student loss
        pred = student(x_t_n_1, t_n_1)
        target = student_target(x_t_n, t_n)
        loss = ((pred - target.detach()) ** 2).mean()
        opt.zero_grad(); loss.backward(); opt.step()
        # EMA update student_target
        for p, p_t in zip(student.parameters(), student_target.parameters()):
            p_t.data.mul_(0.99).add_(0.01 * p.data)

# 1-step sampling: f(eps, T) = x_0_pred
```

### 실험 4 — Consistency Training (Isolation)

```python
# Without teacher — directly learn from data + noise
def consistency_training(student, target_net, opt, n_iter=20000):
    for step in range(n_iter):
        x_0 = sample_target(256)
        eps = torch.randn(256)
        n = torch.randint(1, T, (256,))
        t_n = sigmas(n)
        t_n_1 = sigmas(n + 1)
        # x_{t_{n+1}} = x_0 + t_{n+1} eps
        x_t_n_1 = x_0 + t_n_1.unsqueeze(-1) * eps
        # Single Euler step (no teacher) — approximate trajectory
        x_t_n = x_0 + t_n.unsqueeze(-1) * eps
        # Self-consistency loss
        pred = student(x_t_n_1, t_n_1)
        target = target_net(x_t_n, t_n).detach()
        loss = ((pred - target) ** 2).mean()
        opt.zero_grad(); loss.backward(); opt.step()
        # EMA target
        for p, p_t in zip(student.parameters(), target_net.parameters()):
            p_t.data.mul_(0.99).add_(0.01 * p.data)
```

### 실험 5 — One-Step Quality Comparison

```python
# CIFAR-10 또는 toy dataset 에서:
# - Pure DDPM: 50 step FID
# - Rectified Flow (no reflow): 1 step FID
# - Rectified Flow (1 reflow): 1 step FID
# - Consistency Model (distilled): 1 step FID
# - Original GAN: 1 step FID

# 일반적 trend:
# 1-step quality: GAN ≈ Consistency ≈ Rectified Flow > LCM > LDM
# Consistency Model 이 GAN 수준에 도달 — diffusion 의 1-step generation 가능
```

---

## 🔗 이론과 실전의 간극

### 1. Latent Consistency Model (LCM)

Stable Diffusion + Consistency Model = Latent Consistency Model (Luo 2023):
- VAE encoder
- Diffusion in latent space
- Consistency distillation
- **2-4 step generation** at SD quality

이로 real-time text-to-image (수 sec → 0.1 sec).

### 2. SDXL Turbo 의 Adversarial Distillation

SDXL Turbo (Sauer 2023): adversarial diffusion distillation. Teacher diffusion + GAN-style discriminator. **1-4 step** quality + GAN-like sharpness.

### 3. Production Pipeline 의 Speed Trends

2022: SD 1.5 — 50 step, 5 sec
2023: SDXL — 30 step, 8 sec (large model)
2024: SD3 (Rectified Flow) — 28 step, 5 sec
2024: SDXL Lightning, Turbo, LCM — 1-4 step, 0.5 sec
2024+: real-time generation expected

이 trend 가 generative AI 의 production deployment 의 driver.

---

## ⚖️ 가정과 한계

| 가정 | 한계 |
|------|------|
| Consistency self-consistency 가 정확 | EMA 와 student 의 mismatch 가 source of error |
| Rectified Flow 가 straight | NN approximation 로 약간 curved |
| Reflow iteration 이 converge | 다중 reflow 의 quality 손실 가능 |
| Distillation 이 teacher quality 보존 | One-step 은 항상 약간 worse |
| 1-step 이 충분 | Multi-step 이 종종 quality 향상 |

---

## 📌 핵심 정리

$$\boxed{\text{Consistency: } f_\theta(x_t, t) = x_0 \text{ for all } t \Rightarrow \text{1-step generation}}$$

$$\boxed{\text{Rectified Flow: } x_t = (1-t) x_0 + t \epsilon, \quad v_\theta = \epsilon - x_0}$$

| Method | Steps | Quality |
|--------|-------|---------|
| **Standard DDPM** | 1000 | SOTA |
| **DDIM** | 50 | Same as DDPM |
| **DPM-Solver** | 10-20 | Same |
| **Consistency** | 1-4 | Near DDPM (distillation) |
| **Rectified Flow** | 1-10 | Better with reflow |
| **LCM** | 2-4 | SD quality at lightning speed |
| **SDXL Turbo** | 1-4 | Adversarial distillation |

| Modern Application |  |
|-------------------|---|
| **SD3** | Rectified Flow (default) |
| **DALL-E 3** | Pre-trained + distillation |
| **Real-time generation** | LCM, SDXL Turbo |
| **Mobile deployment** | Distilled diffusion |

---

## 🤔 생각해볼 문제

**문제 1** (기초): Rectified Flow 의 straight trajectory 가 single Euler step 으로 정확한 sampling 가능함을 직접 보여라.

<details>
<summary>해설</summary>

**Ideal Rectified Flow**: $v_\theta(x, t) = \epsilon - x_0$ (constant along trajectory).

**Trajectory**: $x(t) = (1 - t) x_0 + t \epsilon$.

**Single Euler step from $t = 1$ to $t = 0$** with $\Delta t = -1$:

$x(0) = x(1) + \Delta t \cdot v(x(1), 1) = \epsilon + (-1)(\epsilon - x_0) = x_0$ ✓

**핵심**: $v$ 가 trajectory 따라 constant 이므로 step size 무관 — Euler step 정확.

**Curved trajectory 의 비교**: standard diffusion 에서 $v$ 가 $x, t$ 에 따라 변화. Single Euler step:

$x(0) \approx x(1) + (-1) \cdot v(x(1), 1)$ — first-order Taylor, error $O(\Delta t^2 \cdot v')$. Large $\Delta t = 1$ 이면 error 큼.

**시사점**: trajectory geometry 가 ODE solver 의 efficiency 를 직접 결정. Straight 가 fundamental advantage.

</details>

**문제 2** (심화): Consistency Distillation 의 self-consistency objective 가 어떻게 implicitly teacher 의 trajectory 를 학습하는지 설명하라.

<details>
<summary>해설</summary>

**Teacher trajectory**: ODE $\frac{dx}{dt} = -\sigma_t s_\theta(x, t)$ 의 solution. Trajectory points $(x_t, t)$ 가 endpoint $x_0$ 으로 수렴.

**Student goal**: $f_\phi(x_t, t) = x_0$ (trajectory 의 endpoint, $t$ 와 무관).

**Self-consistency loss**:

$$\mathcal{L} = \|f_\phi(x_{t_{n+1}}, t_{n+1}) - f_{\phi^-}(\hat x_{t_n}, t_n)\|^2$$

여기서 $\hat x_{t_n}$ = teacher 의 ODE 로 한 step 진행 ($x_{t_{n+1}} \to x_{t_n}$).

**왜 작동**:

1. **Boundary condition**: $f_\phi(x_0, 0) = x_0$ (architectural constraint)
2. **Adjacent points 가 같은 endpoint**: $x_{t_n}$ 과 $x_{t_{n+1}}$ 모두 같은 teacher trajectory
3. **Self-consistency**: student 가 두 점에서 같은 $x_0$ output 해야

**Inductive learning**:
- $f_\phi(x_0, 0) = x_0$ (boundary, 자동)
- $f_\phi(x_{t_1}, t_1) \approx f_\phi(x_0, 0) = x_0$ (consistency at $t = t_1$)
- $f_\phi(x_{t_2}, t_2) \approx f_\phi(x_{t_1}, t_1) = x_0$
- ...
- $f_\phi(x_T, T) = x_0$

**결과**: 모든 $t$ 에서 $f_\phi(x_t, t) = x_0$ — 1-step generation 가능.

**EMA target $\phi^-$**: target $f_{\phi^-}$ 가 stable, student 가 이를 chase. Stability 를 위해.

**Empirical**: Song 2023 의 CIFAR-10 distillation 에서 1-step student 가 50-step teacher 의 95% quality 도달.

**시사점**: distillation 의 elegance — teacher 의 entire trajectory 를 single network 에 압축. Production 에서 매우 valuable (real-time generation).

</details>

**문제 3** (논문 비평): Stable Diffusion 3 가 Rectified Flow 를 채택한 이유를 분석하라. Standard DDPM 대비 advantages 와 potential drawbacks 는?

<details>
<summary>해설</summary>

**SD3 Architecture (Esser 2024)**:
- VAE compression (latent diffusion)
- MMDiT (Multi-Modal Diffusion Transformer)
- **Rectified Flow loss** instead of DDPM
- 28 default sampling steps

**Advantages of Rectified Flow**:

1. **Theoretical Cleanliness**:
   - Straight trajectory → simpler ODE
   - $x_t = (1-t) x_0 + t \epsilon$ — explicit formula
   - No complex schedule ($\beta_t, \bar\alpha_t$ 등)

2. **Sampling Efficiency**:
   - SD3 가 28 steps (vs SDXL 30-50 steps)
   - Better convergence per step
   - Future distillation 에 유리

3. **Loss Form**:
   - $\|v_\theta - (\epsilon - x_0)\|^2$ — symmetric in $x_0$ and $\epsilon$
   - SNR 에 robust (unlike $\epsilon$-prediction at small $\sigma$)
   - Numerically stable

4. **Empirical Quality**:
   - SD3 가 SDXL 대비 25%+ human preference
   - Text rendering 향상 (SD3 의 강점)
   - Composition complexity 처리 우수

**Potential Drawbacks**:

1. **Pretraining Compute**:
   - DDPM 의 vast literature 와 best practices 활용 못 함
   - Different schedule choices, parameterization 등 재탐색 필요

2. **Distillation Less Mature**:
   - DDPM 의 distillation (Consistency Model 등) 은 잘 연구됨
   - Rectified Flow 의 distillation 은 새 영역

3. **Quality vs Steps Trade-off**:
   - 28 step quality 가 SDXL 30-50 step 과 비슷 — speedup 마이너
   - 1-step quality 는 LCM 같은 distillation 필요

4. **Compatibility**:
   - 기존 DDPM-trained models 와 호환 어려움
   - Migration cost (model weights, fine-tuning, ControlNet 등)

**Why SD3 Made the Switch**:

1. **Long-term Roadmap**: future distillation 의 gold standard
2. **Quality at Same Compute**: clear empirical win
3. **Theoretical Foundation**: cleaner mathematical framework
4. **Differentiation**: SDXL 와 architectural distinction

**Industry Trend**:
- **Imagen 2**: still DDPM-based
- **SD3**: Rectified Flow
- **DALL-E 3**: hybrid (likely DDPM-base)
- **Sora**: Diffusion Transformer (not specified — likely DDPM-style)
- 미래: Rectified Flow 가 mainstream 가능, but 점진적

**시사점**:
- Architectural/loss choices 가 quality 의 marginal but real difference
- Open source community 의 fragmentation (DDPM-based vs Rectified Flow-based)
- Researchers 와 practitioners 의 dual investment 가 필요

**Future Direction**:
- Flow Matching (Lipman 2023) 가 Rectified Flow 의 generalization
- Continuous Time Markov Chain (CTMC) for discrete data
- Hybrid: AR + diffusion + flow matching

</details>

---

<div align="center">

[◀ 이전 (01. 5대 비교)](./01-five-families-comparison.md) | [📚 README](../README.md) | [다음 ▶ (03. EBM)](./03-ebm-revival.md)

</div>
