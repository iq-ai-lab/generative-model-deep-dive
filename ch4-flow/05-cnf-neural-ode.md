# 05. Continuous Normalizing Flow · Neural ODE · FFJORD

## 🎯 핵심 질문

- $\frac{dz}{dt} = f_\theta(z(t), t)$ 의 Neural ODE 로 무한히 많은 layer 의 limit 를 어떻게 정식화하는가?
- Density 의 시간 변화 ODE: $\frac{d \log p(z(t))}{dt} = -\text{tr}(\partial f / \partial z)$ 의 유도?
- FFJORD (Grathwohl 2019) 의 Hutchinson trace estimator 가 어떻게 $\det$ 계산을 $O(D)$ stochastic 으로 만드는가?
- Continuous flow 가 coupling/AR flow 와 비교해 어떤 architectural advantage 와 cost 를 갖는가?
- Score-based diffusion model 이 어떻게 CNF 의 framework 안에서 통합되는가?

---

## 🔍 왜 Continuous Flow 가 결정적 발전인가

Discrete flow (RealNVP, Glow, MAF) 의 제약:
- Coupling: 절반 unchanged
- AR: sequential
- 모든 layer 가 specific architectural form

**Continuous Normalizing Flow** (Chen 2018 Neural ODE, Grathwohl 2019 FFJORD): NN 이 ODE 의 vector field $f_\theta$ 가 됨.

$$\frac{dz(t)}{dt} = f_\theta(z(t), t)$$

이로 얻는 것:
1. **Free-form NN**: $f_\theta$ 가 임의 architecture (no coupling, no mask)
2. **무한 depth limit**: 연속 시간 $t$ 가 layer 의 일반화
3. **Invertibility 자동**: ODE 의 backward integration
4. **Score matching 과 통합**: Diffusion 의 forward/reverse SDE 가 ODE 한계
5. **FFJORD**: Hutchinson trace 로 efficient density

이 framework 가 **Diffusion 의 mathematical foundation** 의 일부. Score-SDE (Ch6-04) 가 같은 ODE/SDE framework 에서 통합. 이 문서에서는 CNF 의 수학과 FFJORD 의 trace estimator, 그리고 diffusion 과의 연결을 다룹니다.

---

## 📐 수학적 선행 조건

- 이전 문서들: 01-change-of-variables, 02-realnvp, 04-maf-iaf
- [Stochastic Differential Equations Deep Dive](https://github.com/iq-ai-lab/sde-deep-dive): ODE, SDE, Itô calculus
- [Linear Algebra Deep Dive](https://github.com/iq-ai-lab/linear-algebra-deep-dive): Trace, Jacobian

---

## 📖 직관적 이해

### "ResNet 의 limit 가 ODE"

ResNet block: $h_{\ell+1} = h_\ell + f_\theta(h_\ell)$. $\Delta t = 1$ 의 Euler step 으로 보면:

$$h_{\ell+1} = h_\ell + \Delta t \cdot f_\theta(h_\ell)$$

$\Delta t \to 0$, layer 수 무한 → ODE:

$$\frac{dh}{dt} = f_\theta(h(t), t)$$

이것이 Neural ODE 의 직관 — ResNet 의 continuous limit.

### "Density 의 시간 진화"

분포 $p(z(t), t)$ 가 시간에 따라 진화. ODE $dz/dt = f$ 하에서:

$$\frac{d \log p(z(t))}{dt} = -\text{tr}\left(\frac{\partial f}{\partial z}\right)$$

이는 instantaneous version of change-of-variables. Discrete flow 의 $\log |\det J_f|$ 가 continuous 에서 $\int \text{tr}(\partial f / \partial z) dt$ 가 됨.

### Hutchinson Trace Estimator

$\text{tr}(A)$ 의 직접 계산: $O(D)$ for diagonal, $O(D^2)$ for off-diagonal access. 큰 $D$ 에서 cost 큼.

**Hutchinson** (1990): $\text{tr}(A) = \mathbb{E}_\epsilon[\epsilon^\top A \epsilon]$ for $\epsilon \sim \mathcal{N}(0, I)$ (또는 Rademacher).

**증명**: $\mathbb{E}[\epsilon^\top A \epsilon] = \mathbb{E}[\sum_{ij} \epsilon_i A_{ij} \epsilon_j] = \sum_{ij} A_{ij} \mathbb{E}[\epsilon_i \epsilon_j] = \sum_{ij} A_{ij} \delta_{ij} = \sum_i A_{ii} = \text{tr}(A)$.

**FFJORD**: $\epsilon^\top \partial f / \partial z \cdot \epsilon$ 을 계산 — JVP (Jacobian-vector product) 1회로 $O(D)$. Stochastic 추정 (variance vs cost trade-off).

### CNF 의 표현력

Discrete flow 의 architectural restriction 없음 — $f_\theta$ 가 임의 NN. 따라서 **이론상 더 expressive**.

**Cost trade-off**:
- Forward: ODE solver (RK4, adaptive) → multiple NN evaluations
- Density: Hutchinson + ODE → expensive
- Discrete flow 보다 **5-10배 느림**

이것이 CNF 가 mainstream image generation 의 architecture 가 되지 못한 이유. Diffusion 이 같은 framework 에서 더 효율적 (no Jacobian 직접 계산).

---

## ✏️ 엄밀한 정의·정리

### 정의 5.1 — Neural ODE (Chen 2018)

NN $f_\theta: \mathbb{R}^D \times [0, T] \to \mathbb{R}^D$ 로 정의된 ODE:

$$\frac{dz(t)}{dt} = f_\theta(z(t), t), \quad z(0) = z_0$$

해 $z(T)$ 는 ODE solver (Euler, RK4, Dopri5 등) 로 numerically 구함.

**Forward**: $z_0 \to z(T) = x$ — base 에서 데이터로.
**Inverse**: $x \to z_0$ — same ODE 를 reverse direction 으로 integrate.

### 정리 5.2 — Density 의 ODE (Continuous Change of Variables)

$z(t) \sim p(z(t), t)$ 가 ODE $dz/dt = f(z, t)$ 따른다면:

$$\frac{d \log p(z(t), t)}{dt} = -\text{tr}\left(\frac{\partial f}{\partial z}(z(t), t)\right)$$

**증명** (Chen 2018, Appendix A):

continuity equation (mass preservation):

$$\frac{\partial p}{\partial t} + \nabla \cdot (p \cdot f) = 0$$

(particle density 의 보존)

Total derivative along trajectory:

$$\frac{dp(z(t), t)}{dt} = \frac{\partial p}{\partial t} + \nabla p \cdot \frac{dz}{dt} = \frac{\partial p}{\partial t} + \nabla p \cdot f$$

continuity equation 으로 $\partial p / \partial t = -\nabla \cdot (pf) = -p (\nabla \cdot f) - \nabla p \cdot f$. 대입:

$$\frac{dp}{dt} = -p(\nabla \cdot f) - \nabla p \cdot f + \nabla p \cdot f = -p \cdot \text{tr}(\partial f / \partial z)$$

(divergence = trace of Jacobian)

따라서 $d \log p / dt = (1/p) dp/dt = -\text{tr}(\partial f / \partial z)$. $\square$

### 정리 5.3 — CNF 의 Likelihood

ODE $dz/dt = f_\theta(z, t)$ on $[0, T]$ with base $z(0) \sim p_Z = \mathcal{N}(0, I)$ and $z(T) = x$:

$$\log p_X(x) = \log p_Z(z(0)) - \int_0^T \text{tr}\left(\frac{\partial f_\theta}{\partial z}(z(t), t)\right) dt$$

**계산**: ODE solver 가 augmented system

$$\frac{d}{dt}\begin{pmatrix} z \\ \ell \end{pmatrix} = \begin{pmatrix} f(z, t) \\ -\text{tr}(\partial f / \partial z) \end{pmatrix}, \quad \ell(0) = 0$$

$\ell(T) = -\int_0^T \text{tr} \, dt$. 따라서 $\log p_X(x) = \log p_Z(z(0)) + \ell(T)$.

### 정리 5.4 — Hutchinson Trace Estimator

$A \in \mathbb{R}^{D \times D}$, $\epsilon \sim \mathcal{N}(0, I_D)$ (또는 Rademacher $\pm 1$):

$$\text{tr}(A) = \mathbb{E}_\epsilon[\epsilon^\top A \epsilon]$$

**Variance**: Rademacher $\epsilon$ 가 Gaussian 보다 lower variance (특히 diagonal $A$ 에서 zero variance for Rademacher).

### 정리 5.5 — FFJORD 의 Stochastic Density

$\partial f / \partial z \in \mathbb{R}^{D \times D}$ 의 trace 를 Hutchinson 으로:

$$\text{tr}(\partial f / \partial z) \approx \epsilon^\top \frac{\partial f}{\partial z} \epsilon$$

**계산**: $\epsilon^\top (\partial f / \partial z)$ 가 vector-Jacobian product (VJP), PyTorch autograd 로 $O(D)$. 다음 $\cdot \epsilon$ 도 $O(D)$.

따라서 trace estimation cost $O(D)$ — full $\det$ 의 $O(D^3)$ 대비 큰 절약.

**Trade-off**: stochastic estimator 의 variance → 학습 시 noisy gradient. 일반적으로 OK.

### 정의 5.6 — Adjoint Method for Backprop

ODE 의 backprop 직접 계산은 memory-prohibitive (모든 intermediate state 저장).

**Adjoint** (Pontryagin): adjoint state $a(t) = \partial L / \partial z(t)$ 가 ODE 따름:

$$\frac{da}{dt} = -a^\top \frac{\partial f}{\partial z}$$

Backward 도 ODE solve — **constant memory** in time. Chen 2018 의 핵심 효율 기여.

---

## 🔬 증명 및 수학적 유도

### 유도 1 — ResNet → Neural ODE 의 수렴

ResNet: $h_{\ell+1} = h_\ell + g_\theta(h_\ell)$. 매 layer $\Delta t = 1$ 의 Euler step.

$\Delta t \to 0$ 의 limit: $L \to \infty$ layers, total time $T = L \cdot \Delta t$ fixed:

$$h_{\ell+1} - h_\ell = \Delta t \cdot g_\theta(h_\ell) / \Delta t \cdot \Delta t$$

만약 $g$ 가 layer 마다 다르고 $g_\theta(h, t = \ell \Delta t)$ 로 보면:

$$\frac{dh}{dt} = f_\theta(h, t), \quad f_\theta = g_\theta(h, t) / \Delta t$$

(scaling 의 detail 무시) — Neural ODE.

**시사점**: ResNet 의 깊은 limit 가 ODE — 무한 depth 와 free-form vector field.

### 유도 2 — Hutchinson 의 Variance

$A$ 가 diagonal: $\epsilon^\top A \epsilon = \sum_i a_i \epsilon_i^2$. $\text{Var} = \sum_i a_i^2 \cdot \text{Var}(\epsilon_i^2)$.

- Gaussian $\epsilon$: $\text{Var}(\epsilon_i^2) = 2$ → $\text{Var} = 2 \sum_i a_i^2$
- Rademacher $\epsilon \in \{\pm 1\}$: $\epsilon_i^2 = 1$ always → $\text{Var} = 0$ for diagonal $A$

**시사점**: Rademacher 가 diagonal 에서 zero variance. General $A$ 에서 Gaussian 보다 lower variance. FFJORD 가 Rademacher 사용.

### 유도 3 — Adjoint Sensitivity Method

Loss $L = L(z(T))$. $\partial L / \partial \theta$ 계산.

**Forward**: $z(0) \to z(T)$ via ODE.

**Adjoint**: $a(T) = \partial L / \partial z(T)$. $a(t)$ 의 ODE:

$$\frac{da}{dt} = -a^\top \frac{\partial f_\theta}{\partial z}$$

**증명**:

$L = L(z(T))$. $z(T)$ 의 perturbation 이 $z(t)$ 의 perturbation 의 ODE-evolved version.

Variation $\delta z(t)$ 의 evolution:

$$\frac{d \delta z}{dt} = \frac{\partial f}{\partial z} \delta z$$

Sensitivity: $\partial L / \partial z(t) = a(t)$ such that $\delta L = a(t)^\top \delta z(t)$. 미분:

$$\frac{d}{dt}(a^\top \delta z) = 0 \quad (\delta L \text{ time-invariant})$$

$\dot a^\top \delta z + a^\top \dot{\delta z} = 0$. $\dot{\delta z} = (\partial f / \partial z) \delta z$:

$$\dot a^\top + a^\top (\partial f / \partial z) = 0 \Rightarrow \dot a = -(\partial f / \partial z)^\top a$$

**Theta gradient**: $\partial L / \partial \theta = -\int_0^T a^\top (\partial f / \partial \theta) dt$.

**Memory advantage**: forward state 저장 안 해도 됨 ($z(t)$ 를 reverse 로 reconstruct), $a(t)$ 만 backward integrate.

### 유도 4 — Diffusion 과의 통합

**Score-SDE** (Song 2021, Ch6-04): forward SDE $dx = f(x, t) dt + g(t) dW$. Reverse SDE:

$$dx = [f(x, t) - g(t)^2 \nabla_x \log p_t(x)] dt + g(t) d\bar W$$

**Probability Flow ODE** (deterministic counterpart):

$$\frac{dx}{dt} = f(x, t) - \frac{1}{2} g(t)^2 \nabla_x \log p_t(x)$$

이 ODE 의 trajectory 가 같은 marginals 를 가짐 (Anderson 1982). **Score-based model = CNF with score $\nabla \log p_t$ 가 vector field**.

따라서 diffusion 의 **deterministic sampling** = CNF, **stochastic sampling** = SDE. 같은 framework 의 두 극.

### 유도 5 — CNF 의 Universal Approximation

ODE $dz/dt = f_\theta$ 의 trajectory 는 임의 diffeomorphism $\phi: \mathbb{R}^D \to \mathbb{R}^D$ 를 근사 가능 (Dupont 2019, with caveats).

**Augmented Neural ODE** (Dupont 2019): $z$ 차원에 추가 dim ($\hat z$) 도입 → 표현력 증가. 일반 CNF 가 표현 못 하는 분포 (e.g., 1D unmixing) 도 augmented 에서 가능.

**시사점**: CNF 가 universal 이지만 architecture 에 따라 표현력 한계. Augmented, multi-stage 등의 변형으로 보완.

---

## 💻 실험 재현 / PyTorch 구현

### 실험 1 — Neural ODE with `torchdiffeq`

```python
import torch
import torch.nn as nn
from torchdiffeq import odeint

class ODEFunc(nn.Module):
    def __init__(self, dim=2, hidden=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim + 1, hidden), nn.Tanh(),
            nn.Linear(hidden, hidden), nn.Tanh(),
            nn.Linear(hidden, dim),
        )

    def forward(self, t, z):
        # t: scalar, z: [batch, dim]
        t_ = t.expand(z.shape[0], 1)
        return self.net(torch.cat([z, t_], dim=-1))

# Forward: integrate from t=0 to t=1
ode_func = ODEFunc(dim=2)
z0 = torch.randn(64, 2)
t = torch.linspace(0, 1, 2)
solution = odeint(ode_func, z0, t, method='dopri5')
z_T = solution[-1]   # [64, 2]
```

### 실험 2 — FFJORD 의 Hutchinson Trace

```python
def hutchinson_trace(f, z, t, n_samples=1):
    """tr(∂f/∂z) ≈ E_ε[ε^T (∂f/∂z) ε]"""
    z = z.requires_grad_(True)
    f_val = f(t, z)
    trace = 0
    for _ in range(n_samples):
        # Rademacher ε
        eps = torch.randint(0, 2, z.shape, device=z.device).float() * 2 - 1
        # JVP: ∂f/∂z @ ε
        jvp = torch.autograd.grad(f_val, z, grad_outputs=eps, create_graph=True)[0]
        trace = trace + (eps * jvp).sum(-1)
    return trace / n_samples

class CNF(nn.Module):
    def __init__(self, dim=2, hidden=64):
        super().__init__()
        self.f = ODEFunc(dim, hidden)

    def forward(self, x, reverse=False):
        # x: data → z (reverse=False) or z → data (reverse=True)
        # Density: integrate (z, log_p) augmented system
        def aug_func(t, state):
            z = state[..., :-1]
            f_val = self.f(t, z)
            tr = hutchinson_trace(self.f, z, t)
            return torch.cat([f_val, -tr.unsqueeze(-1)], dim=-1)

        log_p_init = torch.zeros(x.shape[0], 1, device=x.device)
        state0 = torch.cat([x, log_p_init], dim=-1)

        t = torch.tensor([0.0, 1.0])
        if reverse:
            t = t.flip(0)
        sol = odeint(aug_func, state0, t, method='dopri5')
        z_final = sol[-1, :, :-1]
        log_det = sol[-1, :, -1]   # accumulated -tr
        return z_final, log_det

    def log_p(self, x):
        z, log_det = self.forward(x)
        log_p_z = -0.5 * z.pow(2).sum(-1) - 0.5 * z.shape[-1] * np.log(2*np.pi)
        return log_p_z + log_det

# 학습 (2-moon)
model = CNF(dim=2, hidden=64)
opt = torch.optim.Adam(model.parameters(), lr=1e-3)
# ... training loop ...
```

### 실험 3 — Trajectory 시각화

```python
# Base distribution → data 의 trajectory
@torch.no_grad()
def trajectory(model, n_steps=50):
    z = torch.randn(500, 2)
    t = torch.linspace(0, 1, n_steps)

    def f_only(t, z):
        return model.f(t, z)

    sol = odeint(f_only, z, t, method='dopri5')
    return sol   # [n_steps, batch, dim]

traj = trajectory(model)
# t=0: Gaussian, t=1: 2-moons
# 중간 시간 에서의 분포 시각화 — 점진적 변형
```

### 실험 4 — Discrete vs Continuous Flow Cost 비교

```python
# 같은 데이터 (2-moons) 에 RealNVP 와 CNF 학습
# Forward time, density evaluation time 비교

import time

# RealNVP
realnvp = RealNVP(dim=2, n_layers=10)
t0 = time.time()
for _ in range(100):
    log_p = realnvp.log_p(torch.randn(64, 2))
print(f"RealNVP density: {time.time() - t0:.3f}s")

# CNF (same data)
cnf = CNF(dim=2, hidden=64)
t0 = time.time()
for _ in range(100):
    log_p = cnf.log_p(torch.randn(64, 2))
print(f"CNF density:     {time.time() - t0:.3f}s")
# 일반: CNF 가 5-20배 느림 (ODE solve cost)
```

---

## 🔗 이론과 실전의 간극

### 1. ODE Solver 의 Cost

Adaptive solver (Dopri5, RK45) 의 NFE (Number of Function Evaluations):
- 단순 분포: 20-50 NFE
- 복잡한 분포: 100-1000 NFE
- 각 NFE = NN forward pass

**비교**:
- Discrete flow: $L$ layer = $L$ forward pass (constant)
- CNF: NFE depends on problem difficulty

이것이 CNF 가 production 에서 흔하지 않은 이유 — variable cost.

### 2. Free-Form vs Architectural Constraint 의 Trade-off

**CNF 의 free-form**:
- Any NN architecture
- Theoretical expressiveness 우월

**Discrete flow 의 constraint**:
- Coupling, AR pattern
- Less expressive per layer
- 그러나 **predictable cost** (constant), **stable training**

**현대적 동향**: Diffusion 이 score-based parameterization 으로 유사한 free-form (no Jacobian 직접 계산) → CNF 의 efficiency 한계 회피.

### 3. Score-Based Generative Models 와의 통합

CNF 의 vector field $f_\theta$ 가 score $\nabla \log p_t$ 와 관련:

**Probability Flow ODE** (Song 2021, Ch6-04):
$$\frac{dx}{dt} = f(x, t) - \frac{1}{2} g(t)^2 \nabla_x \log p_t(x)$$

이 ODE 의 학습은 score $\nabla \log p_t$ 학습 (denoising score matching) — **Jacobian 직접 계산 회피**.

따라서 **diffusion = special CNF with score field**, training 이 더 효율적.

---

## ⚖️ 가정과 한계

| 가정 | 한계 |
|------|------|
| ODE solver 가 정확 | Stiff ODE 또는 큰 vector field 에서 numerical 어려움 |
| Hutchinson trace 가 unbiased | High variance — 학습 noisy |
| Adjoint method memory-efficient | Numerical error 누적 가능 |
| Free-form NN 이 expressive | Augmented (Dupont) 가 일부 case 에서 필요 |
| ODE 의 $T$ fixed | $T$ 가 hyperparameter, tuning 필요 |
| Continuous data | Discrete 는 dequantization |

---

## 📌 핵심 정리

$$\boxed{\frac{dz(t)}{dt} = f_\theta(z(t), t), \quad \frac{d \log p(z(t))}{dt} = -\text{tr}\left(\frac{\partial f}{\partial z}\right)}$$

$$\boxed{\text{tr}(A) = \mathbb{E}_\epsilon[\epsilon^\top A \epsilon] \quad — \text{Hutchinson estimator, } O(D)}$$

| Component | 역할 |
|-----------|------|
| **Neural ODE (Chen 2018)** | $f_\theta$ 가 free-form vector field |
| **FFJORD (Grathwohl 2019)** | + Hutchinson trace, $O(D)$ density |
| **Adjoint method** | Constant-memory backprop |
| **ODE solver** | Dopri5, RK45 등 — variable NFE |
| **Probability Flow ODE** | Score field → Diffusion 의 deterministic limit |

| 비교 | Discrete Flow | CNF |
|------|---------------|-----|
| Architecture | Coupling/AR pattern 강제 | Free-form NN |
| Forward cost | $O(L)$ constant | $O(\text{NFE})$ variable |
| Density cost | $O(L)$ | $O(\text{NFE} \cdot D)$ via trace |
| Memory | $O(L)$ | $O(1)$ via adjoint |
| Expressiveness | Restricted | Universal (in limit) |
| Training stability | High | Lower (stochastic trace) |

---

## 🤔 생각해볼 문제

**문제 1** (기초): $f_\theta(z, t) = A z$ ($A$ constant matrix) 의 ODE solution 을 explicit 으로 구하라. $\log p$ 의 변화는?

<details>
<summary>해설</summary>

**ODE**: $dz/dt = Az$. Linear ODE solution: $z(t) = e^{tA} z(0)$ (matrix exponential).

**Density**: $-\text{tr}(\partial f / \partial z) = -\text{tr}(A) = $ constant.

$\log p$ 의 변화: $\int_0^T -\text{tr}(A) dt = -T \cdot \text{tr}(A)$.

Equivalently:
$$\log p(z(T)) = \log p(z(0)) - T \cdot \text{tr}(A)$$

$z(T) = e^{TA} z(0)$ — linear flow.

**Discrete flow 와의 일치**: $f(z) = e^{TA} z$ 의 Jacobian = $e^{TA}$. $\log |\det e^{TA}| = T \cdot \text{tr}(A)$ (matrix exponential 의 determinant = exp(trace)). ✓

**시사점**: Linear case 에서 CNF 가 정확히 standard linear flow. Non-linear $f$ 에서 더 expressive.

</details>

**문제 2** (심화): Hutchinson estimator $\hat{\text{tr}} = \epsilon^\top A \epsilon$ 의 variance 를 Rademacher $\epsilon$ 와 Gaussian $\epsilon$ 에 대해 비교하라. Diagonal $A$ 에서 왜 Rademacher 가 zero variance 인가?

<details>
<summary>해설</summary>

**$A$ 가 diagonal**, $A = \text{diag}(a_1, \ldots, a_D)$:

$\hat{\text{tr}} = \sum_i a_i \epsilon_i^2$. 

**Rademacher** $\epsilon_i \in \{\pm 1\}$: $\epsilon_i^2 = 1$ always → $\hat{\text{tr}} = \sum_i a_i = \text{tr}(A)$. **Constant, Variance = 0**.

**Gaussian** $\epsilon_i \sim \mathcal{N}(0, 1)$: $\epsilon_i^2 \sim \chi^2_1$, $\mathbb{E}[\epsilon_i^2] = 1, \text{Var}(\epsilon_i^2) = 2$. $\text{Var}(\hat{\text{tr}}) = 2 \sum_i a_i^2$.

**General $A$ (off-diagonal)**:

$\hat{\text{tr}} = \sum_{ij} A_{ij} \epsilon_i \epsilon_j = \text{tr}(A) + \sum_{i \neq j} A_{ij} \epsilon_i \epsilon_j$.

- Rademacher: $\mathbb{E}[\epsilon_i \epsilon_j] = 0$ for $i \neq j$ (independent), so unbiased. Variance from cross terms $\sum_{i \neq j} A_{ij}^2$.
- Gaussian: same unbiased, but additional variance from $\epsilon_i^2 - 1$.

**일반적**: Rademacher 가 lower variance — FFJORD 의 default choice.

**시사점**: trace estimator 의 variance 가 Jacobian 의 off-diagonal 강도 에 의존. Sparse Jacobian 에서 estimator 빠름. Dense 에서는 더 많은 sample 필요 (variance reduction).

</details>

**문제 3** (논문 비평): CNF (FFJORD) 가 discrete flow 보다 expressive 하지만 production 에서는 거의 사용되지 않음. Diffusion 이 같은 ODE/SDE framework 에서 SOTA 인 이유를 비교 분석하라.

<details>
<summary>해설</summary>

**CNF 의 한계**:
1. **ODE solver cost**: NFE = 50-1000, 각 NN forward pass — 매우 비쌈
2. **Hutchinson variance**: stochastic trace estimator → noisy gradient → unstable training
3. **Direct $\det J$ tracking**: Jacobian computation 이 핵심 cost
4. **Memory vs accuracy trade-off**: adjoint method 의 numerical error

**Diffusion 의 우월성**:
1. **Score parameterization**: $\nabla \log p_t$ 직접 학습 → **Jacobian 계산 우회**
2. **Denoising score matching**: tractable per-step loss, no ODE solve at training
3. **Discretized chain**: $T$ steps fixed, predictable cost
4. **Probability Flow ODE**: deterministic sampling = same architecture as CNF, but score-based learning
5. **Stable training**: per-step supervised denoising, no Jacobian gradient noise

**같은 framework, 다른 학습**:
- CNF: vector field $f_\theta$ 직접, density via $\det J$ tracking
- Diffusion: score $\nabla \log p_t$, vector field 가 $f_\text{drift} - \frac{1}{2} g^2 \nabla \log p$

**Diffusion 의 sampling**:
- Stochastic (SDE): forward/reverse SDE, multi-step
- Deterministic (Probability Flow ODE): same as CNF, but **score-trained**

**시사점**:
- CNF 의 문제는 vector field 학습 시 density tracking 의 cost
- Diffusion 이 score 의 indirect learning 으로 회피
- Same theoretical framework, different empirical efficiency

**현대적 동향**:
- **Flow Matching** (Lipman 2023): vector field 의 다른 학습 방법, denoising 과 유사
- **Rectified Flow** (Liu 2022): straight trajectory 로 cost 감소
- **Stable Diffusion 3**: Rectified Flow + DiT — Flow 의 부활

**CNF 의 잔존 응용**:
- Density estimation (anomaly detection)
- Bayesian inference (Flow as variational posterior)
- 과학적 응용 (molecular flow, lattice gauge)

**요약**: CNF 의 universal expressiveness 가 한계는 학습 cost. Diffusion 이 같은 framework 에서 score-based shortcut 으로 우월. Flow 는 niche 응용에서, 또는 diffusion 의 ODE 형태로 (Probability Flow) 살아있음.

</details>

---

<div align="center">

[◀ 이전 (04. MAF/IAF)](./04-maf-iaf.md) | [📚 README](../README.md) | [다음 ▶ (Ch5-01. GAN Minimax)](../ch5-gan/01-minimax-formulation.md)

</div>
