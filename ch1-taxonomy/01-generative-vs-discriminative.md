# 01. Generative vs Discriminative 모델

## 🎯 핵심 질문

- Generative 모델과 discriminative 모델의 수학적 차이는 무엇인가? "$p(x)$ vs $p(y|x)$" 라는 한 줄 설명만으로 충분한가?
- Bayes 정리 $p(y|x) \propto p(x|y) p(y)$ 는 두 패러다임을 어떻게 연결하는가? Generative 가 discriminative 를 **포함**한다는 주장은 어떤 의미인가?
- 생성 모델이 수행할 수 있는 4가지 역할 (sampling, likelihood evaluation, representation learning, anomaly detection) 은 각각 어떤 수학적 객체에 의존하는가?
- 같은 데이터에서 Naive Bayes (generative) 와 Logistic Regression (discriminative) 은 왜 점근 정확도가 다르고, 작은 $n$ 에서는 누가 우월한가? (Ng & Jordan 2002)
- "분류만 잘 하면 된다" 는 실용적 관점이 있는데도 왜 generative 모델을 연구하는가?

---

## 🔍 왜 이 개념이 생성 모델 학습에 필수인가

생성 모델을 공부할 때 가장 처음 마주하는 표현은 **"discriminative 는 $p(y|x)$, generative 는 $p(x)$ 또는 $p(x, y)$ 를 학습한다"** 입니다. 이 한 문장은 정확하지만, 이 단순한 진술 뒤에 여러 결정적 결과들이 숨어 있습니다.

1. **모델 용량의 분배** — Discriminative 는 결정 경계에만 모델 용량을 할당하지만, generative 는 데이터 분포 전체를 모델링해야 합니다. 같은 파라미터 수에서 expressive 함이 다릅니다.
2. **Sample complexity 의 차이** — Ng & Jordan 2002 는 Naive Bayes (generative) 가 Logistic Regression (discriminative) 보다 **빠르게 점근 오차에 도달**함을 보였습니다. 데이터가 부족할 때 generative 가 유리한 수학적 이유입니다.
3. **재사용 가능성** — $p(x)$ 만 알면 missing-data imputation, anomaly detection, semi-supervised learning, conditional generation 등이 모두 같은 확률 모델로 처리됩니다. Discriminative 는 매 task 마다 재훈련이 필요합니다.

이 문서에서는 두 패러다임의 정의를 엄밀히 정립하고, **Bayes 정리로 generative 와 discriminative 가 연결되는 방식**, **두 접근의 sample complexity 비교**, **생성 모델이 수행하는 4가지 역할** 을 차례로 다룹니다.

---

## 📐 수학적 선행 조건

- [Probability Theory Deep Dive](https://github.com/iq-ai-lab/probability-theory-deep-dive): 결합 분포, 조건부 확률, Bayes 정리, 기댓값
- [Information Theory Deep Dive](https://github.com/iq-ai-lab/information-theory-deep-dive): KL divergence, conditional entropy, mutual information
- [Bayesian ML Deep Dive](https://github.com/iq-ai-lab/bayesian-ml-deep-dive): Posterior, MAP, marginal likelihood

---

## 📖 직관적 이해

### "고양이를 분류하는 두 가지 방법"

이미지 $x$ 가 주어졌을 때 라벨 $y \in \{$고양이, 개$\}$ 를 예측하는 두 접근법:

**Discriminative 접근**: "고양이와 개를 가장 잘 구분하는 결정 경계가 무엇인가?" — 고양이의 특징과 개의 특징을 따로 모델링하지 않고, **둘을 가르는 함수** $f: x \mapsto y$ 만 학습. 예: SVM, Logistic Regression, ResNet 분류기.

**Generative 접근**: "고양이는 어떻게 생겼는가? 개는 어떻게 생겼는가?" — 각 클래스의 분포 $p(x | y)$ 를 학습한 뒤, Bayes 정리로 분류. 예: Naive Bayes, Gaussian Discriminant Analysis, Diffusion classifier.

직관적 비유: discriminative 는 **"국경선만 그리는 사람"**, generative 는 **"각 나라의 지도를 다 그리는 사람"**. 국경선만 필요하면 discriminative 가 효율적이지만, "한국에서 새로운 도시를 상상해보라" 같은 작업은 generative 만 가능합니다.

### Generative 가 가진 추가 능력

$p(x)$ 또는 $p(x, y)$ 를 알면:

1. **Sampling**: $x \sim p(x)$ 로 새로운 데이터 생성 — 이미지 · 문장 · 음성 · 분자 구조
2. **Likelihood evaluation**: $\log p(x)$ 로 "이 데이터가 학습 분포에 얼마나 맞는가" 평가 — anomaly detection
3. **Representation learning**: latent variable $z$ 가 있는 모델에서 $p(x | z)$ 와 $p(z)$ 가 의미 있는 표현 제공 — VAE, Flow
4. **Conditional generation**: $p(x | y)$ 또는 $p(x | \text{text})$ 로 조건부 생성 — Stable Diffusion, GPT

Discriminative 모델은 이 중 어느 것도 직접 할 수 없습니다.

### 왜 Bayes 정리가 다리인가

라벨이 있는 generative 모델 $p(x, y) = p(x | y) p(y)$ 가 있다면, Bayes 정리로:

$$p(y | x) = \frac{p(x | y) p(y)}{p(x)} = \frac{p(x | y) p(y)}{\sum_{y'} p(x | y') p(y')}$$

즉, **generative 모델은 discriminative 모델을 자동으로 만들어 줍니다**. 역방향은 불가능 — $p(y | x)$ 만으로 $p(x | y)$ 를 복원할 수 없습니다 (정보 손실).

따라서 generative 는 discriminative 를 **포함**합니다. 단, 항상 그것이 효율적이지는 않다는 점이 핵심입니다.

---

## ✏️ 엄밀한 정의·정리

### 정의 1.1 — Discriminative 모델

라벨된 데이터 $\mathcal{D} = \{(x_i, y_i)\}_{i=1}^n$ 에서, 파라미터 $\theta$ 에 의해 매개변수화된 조건부 분포

$$p_\theta(y | x), \quad \theta \in \Theta$$

를 학습하는 모델을 **discriminative** 라 한다. 입력 $x$ 의 주변 분포 $p(x)$ 는 모델링하지 않는다.

**예시**: Logistic Regression $p_\theta(y = 1 | x) = \sigma(\theta^\top x)$, MLP 분류기, ResNet, BERT for classification.

### 정의 1.2 — Generative 모델

데이터 $\mathcal{D} = \{x_i\}_{i=1}^n$ (라벨 없음) 또는 $\{(x_i, y_i)\}$ 에서, 파라미터 $\theta$ 에 의해 매개변수화된 결합 분포

$$p_\theta(x) \quad \text{또는} \quad p_\theta(x, y)$$

를 학습하는 모델을 **generative** 라 한다.

**예시**:
- 라벨 없음: VAE, GAN, Normalizing Flow, Diffusion, PixelCNN
- 라벨 있음 (class-conditional): Conditional VAE, Conditional GAN, Naive Bayes, GDA

### 정리 1.3 — Bayes 정리에 의한 분류기 유도

라벨된 generative 모델 $p_\theta(x, y) = p_\theta(x | y) p_\theta(y)$ 가 있다면, Bayes 정리로

$$p_\theta(y | x) = \frac{p_\theta(x | y) p_\theta(y)}{\sum_{y'} p_\theta(x | y') p_\theta(y')}$$

가 well-defined 분류기다.

**증명**: $p_\theta(y | x) = p_\theta(x, y) / p_\theta(x)$. 분모 $p_\theta(x) = \sum_{y'} p_\theta(x, y') = \sum_{y'} p_\theta(x | y') p_\theta(y')$. $\square$

### 정의 1.4 — MLE 의 두 형태

**Generative MLE**: $\hat\theta^G = \arg\max_\theta \sum_i \log p_\theta(x_i, y_i)$ — 결합 분포의 likelihood 최대화

**Discriminative MLE (= Conditional MLE)**: $\hat\theta^D = \arg\max_\theta \sum_i \log p_\theta(y_i | x_i)$ — 조건부 likelihood 최대화

**관계**: $\log p_\theta(x, y) = \log p_\theta(y | x) + \log p_\theta(x)$. Generative MLE 는 두 항을 모두 최적화, discriminative MLE 는 둘째 항을 무시.

### 정리 1.5 — Asymptotic Bias of Naive Bayes vs Logistic Regression (Ng & Jordan 2002)

같은 conditional family (e.g., Gaussian class-conditional with shared covariance) 에서 정의된 Naive Bayes (generative) 와 Logistic Regression (discriminative) 에 대해:

1. **Asymptotic ($n \to \infty$)**: discriminative 의 분류 오차가 generative 의 그것보다 작거나 같다 (모델이 옳지 않을 때 generative 는 model misspecification 페널티 부담).

2. **Finite-sample ($n$ 작을 때)**: generative 는 $O(\log d / n)$ 으로 점근 오차에 수렴, discriminative 는 $O(\sqrt{d / n})$ 으로 수렴 ($d$ = 입력 차원). **작은 $n$ 에서는 generative 가 더 빠름**.

(증명 스케치는 다음 섹션, 자세한 유도는 원 논문 참조)

---

## 🔬 증명 및 수학적 유도

### 유도 1 — Generative ⊇ Discriminative 의 정확한 의미

라벨된 generative 모델 $p_\theta(x, y)$ 의 family 를 $\mathcal{P}^G$ 라 하고, 그로부터 Bayes 로 유도되는 discriminative 모델의 family 를 $\mathcal{P}^G_{\text{induced}} = \{p_\theta(y|x) : \theta \in \Theta\}$ 라 하자.

**관찰**: $\mathcal{P}^G_{\text{induced}} \subseteq \mathcal{P}^D$ 일 수 있지만 일반적으로 strict subset.

예: Naive Bayes Gaussian (generative) 의 induced posterior 는 항상 logistic 형태이지만, Logistic Regression (discriminative) 의 family 는 더 넓다 (제한 없는 weight). 즉:

$$\text{Naive Bayes Gaussian} \xrightarrow{\text{Bayes}} \text{LR with constrained weights} \subsetneq \text{LR}$$

따라서 generative 는 discriminative 를 **항상** 포함하지 않습니다 — induced 형태가 제약될 수 있기 때문입니다. 다만 **충분히 expressive 한 generative model** (예: VAE classifier, Diffusion classifier) 은 임의의 discriminative 를 근사할 수 있습니다.

### 유도 2 — Generative 모델의 4가지 역할의 수학적 정식화

생성 모델 $p_\theta(x)$ 가 주어졌을 때:

**1) Sampling**:
$$x \sim p_\theta(x)$$
구체적 절차는 모델마다 다름 — AR (sequential), VAE (decoder), Flow (inverse transform), GAN (push-forward), Diffusion (reverse SDE).

**2) Likelihood Evaluation**:
$$\ell(x) = \log p_\theta(x)$$
명시적으로 가능: AR, Flow. 하한만 가능: VAE, Diffusion. 불가능: GAN.

**3) Representation Learning**:
Latent variable model $p_\theta(x) = \int p_\theta(x | z) p(z) dz$ 에서 posterior $p_\theta(z | x)$ 가 representation. VAE 는 $q_\phi(z | x)$ 로 amortize.

**4) Anomaly Detection**:
임계값 $\tau$ 에 대해 $p_\theta(x_{\text{test}}) < \tau$ 이면 anomaly. 단, "데이터 매니폴드의 기하학" 과 "확률 밀도" 가 항상 일치하지 않음 (Nalisnick 2019의 OOD 역설).

### 유도 3 — Naive Bayes Sample Complexity (스케치)

Naive Bayes 는 각 class-conditional $p(x_j | y)$ 를 독립으로 가정하므로 $d$ 차원에서 $O(dK)$ 파라미터 ($K$ = class 수). 각 파라미터의 MLE 는 closed-form (counts 또는 sample mean), variance $O(1/n)$.

따라서 추정 오차는 $O(d / n)$ 의 $\ell_2$ norm, classification 오차로 환산하면 $O(\sqrt{\log d / n})$ 정도 (Hoeffding-style argument).

Logistic Regression 은 joint optimization 으로 $\theta \in \mathbb{R}^d$ 추정, central limit theorem 으로 추정 오차 $O(\sqrt{d / n})$.

비율: Naive Bayes 의 sample complexity 는 $\sqrt{\log d / n}$, LR 은 $\sqrt{d/n}$ — $d$ 가 클 때 Naive Bayes 가 훨씬 빠르게 수렴.

**단**, Naive Bayes 의 conditional independence 가정이 깨지면 asymptotic bias 발생. 따라서 "small $n$ + correct model → Naive Bayes 가 우월", "large $n$ + flexible discriminative → LR 이 우월" 의 cross-over 가 존재합니다.

---

## 💻 실험 재현 / PyTorch 구현

### 실험 1 — Naive Bayes vs Logistic Regression Sample Complexity

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.datasets import make_classification

# 동일한 데이터 생성 분포 (Gaussian, shared covariance — 두 모델 모두 well-specified)
np.random.seed(42)
n_train_list = [10, 20, 50, 100, 200, 500, 1000, 2000, 5000]
n_test = 5000
n_features = 20
n_trials = 30

errors_nb, errors_lr = [], []
for n_train in n_train_list:
    err_nb_trials, err_lr_trials = [], []
    for _ in range(n_trials):
        X, y = make_classification(
            n_samples=n_train + n_test, n_features=n_features,
            n_informative=n_features, n_redundant=0, n_clusters_per_class=1,
            class_sep=1.0
        )
        X_tr, y_tr = X[:n_train], y[:n_train]
        X_te, y_te = X[n_train:], y[n_train:]

        nb = GaussianNB().fit(X_tr, y_tr)
        lr = LogisticRegression(max_iter=1000).fit(X_tr, y_tr)

        err_nb_trials.append(1 - nb.score(X_te, y_te))
        err_lr_trials.append(1 - lr.score(X_te, y_te))

    errors_nb.append(np.mean(err_nb_trials))
    errors_lr.append(np.mean(err_lr_trials))

plt.figure(figsize=(7, 5))
plt.plot(n_train_list, errors_nb, 'o-', label='Naive Bayes (generative)')
plt.plot(n_train_list, errors_lr, 's-', label='Logistic Regression (discriminative)')
plt.xscale('log'); plt.xlabel('Training set size n')
plt.ylabel('Test error'); plt.title('Ng & Jordan 2002 재현')
plt.legend(); plt.grid(alpha=0.3); plt.show()
# 관찰: small n 에서 NB 가 LR 보다 우월, n 이 커지면 LR 이 따라잡고 추월
```

### 실험 2 — Generative 가 Discriminative 를 "포함" 함의 시각적 검증

```python
import torch
import torch.nn as nn

# 작은 generative model: GaussianMixture-style
class GenerativeClassifier:
    """p(x|y) = N(mu_y, Sigma_y), p(y) = uniform"""
    def fit(self, X, y):
        self.mus = torch.tensor([X[y == c].mean(0) for c in [0, 1]])
        self.Sigmas = torch.tensor([np.cov(X[y == c].T) + 1e-3 * np.eye(X.shape[1])
                                    for c in [0, 1]])
        self.pi = torch.tensor([0.5, 0.5])
    def log_p_x_given_y(self, x, c):
        diff = x - self.mus[c]
        prec = torch.linalg.inv(self.Sigmas[c])
        return -0.5 * (diff @ prec @ diff.T).diagonal() \
             - 0.5 * torch.logdet(self.Sigmas[c])
    def predict_proba(self, X):
        log_p0 = self.log_p_x_given_y(X, 0) + torch.log(self.pi[0])
        log_p1 = self.log_p_x_given_y(X, 1) + torch.log(self.pi[1])
        return torch.softmax(torch.stack([log_p0, log_p1], dim=-1), dim=-1)
    def sample(self, c, n=1):
        L = torch.linalg.cholesky(self.Sigmas[c])
        return self.mus[c] + torch.randn(n, self.mus.shape[-1]) @ L.T

# 동일 데이터에 GenerativeClassifier 와 LogisticRegression 적용
# → predict_proba 비교 (induced posterior 가 LR 의 특수 형태임을 확인)
# → GenerativeClassifier.sample(c) 로 새 샘플 생성 (LR 은 불가능)
```

### 실험 3 — Generative 의 4가지 역할 시연

```python
# 1) Sampling: 학습된 KDE (kernel density estimate) 로 샘플 생성
from sklearn.neighbors import KernelDensity
kde = KernelDensity(bandwidth=0.5).fit(X_train)
samples = kde.sample(100)   # 새로운 100개 샘플

# 2) Likelihood evaluation
log_p = kde.score_samples(X_test)
print(f"평균 log p(x): {log_p.mean():.3f}")

# 3) Anomaly detection
threshold = np.percentile(kde.score_samples(X_train), 5)   # 하위 5%
anomalies = X_test[kde.score_samples(X_test) < threshold]
print(f"Anomaly 개수: {len(anomalies)}")

# 4) Representation: VAE 나 Flow 의 latent (다음 챕터들에서 다룸)
```

---

## 🔗 이론과 실전의 간극

### 1. "$p(x)$ 를 정말 모델링하는가" 의 의문

GAN 이나 implicit model 은 $p(x)$ 를 명시적으로 학습하지 않고 **샘플링 가능한 분포** 만 제공합니다. 따라서 "generative 모델" 의 정의를 어떻게 할지 — sampling 가능성? density evaluation 가능성? — 은 학파마다 다릅니다.

이 레포는 **"$p(x)$ 를 명시적으로 (또는 그로부터 유도되는 객체를) 학습하는 모델"** 을 generative 로 정의합니다. 따라서:
- Explicit: AR · VAE · Flow · Diffusion (모두 likelihood 또는 lower bound 가능)
- Implicit: GAN · EBM (sampling 또는 unnormalized density 만)

### 2. "Discriminative 가 더 강력하다" 라는 통념의 함정

ImageNet 분류에서 ResNet (discriminative) 이 ResNet-VAE classifier (generative) 보다 정확도가 높습니다. 이는 사실이지만 다음을 간과합니다:

- 데이터가 많을 때 (1.2M images) 의 비교
- 분류 task 만 비교 (sampling, anomaly detection 무시)
- 모델 표현력 (parameters) 동일하지 않음

데이터가 부족할 때 (수백 ~ 수천 샘플), 또는 multi-task (분류 + 생성 + outlier 검출) 가 필요할 때, generative 가 우월할 수 있습니다.

### 3. Diffusion Classifier 의 부활 (Li et al. 2023)

Diffusion 모델 $p_\theta(x | y)$ 로 Bayes 분류기를 만든 결과, ImageNet 에서 ResNet 과 경쟁력 있고, **adversarial robustness · OOD detection 에서 우월**한 결과 (Li et al., "Your Diffusion Model is Secretly a Zero-Shot Classifier", 2023). Generative classifier 의 현대적 부활.

---

## ⚖️ 가정과 한계

| 가정 | 한계 |
|------|------|
| Generative 가 항상 discriminative 를 포함 | Induced posterior 가 제약될 수 있음 (NB → restricted LR) |
| MLE = KL 최소화 | 모델 misspecification 시 KL 이 적절한 metric 이 아닐 수 있음 (Ch1-03 에서 논의) |
| Sample complexity 비교 | 모델이 well-specified 일 때만 성립, real-world 에서는 깨질 수 있음 |
| $p(x)$ 학습 = anomaly detection | Density 가 OOD 와 항상 일치하지 않음 (Nalisnick 2019) |
| 분류만 잘 하면 충분 | 실전에서는 sampling · representation · uncertainty 가 함께 필요한 경우 多 |

---

## 📌 핵심 정리

$$\boxed{\text{Discriminative: } p_\theta(y|x), \quad \text{Generative: } p_\theta(x) \text{ 또는 } p_\theta(x, y)}$$

$$\boxed{p_\theta(y|x) = \frac{p_\theta(x|y) p_\theta(y)}{\sum_{y'} p_\theta(x|y') p_\theta(y')} \text{ — Bayes 로 generative → discriminative 유도}}$$

| 개념 | 정의 |
|------|------|
| **Discriminative** | $p_\theta(y|x)$ 만 학습, 결정 경계만 모델링 |
| **Generative** | $p_\theta(x)$ 또는 $p_\theta(x, y)$ 학습, 데이터 분포 전체 모델링 |
| **Bayes 연결** | Generative → Bayes 정리 → discriminative (역방향 불가능) |
| **MLE** | Generative: $\max \sum \log p(x, y)$, Discriminative: $\max \sum \log p(y|x)$ |
| **Ng & Jordan 2002** | Small $n$: NB > LR, Large $n$: LR ≥ NB (cross-over 존재) |
| **Generative 의 4역할** | Sampling, likelihood, representation, anomaly detection |

---

## 🤔 생각해볼 문제

**문제 1** (기초): Naive Bayes 가 $K = 2$ 클래스, $d$ 차원 binary feature 에서 사용하는 파라미터 수를 계산하라. Logistic Regression 의 파라미터 수와 비교하라.

<details>
<summary>해설</summary>

**Naive Bayes**: 각 클래스 $y \in \{0, 1\}$ 와 각 feature $x_j \in \{0, 1\}$ 에 대해 $p(x_j = 1 | y)$ 추정 — $2 \cdot d$ 개 파라미터. 추가로 prior $p(y) = \pi$ 1개. 총 **$2d + 1$ 개**.

**Logistic Regression**: $\theta_0 \in \mathbb{R}$ (bias), $\theta \in \mathbb{R}^d$ — 총 **$d + 1$ 개**.

비율: NB 가 LR 의 약 2배 파라미터를 사용합니다. 단, NB 는 각 파라미터를 **독립으로** 추정하므로 (count 만으로 closed-form), $n$ 이 작아도 안정적. LR 은 모든 파라미터를 joint optimize, $n$ 이 작으면 overfitting 위험.

</details>

**문제 2** (심화): 두 클래스 Gaussian 분포 $p(x | y = c) = \mathcal{N}(\mu_c, \Sigma)$ (shared $\Sigma$) 에서 Bayes 정리로 유도된 $p(y = 1 | x)$ 가 $\sigma(w^\top x + b)$ 형태임을 보이고, $w, b$ 를 $\mu_0, \mu_1, \Sigma$ 로 표현하라.

<details>
<summary>해설</summary>

$$p(y = 1 | x) = \frac{p(x | 1) p(1)}{p(x | 0) p(0) + p(x | 1) p(1)}$$

$\pi = p(y = 1)$ 라 하면:

$$= \frac{1}{1 + \frac{p(x|0)(1-\pi)}{p(x|1)\pi}} = \sigma\left(\log \frac{p(x|1)}{p(x|0)} + \log \frac{\pi}{1-\pi}\right)$$

Gaussian 비율의 log:

$$\log \frac{p(x|1)}{p(x|0)} = -\frac{1}{2}(x - \mu_1)^\top \Sigma^{-1}(x - \mu_1) + \frac{1}{2}(x - \mu_0)^\top \Sigma^{-1}(x - \mu_0)$$

전개하면 $x^\top \Sigma^{-1}(\mu_1 - \mu_0) - \frac{1}{2}(\mu_1 - \mu_0)^\top \Sigma^{-1}(\mu_1 + \mu_0)$ — 즉 $x$ 에 대해 **선형**.

따라서:
- $w = \Sigma^{-1}(\mu_1 - \mu_0)$
- $b = -\frac{1}{2}(\mu_1 - \mu_0)^\top \Sigma^{-1}(\mu_1 + \mu_0) + \log\frac{\pi}{1-\pi}$

이것이 GDA (Gaussian Discriminant Analysis) 가 LR 의 특수 형태임을 보여줍니다. $\square$

</details>

**문제 3** (논문 비평): Diffusion Classifier (Li et al. 2023) 는 $p_\theta(x | y)$ 로 Bayes 분류 시, $\log p_\theta(x | y)$ 를 ELBO 로 근사한다. 이때 **하한** 을 사용하는 것이 분류 정확도에 미치는 영향을 논하라. 또한 왜 같은 모델이 OOD detection 에서 ResNet 보다 우월한지 설명하라.

<details>
<summary>해설</summary>

**ELBO 사용의 영향**: $\log p_\theta(x | y) \geq \text{ELBO}(x, y)$. Bayes 분류는

$$\arg\max_y p_\theta(y | x) = \arg\max_y \log p_\theta(x | y) + \log p(y)$$

이를 ELBO 로 근사:

$$\arg\max_y \text{ELBO}(x, y) + \log p(y)$$

문제는 ELBO 의 **gap** ($\log p - \text{ELBO}$) 이 $y$ 에 따라 다를 수 있다는 점. 만약 모든 $y$ 에 대해 gap 이 동일하면 분류 결과는 정확하지만, 일반적으로 gap 이 다르므로 **bias** 가 들어갑니다. Li et al. 은 다수의 noise level $t$ 에서 평균하여 이 bias 를 줄입니다.

**OOD 우월성**: Discriminative ResNet 은 결정 경계에만 집중하므로, in-distribution 과 OOD 입력 모두에 대해 confident 한 분포를 출력 (overconfidence). Generative 모델은 $p(x | y)$ 가 OOD 에서 모두 작아지므로 **uncertainty 표현** 이 자연스럽고, $\max_y p(y | x)$ 의 절댓값으로 OOD 검출 가능. 또한 Diffusion 의 per-noise-level evaluation 이 ensemble 효과를 가져 robustness 향상.

</details>

---

<div align="center">

[◀ 이전 (README)](../README.md) | [📚 README](../README.md) | [다음 ▶ (02. Explicit vs Implicit)](./02-explicit-vs-implicit.md)

</div>
