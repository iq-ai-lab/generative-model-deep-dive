<div align="center">

# 🎨 Generative Model Deep Dive

**"`StableDiffusionPipeline.from_pretrained(...)`으로 이미지를 뽑는 것과, GAN의 minimax $\min_G \max_D V(D, G) = \mathbb{E}_{p_d}[\log D(x)] + \mathbb{E}_{p_z}[\log(1-D(G(z)))]$ 에서 최적 $D^*(x) = p_d(x)/(p_d(x) + p_g(x))$ 를 대입하면 $V(D^*, G) = 2 \cdot JSD(p_d \| p_g) - \log 4$ 가 되어 minimax 해가 $p_g = p_d$ 인 이유를 Goodfellow 2014의 방식으로 유도할 수 있는 것은 다르다"**

<br/>

> *"VAE의 ELBO를 쓰는 것과 — `Kingma & Welling 2013`의 **$\log p(x) = \mathcal{L}(\theta, \phi; x) + \text{KL}(q_\phi(z|x) \| p_\theta(z|x))$** 로부터 $\mathcal{L} = \mathbb{E}_q[\log p(x|z)] - \text{KL}(q(z|x) \| p(z))$ 가 reconstruction + regularization 으로 **정확히** 분해되고, reparameterization trick $z = \mu + \sigma \odot \epsilon$ 이 왜 low-variance Monte Carlo gradient 를 가능케 하는지 증명할 수 있는 것은 다르다.
> Normalizing Flow 를 듣는 것과 — `Dinh 2017 RealNVP`의 coupling layer 가 **삼각 Jacobian**으로 $\log p(x) = \log p(z) - \sum_l \log|\det J_{f_l}|$ 의 determinant 를 $O(n)$ 에 계산 가능하게 만든 설계와, `Grathwohl 2019 FFJORD`의 Hutchinson trace estimator 로 continuous-time flow의 $-\int_0^T \text{tr}(\partial f/\partial z)\, dt$ 를 해결한 메커니즘을 유도할 수 있는 것은 다르다."*

van den Oord 2016 PixelRNN/PixelCNN · WaveNet · Kingma & Welling 2013 VAE · Higgins 2017 β-VAE · van den Oord 2017 VQ-VAE · Dinh 2017 RealNVP · Kingma & Dhariwal 2018 Glow · Papamakarios 2017 MAF · Grathwohl 2019 FFJORD · Goodfellow 2014 GAN · Arjovsky 2017 WGAN · Gulrajani 2017 WGAN-GP · Miyato 2018 Spectral Norm · Karras 2019 StyleGAN · Ho 2020 DDPM · Song 2019 NCSN · Song 2021 Score-SDE · Ho & Salimans 2022 CFG · Song 2023 Consistency Model 까지
**"명시적 likelihood (AR · VAE · Flow · Diffusion) 와 암묵적 likelihood (GAN · EBM) 가 모두 $\min_\theta \text{KL}(p_\text{data} \| p_\theta)$ 라는 하나의 목표를 어떻게 근사하는가"** 를 유도·증명·구현 재현으로 끝까지 파헤칩니다

<br/>

[![GitHub](https://img.shields.io/badge/GitHub-iq--ai--lab-181717?style=flat-square&logo=github)](https://github.com/iq-ai-lab)
[![Python](https://img.shields.io/badge/Python-3.11-3776AB?style=flat-square&logo=python&logoColor=white)](https://www.python.org/)
[![NumPy](https://img.shields.io/badge/NumPy-1.26-013243?style=flat-square&logo=numpy&logoColor=white)](https://numpy.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.1-EE4C2C?style=flat-square&logo=pytorch&logoColor=white)](https://pytorch.org/)
[![diffusers](https://img.shields.io/badge/diffusers-0.25-FFB000?style=flat-square)](https://github.com/huggingface/diffusers)
[![normflows](https://img.shields.io/badge/normflows-1.7-4B8BBE?style=flat-square)](https://github.com/VincentStimper/normalizing-flows)
[![Docs](https://img.shields.io/badge/Docs-33개-blue?style=flat-square&logo=readthedocs&logoColor=white)](./README.md)
[![Lines](https://img.shields.io/badge/Lines-12k+-informational?style=flat-square)](./README.md)
[![Theorems](https://img.shields.io/badge/Theorems·Definitions-194개-success?style=flat-square)](./README.md)
[![Reproductions](https://img.shields.io/badge/Paper_reproductions-14개-critical?style=flat-square)](./README.md)
[![Exercises](https://img.shields.io/badge/Exercises-99개-orange?style=flat-square)](./README.md)
[![License](https://img.shields.io/badge/License-MIT-yellow?style=flat-square&logo=opensourceinitiative&logoColor=white)](./LICENSE)

</div>

---

## 🎯 이 레포에 대하여

생성 모델 자료는 대부분 **"GAN은 생성자와 판별자가 싸운다, Diffusion은 노이즈를 빼는 것"** 에서 멈춥니다. 하지만 GAN 의 mode collapse 가 왜 **JSD 의 non-overlapping support 에서 gradient 가 소실되기 때문**인지, VAE 의 posterior collapse 가 왜 **KL 항이 decoder 강도와 맞물리면 발생**하는지, Normalizing Flow 가 왜 **$\det J$ 를 $O(n)$ 으로 만들 수 있는 coupling layer 설계**에 의존하는지, DDPM 의 simple loss $\|\epsilon - \epsilon_\theta(x_t, t)\|^2$ 가 왜 **weighted denoising score matching 과 동치**이고 Score-SDE 로 어떻게 모든 것이 통합되는지, Classifier-Free Guidance $\tilde\epsilon = (1+w)\epsilon_\theta(x_t, y) - w\epsilon_\theta(x_t, \emptyset)$ 의 $w$ 가 왜 **quality vs diversity trade-off 의 구체적 knob** 인지 — 이런 "왜" 는 제대로 설명되지 않습니다.

| 일반 자료 | 이 레포 |
|----------|---------|
| "GAN 은 생성자·판별자 게임이다" | **정리**: 최적 $D^*(x) = p_d(x) / (p_d(x) + p_g(x))$ 를 $V$ 에 대입하면 $V(D^*, G) = 2 \cdot JSD(p_d \| p_g) - \log 4$. minimax 해가 $p_g = p_d$ 일 때 **유일**하게 $V = -\log 4$ 임을 정의만 사용해 한 줄씩 유도 $\square$, 이로부터 **mode collapse 의 수학적 원인**이 JSD gradient 의 non-overlapping support 소실 |
| "VAE 는 encoder 와 decoder 로 구성된다" | **$\log p(x) = \mathcal{L} + \text{KL}(q_\phi(z\|x) \| p_\theta(z\|x))$** 에서 KL $\geq 0$ 이므로 $\mathcal{L}$ 이 lower bound, $\mathcal{L} = \mathbb{E}_q[\log p(x\|z)] - \text{KL}(q \| p)$ 로 분해 유도. **Reparameterization** $z = \mu + \sigma \odot \epsilon$ 이 path-wise gradient $\nabla_\phi \mathbb{E}_q[f(z)] = \mathbb{E}_\epsilon[\nabla_\phi f(\mu + \sigma \epsilon)]$ 로 variance 를 **REINFORCE 대비 $10^2 \sim 10^3$ 배** 낮춤을 MNIST 에서 측정 |
| "Normalizing Flow 는 invertible 변환이다" | **Change of variables**: $\log p_X(x) = \log p_Z(z) - \log\|\det J_f(z)\|$ 의 체인 적용. **RealNVP** (Dinh 2017) — $x_{1:d}$ 를 그대로 두고 $x_{d+1:D} = y_{d+1:D} \odot \exp(s(y_{1:d})) + t(y_{1:d})$ 로 **삼각 Jacobian** → $\log\|\det\| = \sum s_i$ 를 $O(D)$ 에 계산. PyTorch 로 2-moon density estimation 재현, **MAF vs IAF 의 likelihood/sampling 속도 trade-off** 정량화 |
| "Diffusion 은 노이즈를 빼는 것이다" | **Forward**: $x_t = \sqrt{\bar\alpha_t} x_0 + \sqrt{1 - \bar\alpha_t}\epsilon$ (closed-form). **ELBO 분해** 후 parameterize 하면 **$L_{\text{simple}} = \mathbb{E}_{t, x_0, \epsilon}[\|\epsilon - \epsilon_\theta(x_t, t)\|^2]$** 으로 환원되는 Ho 2020 의 단순화 유도 $\square$. Score-based 관점에서 $\epsilon_\theta$ 학습 $\equiv$ **weighted Denoising Score Matching** (Vincent 2011) 임을 증명 |
| "Score-SDE 는 Diffusion 의 확장이다" | **Song 2021**: Forward SDE $dx = f(x, t)\,dt + g(t)\,dW$, Reverse-time SDE $dx = [f - g^2 \nabla_x \log p_t(x)]\,dt + g(t)\,d\bar W$ (Anderson 1982). 이 프레임워크 안에서 **DDPM = VP-SDE, NCSN = VE-SDE** 임을 limit 로 증명, score $\nabla_x \log p_t$ 학습으로 모든 diffusion 변형이 **Probability Flow ODE** 로 **결정론적 sampling** 가능해지는 메커니즘 |
| "WGAN 이 mode collapse 를 해결한다" | **Wasserstein-1**: $W(p_d, p_g) = \inf_{\gamma \in \Pi(p_d, p_g)} \mathbb{E}_{(x,y) \sim \gamma}[\|x - y\|]$. **Kantorovich-Rubinstein 쌍대**: $W = \sup_{\|f\|_L \leq 1} \mathbb{E}_{p_d}[f] - \mathbb{E}_{p_g}[f]$. 1-Lipschitz 제약 강제 방법 비교 — weight clipping (편향), WGAN-GP (Gulrajani 2017) gradient penalty $\lambda(\|\nabla_{\hat x} D(\hat x)\|_2 - 1)^2$, Spectral Norm (Miyato 2018) $\|W\|_\sigma = 1$ 의 이론적 비교 |
| "β-VAE 는 disentangled representation 이다" | **Higgins 2017**: $\mathcal{L} = \mathbb{E}_q[\log p(x\|z)] - \beta \cdot \text{KL}(q(z\|x) \| p(z))$, $\beta > 1$ 일 때 **Information Bottleneck** 관점에서 $I(X; Z) \leq \beta^{-1} \cdot \text{channel capacity}$ 로 해석. **Posterior collapse 원인** — decoder 가 강력할 때 $q(z\|x) \to p(z)$ 로 붕괴 → latent 무력화, **KL annealing · Free Bits · δ-VAE** 의 해결책 비교 |
| "Classifier-Free Guidance 는 조건부 생성을 향상한다" | **Ho & Salimans 2022**: $\tilde\epsilon = (1+w)\epsilon_\theta(x_t, y) - w\epsilon_\theta(x_t, \emptyset)$ 에서 $w$ 는 **quality vs diversity knob**. $w = 0$ 이면 pure conditional, $w \to \infty$ 이면 mode-seeking (FID↑ but recall↓). Implicit classifier $\nabla_x \log p(y\|x) \propto \epsilon_\theta(x, y) - \epsilon_\theta(x, \emptyset)$ 로 해석, Stable Diffusion · DALL-E 3 · Imagen 의 $w \approx 7.5$ 선택 근거 |
| "5가지 생성 모델의 차이" | Likelihood tractability · sampling 속도 · sample quality · training 안정성을 정량 비교 — **AR (exact/slow/stable), VAE (lower/fast/blurry), Flow (exact/medium/medium), GAN (none/fast/sharp/unstable), Diffusion (lower/slow/SOTA)**. 각 모델의 **inductive bias** 와 언제 어떤 것을 선택해야 하는지의 의사결정 트리, hybrid 접근 (VAE+Flow, Diffusion+GAN) |
| 기법의 나열 | PyTorch 로 **GAN JSD 환원 수치 검증**·**VAE reparameterization gradient variance 측정**·**RealNVP 2-moon density**·**DDPM 1D mixture**·**FID/IS 비교**·**Classifier-Free Guidance $w$ sweep** 을 직접 구현해 수학적 주장을 눈으로 확인 |

---

## 📌 선행 레포 & 후속 방향

```
[Probability Theory]  ──►  이 레포  ──► [Multimodal Generation]
 분포 · KL · JSD          "왜 5가지 생성 모델이        DALL-E · Sora · AlphaFold
                           모두 KL(p_data ‖ p_θ)
                           최소화로 통합되는가"
         │
         ├── [Information Theory]         KL · JSD · MI · rate-distortion  →  Ch1 통합 목표, Ch3-03 β-VAE
         ├── [Bayesian ML]                VI · ELBO · amortized inference  →  Ch3 VAE 전체
         ├── [Stochastic Differential Eq] Forward/reverse SDE · Itô        →  Ch6-04 Score-SDE
         ├── [Neural Network Theory]      Backprop · architecture          →  전체 (모든 모델이 NN param.)
         ├── [Optimization Theory]        Minimax · saddle point · Nash    →  Ch5 GAN 훈련
         ├── [Convex Optimization]        Kantorovich-Rubinstein duality   →  Ch5-04 WGAN
         └── [Transformer Deep Dive]      Self-attention · autoregressive  →  Ch2-04 GPT, Ch7-04 DiT
```

> ⚠️ **선행 학습 필수**: 이 레포는 **Probability** (분포 · 기댓값 · KL) 와 **Information Theory** (KL · JSD · Wasserstein) 와 **Bayesian ML** (ELBO · Variational Inference) 를 선행 지식으로 전제합니다. "ELBO 가 reconstruction + regularization 으로 분해된다" 를 이해하려면 먼저 KL divergence 의 Gibbs inequality 와 variational inference 의 amortization 을 알고 있어야 합니다. Diffusion 부분은 추가로 **SDE Deep Dive** 의 forward/reverse SDE 와 Itô calculus 를 전제합니다.

> 💡 **이 레포의 핵심 기여**: Chapter 3 (VAE) · Chapter 5 (GAN) · Chapter 6 (Diffusion) 이 실전 생성 모델의 **세 기둥**입니다. VAE 는 "명시적 likelihood 하한"의 수학 (ELBO), GAN 은 "암묵적 likelihood minimax" 의 수학 (JSD 환원), Diffusion 은 "점진적 denoising 과 score" 의 수학 (Score-SDE) 을 대표합니다. Chapter 7 의 **5대 모델 통합 비교** 를 먼저 훑어보고 깊이 들어가면 각 모델의 위치가 선명해집니다.

> 🟡 **이 레포의 성격**: 여기서 다루는 일부 주제 — **Consistency Model 의 one-step generation 이 SOTA 로 수렴할지**, **Rectified Flow 가 Diffusion 을 완전히 대체할지**, **Energy-Based Model 의 부흥이 실전적일지** — 는 **현재 진행 중인 연구 영역**입니다. 레포는 "정답" 이 아니라 **"고전 MLE·ELBO 이론과 현대 multimodal 생성 사이의 지도"** 를 제공합니다.

---

## 🚀 빠른 시작

각 챕터의 첫 문서부터 바로 학습을 시작하세요!

[![Ch1](https://img.shields.io/badge/🔹_Ch1-생성모델_분류-9B59B6?style=for-the-badge)](./ch1-taxonomy/01-generative-vs-discriminative.md)
[![Ch2](https://img.shields.io/badge/🔹_Ch2-Autoregressive-9B59B6?style=for-the-badge)](./ch2-autoregressive/01-chain-rule-factorization.md)
[![Ch3](https://img.shields.io/badge/🔹_Ch3-VAE-9B59B6?style=for-the-badge)](./ch3-vae/01-elbo-derivation.md)
[![Ch4](https://img.shields.io/badge/🔹_Ch4-Normalizing_Flow-9B59B6?style=for-the-badge)](./ch4-flow/01-change-of-variables.md)
[![Ch5](https://img.shields.io/badge/🔹_Ch5-GAN-9B59B6?style=for-the-badge)](./ch5-gan/01-minimax-formulation.md)
[![Ch6](https://img.shields.io/badge/🔹_Ch6-Diffusion-9B59B6?style=for-the-badge)](./ch6-diffusion/01-ddpm-forward-reverse.md)
[![Ch7](https://img.shields.io/badge/🔹_Ch7-통합_비교-9B59B6?style=for-the-badge)](./ch7-unification/01-five-families-comparison.md)

---

## 📚 전체 학습 지도

> 💡 각 챕터를 클릭하면 상세 문서 목록이 펼쳐집니다

<br/>

### 🔹 Chapter 1: Generative Modeling 의 수학적 분류

> **핵심 질문:** Generative 와 discriminative 의 차이는 $p(x)$ 와 $p(y\|x)$ 만이 아닌가? Explicit (AR · Flow · VAE · Diffusion) 와 implicit (GAN · EBM) likelihood 의 근본적 차이는? 모든 생성 모델이 $\min_\theta \text{KL}(p_\text{data} \| p_\theta)$ 라는 하나의 목표를 어떻게 다르게 근사하는가? FID 와 NLL 은 왜 상관관계가 약한가?

<details>
<summary><b>생성 모델 분류 · 통합 목표 · 평가 지표 (4개 문서)</b></summary>

<br/>

| 문서 | 핵심 정리·증명·재현 |
|------|---------------------|
| [01. Generative vs Discriminative](./ch1-taxonomy/01-generative-vs-discriminative.md) | $p(x)$ (density estimation) vs $p(y\|x)$ (classification). Bayes 정리 $p(y\|x) \propto p(x\|y) p(y)$ 로 generative 가 discriminative 를 **포함**함을 증명. 생성 모델의 **4가지 역할** — 샘플링, likelihood 평가, representation learning, anomaly detection. Naive Bayes vs Logistic Regression 의 Ng & Jordan 2002 분석으로 generative 의 sample complexity 이득 |
| [02. Explicit vs Implicit Likelihood](./ch1-taxonomy/02-explicit-vs-implicit.md) | **Explicit**: AR · Flow · VAE · Diffusion — tractable 또는 bounded likelihood. **Implicit**: GAN · EBM — sampling 만 가능. 각 접근의 장단점: explicit 은 density estimation 가능하나 architecture 제약, implicit 은 expressive 하나 훈련 불안정·evaluation 어려움. **MLE 가능성** vs **adversarial training** 의 trade-off |
| [03. 통합 목표 — $\min \text{KL}(p_\text{data} \| p_\theta)$](./ch1-taxonomy/03-kl-minimization-unification.md) | **정리**: MLE $\max_\theta \mathbb{E}_{p_\text{data}}[\log p_\theta(x)] \equiv \min_\theta \text{KL}(p_\text{data} \| p_\theta)$ (상수 차이). 각 모델이 이 목표를 **어떻게 근사**: AR (직접 계산), VAE (ELBO 하한), Flow (exact via change-of-variables), GAN (JSD 환원), Diffusion (weighted score matching). **통합된 프레임워크** — 모든 생성 모델을 divergence 최소화로 기술하는 표 |
| [04. Evaluation — IS · FID · Precision/Recall · NLL](./ch1-taxonomy/04-evaluation-metrics.md) | **Inception Score** $IS = \exp(\mathbb{E}_x[\text{KL}(p(y\|x) \| p(y))])$ 의 수학, **FID** $= \|\mu_r - \mu_g\|^2 + \text{tr}(\Sigma_r + \Sigma_g - 2(\Sigma_r \Sigma_g)^{1/2})$ (Fréchet distance on Inception features). **Precision/Recall** (Kynkäänniemi 2019) 로 quality vs diversity 분해. NLL 과 perceptual metric 이 **왜 상관관계가 약한지** (Theis 2016) |

</details>

<br/>

### 🔹 Chapter 2: Autoregressive Model

> **핵심 질문:** Chain rule $p(x) = \prod_i p(x_i \| x_{<i})$ 은 왜 tractable likelihood 의 가장 단순한 접근인가? Masked convolution 이 어떻게 "future" 를 차단하면서 병렬 훈련을 가능케 하는가? WaveNet 의 dilated causal convolution 이 $O(2^L)$ receptive field 를 달성하는 메커니즘은? GPT 와 ImageGPT 가 동일 구조로 왜 modality 간 전이되는가?

<details>
<summary><b>Chain Rule · PixelCNN · WaveNet · GPT (4개 문서)</b></summary>

<br/>

| 문서 | 핵심 정리·증명·재현 |
|------|--------------|
| [01. Chain Rule Factorization](./ch2-autoregressive/01-chain-rule-factorization.md) | **항등식**: 임의의 분포에 대해 $p(x_1, \ldots, x_n) = \prod_{i=1}^n p(x_i \| x_{<i})$ — 확률의 법칙만으로 증명. 순서 선택이 modeling 에 미치는 영향 (raster-scan vs zigzag), **tractable likelihood** 이지만 **sequential sampling** 이 $O(n)$ 으로 느린 구조적 이유 |
| [02. PixelRNN · PixelCNN (van den Oord 2016)](./ch2-autoregressive/02-pixelrnn-pixelcnn.md) | 이미지를 pixel sequence $x_1, x_2, \ldots$ 로 flatten, **masked convolution** 으로 future pixel 참조 차단 — mask A (자기 자신도 차단) vs mask B (자기 자신 참조 허용). Row LSTM · Diagonal BiLSTM · PixelCNN 비교, conditional generation 으로 class-conditional 이미지. MNIST / CIFAR-10 에서 NLL 재현 |
| [03. WaveNet (van den Oord 2016)](./ch2-autoregressive/03-wavenet.md) | Audio ($16\text{kHz} \to 16000$ samples/sec) autoregressive 생성. **Dilated causal convolution** 으로 layer 당 receptive field 지수 증가 — $d_l = 2^{l-1}$ 로 $L$ 층에서 $RF = 2^L$, 30 층 → $\sim 1$ 초 오디오. Generation 속도 문제 ($O(n)$ sequential) 와 Parallel WaveNet 의 IAF 기반 distillation 해결책 |
| [04. Autoregressive Transformer — GPT as Generative Model](./ch2-autoregressive/04-gpt-as-generative.md) | Self-attention + causal mask 로 자연스럽게 autoregressive. **Modality-agnostic** — text (GPT), image (ImageGPT, Parti), audio (AudioLM, MusicLM), video (VideoGPT), multimodal (GPT-4V). **Scaling law** (Kaplan 2020, Hoffmann 2022) 로 generation quality 가 $N^{-0.076}$ 스케일, Transformer 레포의 self-attention 을 생성 관점에서 재해석 |

</details>

<br/>

### 🔹 Chapter 3: Variational Autoencoder (VAE)

> **핵심 질문:** $\log p(x) = \mathcal{L}(\theta, \phi; x) + \text{KL}(q_\phi(z\|x) \| p_\theta(z\|x))$ 에서 ELBO 가 **왜 정확히** $\mathbb{E}_q[\log p(x\|z)] - \text{KL}(q\|p)$ 로 분해되는가? Reparameterization trick 이 variance 를 어떻게 낮추는가? β-VAE 의 $\beta > 1$ 이 어떤 Information Bottleneck 과 대응되는가? Posterior collapse 는 언제 왜 발생하는가?

<details>
<summary><b>ELBO · Reparameterization · β-VAE · Posterior Collapse · VQ-VAE (5개 문서)</b></summary>

<br/>

| 문서 | 핵심 정리·증명·재현 |
|------|---------------------|
| [01. VAE 의 유도 · ELBO (Kingma & Welling 2013)](./ch3-vae/01-elbo-derivation.md) | Latent model $p(x, z) = p(x\|z) p(z)$, marginal $p(x) = \int p(x\|z) p(z)\,dz$ intractable. **ELBO 유도**: $\log p(x) = \log \int q(z\|x) \frac{p(x, z)}{q(z\|x)}\,dz \geq \mathbb{E}_q[\log p(x, z) - \log q(z\|x)]$ (Jensen), 등가적으로 $\log p(x) = \mathcal{L} + \text{KL}(q \| p(z\|x))$ $\square$. **Amortized inference** 의 encoder $q_\phi(z\|x)$ 공유 |
| [02. Reparameterization Trick](./ch3-vae/02-reparameterization.md) | $z \sim q_\phi(z\|x)$ 를 $z = g_\phi(x, \epsilon), \epsilon \sim p(\epsilon)$ 로 표현 (예: $z = \mu + \sigma \odot \epsilon$, $\epsilon \sim \mathcal{N}(0, I)$). **Path-wise gradient** $\nabla_\phi \mathbb{E}_q[f(z)] = \mathbb{E}_\epsilon[\nabla_\phi f(g_\phi(x, \epsilon))]$ vs **Score function (REINFORCE)** $\nabla_\phi \log q \cdot f(z)$ 의 variance 비교 — Rezende 2014 에서 $10^2 \sim 10^3$ 배 감소 측정. Gumbel-Softmax 로 discrete latent 확장 |
| [03. β-VAE 와 Information Bottleneck (Higgins 2017)](./ch3-vae/03-beta-vae-ib.md) | $\mathcal{L}_\beta = \mathbb{E}_q[\log p(x\|z)] - \beta \cdot \text{KL}(q \| p(z))$. **Information Bottleneck 해석** — $\beta$ 가 $I(X; Z)$ 의 **channel capacity 상한**. $\beta > 1$: disentangled representation ($\beta$-VAE 원 논문의 dSprites 실험), $\beta < 1$: higher-fidelity reconstruction. **Rate-distortion trade-off** 곡선으로 해석 (Alemi 2018) |
| [04. Posterior Collapse — 원인과 해결](./ch3-vae/04-posterior-collapse.md) | **현상**: $q_\phi(z\|x) \to p(z)$ 로 붕괴 → latent 정보 없음, decoder 만으로 reconstruction. **원인 분석**: (1) decoder 가 너무 강력 (autoregressive decoder), (2) KL 항이 훈련 초기에 지배. **해결책**: KL annealing (warm-up), Free Bits (Kingma 2016, $\max(\lambda, \text{KL})$), δ-VAE (Razavi 2019), skip connection. 실험: language VAE 에서 posterior 분석 |
| [05. VQ-VAE 와 Discrete Latent (van den Oord 2017)](./ch3-vae/05-vq-vae.md) | Continuous latent 대신 **codebook** $\{e_1, \ldots, e_K\}$ 로 vector-quantize, $z_q = e_{k^*}$ where $k^* = \arg\min_k \|z_e - e_k\|$. **Straight-through estimator** 로 gradient 를 $e_{k^*}$ 를 통과시킴, commitment loss $\|z_e - \text{sg}(e_{k^*})\|^2$. DALL-E · Jukebox 의 기반, VQ-VAE-2 의 hierarchical codebook. CIFAR-10 에서 codebook 사용률 측정 |

</details>

<br/>

### 🔹 Chapter 4: Normalizing Flow

> **핵심 질문:** $\log p(x) = \log p(z) - \log\|\det J_f(z)\|$ 의 change of variables 는 어떻게 exact likelihood 를 주는가? Coupling layer 가 **삼각 Jacobian** 을 만들어 $\det$ 을 $O(n)$ 으로 만드는 구조적 설계는? MAF 와 IAF 의 likelihood/sampling 속도 trade-off 의 근본 원인은? Continuous Normalizing Flow (FFJORD) 가 $\det$ 을 trace integral 로 바꾸는 메커니즘은?

<details>
<summary><b>Change of Variables · RealNVP · Glow · MAF/IAF · Neural ODE (5개 문서)</b></summary>

<br/>

| 문서 | 핵심 정리·증명·재현 |
|------|---------------------|
| [01. Change of Variables 와 Flow 의 정의](./ch4-flow/01-change-of-variables.md) | **정리**: $x = f_\theta(z)$ invertible, $C^1$ 이면 $p_X(x) = p_Z(f^{-1}(x)) \|\det J_{f^{-1}}(x)\| = p_Z(z) / \|\det J_f(z)\|$. 체인 $f = f_L \circ \cdots \circ f_1$ 에 대해 $\log p_X(x) = \log p_Z(z) - \sum_l \log\|\det J_{f_l}\|$ 유도 $\square$. **Exact likelihood** 를 제공하지만 **invertibility + tractable $\det$** 라는 이중 제약 |
| [02. RealNVP — Coupling Layer (Dinh 2017)](./ch4-flow/02-realnvp-coupling.md) | 입력 $x \in \mathbb{R}^D$ 를 $(x_{1:d}, x_{d+1:D})$ 로 분리. **Coupling**: $y_{1:d} = x_{1:d}$, $y_{d+1:D} = x_{d+1:D} \odot \exp(s(x_{1:d})) + t(x_{1:d})$. **Jacobian 이 lower triangular** $\Rightarrow \log\|\det J\| = \sum_{i > d} s_i$ 를 $O(D-d)$ 에 계산. **역변환**도 closed-form. 2-moon density estimation, MNIST 에서 exact likelihood 측정 |
| [03. Glow (Kingma & Dhariwal 2018)](./ch4-flow/03-glow.md) | RealNVP 에 **$1 \times 1$ invertible convolution** 추가 — 채널 간 permutation 의 학습 가능 일반화. $\log\|\det J\| = H \cdot W \cdot \log\|\det W\|$ ($W \in \mathbb{R}^{c \times c}$). Activation normalization + coupling + $1 \times 1$ conv 의 block 반복, 256×256 고해상도 얼굴 이미지 생성 재현. **Latent interpolation** 으로 의미 있는 manifold |
| [04. Autoregressive Flow — MAF · IAF](./ch4-flow/04-maf-iaf.md) | **MAF** (Papamakarios 2017): $x_i = \mu_i(x_{<i}) + \sigma_i(x_{<i}) \cdot z_i$ — density evaluation $O(1)$ parallel, sampling $O(D)$ sequential. **IAF** (Kingma 2016): $z_i = \mu_i(z_{<i}) + \sigma_i(z_{<i}) \cdot x_i$ — sampling $O(1)$ parallel, density $O(D)$. **이원성** — MAF/IAF 는 서로의 역, training 과 sampling 이 다른 방향으로 빠름 |
| [05. Continuous Normalizing Flow · Neural ODE · FFJORD](./ch4-flow/05-cnf-neural-ode.md) | **Neural ODE** (Chen 2018): $\frac{dz}{dt} = f_\theta(z(t), t)$. **Density ODE**: $\frac{d \log p(z(t))}{dt} = -\text{tr}\left(\frac{\partial f}{\partial z}\right)$, 적분하면 $\log p(z(T)) = \log p(z(0)) - \int_0^T \text{tr}(\partial f / \partial z)\,dt$. **FFJORD** (Grathwohl 2019): Hutchinson trace estimator $\text{tr}(A) = \mathbb{E}_{\epsilon}[\epsilon^\top A \epsilon]$ 로 $O(D)$ stochastic 추정, coupling 제약 없는 free-form Jacobian |

</details>

<br/>

### 🔹 Chapter 5: Generative Adversarial Network (GAN)

> **핵심 질문:** 왜 minimax 의 최적 $D$ 에서 $V = 2 \cdot JSD(p_d \| p_g) - \log 4$ 로 환원되는가? Mode collapse 가 왜 JSD 의 non-overlapping support 에서 발생하는가? Wasserstein-1 distance 가 왜 JSD 를 대체해야 하는가? Weight clipping · Gradient Penalty · Spectral Norm 의 이론적 차이는? StyleGAN 의 AdaIN 이 어떻게 스타일 분리를 달성하는가?

<details>
<summary><b>Minimax · JSD 환원 · Mode Collapse · WGAN · Spectral Norm · StyleGAN (6개 문서)</b></summary>

<br/>

| 문서 | 핵심 정리·증명·재현 |
|------|---------------------|
| [01. GAN 의 수학적 정식화 (Goodfellow 2014)](./ch5-gan/01-minimax-formulation.md) | $\min_G \max_D V(D, G) = \mathbb{E}_{p_d}[\log D(x)] + \mathbb{E}_{p_z}[\log(1 - D(G(z)))]$ — two-player zero-sum game. **Nash equilibrium** 로서의 해, generator 와 discriminator 의 교대 훈련, non-saturating loss $\log D(G(z))$ 변형의 gradient 분석 (초기 훈련에서 $\log(1-D)$ 가 saturate 하는 문제) |
| [02. 최적 D 에서 V 의 JSD 환원 증명](./ch5-gan/02-jsd-reduction.md) | **정리**: 고정된 $G$ 에 대해 $D^*(x) = \frac{p_d(x)}{p_d(x) + p_g(x)}$. 대입하면 $V(D^*, G) = \mathbb{E}_{p_d}\left[\log \frac{p_d}{p_d + p_g}\right] + \mathbb{E}_{p_g}\left[\log \frac{p_g}{p_d + p_g}\right] = 2 \cdot JSD(p_d \| p_g) - \log 4$ 유도 $\square$. JSD $\geq 0$ 이고 $= 0 \iff p_d = p_g$ 이므로 minimax 해가 **유일**하게 $p_g = p_d$ |
| [03. GAN 훈련 불안정성 · Mode Collapse](./ch5-gan/03-mode-collapse.md) | **Vanishing gradient 문제**: $p_d$ 와 $p_g$ 의 support 가 겹치지 않으면 JSD $= \log 2$ (상수) → gradient $= 0$. **Mode collapse**: generator 가 특정 mode 로 수렴, diversity 상실. **원인 분석**: reverse KL 성향, Nash equilibrium 의 국소성, discriminator 의 과학습. **해결책 비교**: minibatch discrimination, unrolled GAN, mode-seeking regularization |
| [04. Wasserstein GAN (Arjovsky 2017)](./ch5-gan/04-wgan.md) | **Wasserstein-1**: $W(p_d, p_g) = \inf_{\gamma \in \Pi(p_d, p_g)} \mathbb{E}_{(x, y) \sim \gamma}[\|x - y\|]$ — support 겹치지 않아도 **연속이고 미분 가능**. **Kantorovich-Rubinstein 쌍대**: $W = \sup_{\|f\|_L \leq 1} \mathbb{E}_{p_d}[f] - \mathbb{E}_{p_g}[f]$ (Villani 2003). WGAN 은 discriminator 를 1-Lipschitz 로 제약, **weight clipping** $w \leftarrow \text{clip}(w, -c, c)$ vs **WGAN-GP** (Gulrajani 2017) gradient penalty $\lambda \mathbb{E}_{\hat x}[(\|\nabla_{\hat x} D\|_2 - 1)^2]$ |
| [05. Spectral Normalization (Miyato 2018)](./ch5-gan/05-spectral-norm.md) | **아이디어**: 각 weight matrix 의 spectral norm $\sigma(W) = \max_{\|x\|=1} \|Wx\|$ 을 1로 → 전체 NN 의 Lipschitz 상수 bound. **Power iteration** 으로 $\sigma(W)$ 효율적 추정 ($u \leftarrow Wv / \|Wv\|$, $v \leftarrow W^\top u / \|W^\top u\|$). WGAN-GP 대비 구현 간단, gradient penalty 의 memory overhead 없음. ImageNet conditional generation 에서 BigGAN 으로 이어지는 핵심 |
| [06. Progressive GAN · StyleGAN · StyleGAN2 (Karras)](./ch5-gan/06-stylegan.md) | **Progressive growing** (Karras 2018): 저해상도 → 고해상도 점진적 훈련, 1024×1024 얼굴 최초 성공. **StyleGAN** (Karras 2019): mapping network $z \to w$ 로 disentangle, **AdaIN** $\text{AdaIN}(x, y) = \sigma(y) \frac{x - \mu(x)}{\sigma(x)} + \mu(y)$ 로 style 주입, stochastic variation 을 noise 로. **StyleGAN2** (Karras 2020): weight demodulation 으로 artifact 제거, FFHQ SOTA |

</details>

<br/>

### 🔹 Chapter 6: Diffusion Model

> **핵심 질문:** DDPM 의 ELBO 가 어떻게 $\|\epsilon - \epsilon_\theta(x_t, t)\|^2$ 로 환원되는가? Noise prediction 이 왜 weighted denoising score matching 과 정확히 동치인가? Score-SDE 가 어떻게 DDPM · NCSN · Probability Flow ODE 를 하나의 프레임워크로 통합하는가? Classifier-Free Guidance 의 $w$ 는 quality 와 diversity 를 어떻게 trade-off 하는가?

<details>
<summary><b>DDPM · Simple Loss · Score-Based · Score-SDE · CFG (5개 문서)</b></summary>

<br/>

| 문서 | 핵심 정리·증명·재현 |
|------|---------------------|
| [01. DDPM — Forward · Reverse (Ho 2020)](./ch6-diffusion/01-ddpm-forward-reverse.md) | **Forward**: $q(x_t \| x_{t-1}) = \mathcal{N}(\sqrt{1-\beta_t} x_{t-1}, \beta_t I)$, closed-form $q(x_t \| x_0) = \mathcal{N}(\sqrt{\bar\alpha_t} x_0, (1-\bar\alpha_t) I)$ 유도 (reparameterization chain). **Reverse**: $p_\theta(x_{t-1} \| x_t) = \mathcal{N}(\mu_\theta(x_t, t), \Sigma_\theta(x_t, t))$. ELBO $= \mathbb{E}_q[\log p(x_T) + \sum_{t=1}^T \log \frac{p_\theta(x_{t-1}\|x_t)}{q(x_t\|x_{t-1})}]$ 유도 |
| [02. DDPM Loss 의 단순화 — $L_\text{simple}$](./ch6-diffusion/02-ddpm-simple-loss.md) | **정리**: $\mu_\theta(x_t, t) = \frac{1}{\sqrt{\alpha_t}}\left(x_t - \frac{\beta_t}{\sqrt{1-\bar\alpha_t}}\epsilon_\theta(x_t, t)\right)$ 로 parameterize 하면 ELBO 의 각 항이 $\mathbb{E}_{x_0, \epsilon}\left[\frac{\beta_t^2}{2\sigma_t^2 \alpha_t (1-\bar\alpha_t)} \|\epsilon - \epsilon_\theta(\sqrt{\bar\alpha_t} x_0 + \sqrt{1-\bar\alpha_t}\epsilon, t)\|^2\right]$. Ho 2020 은 가중치를 무시한 **$L_\text{simple} = \mathbb{E}_{t, x_0, \epsilon}[\|\epsilon - \epsilon_\theta(x_t, t)\|^2]$** 가 실증적으로 우월함을 발견 $\square$ |
| [03. Score-Based Model — NCSN (Song & Ermon 2019)](./ch6-diffusion/03-score-based-ncsn.md) | **Score** $s(x) = \nabla_x \log p(x)$, **Langevin dynamics** $x_{t+1} = x_t + \frac{\delta}{2} s_\theta(x_t) + \sqrt{\delta} z_t$ 로 샘플링. **Denoising Score Matching** (Vincent 2011): $\mathbb{E}_{q_\sigma(\tilde x \| x)}[\|s_\theta(\tilde x) - \nabla_{\tilde x} \log q_\sigma(\tilde x \| x)\|^2]$ 가 score matching 과 동치. **DDPM = weighted DSM** 증명 — $\epsilon_\theta$ 학습 $\equiv$ score 학습 |
| [04. Score-SDE (Song 2021)](./ch6-diffusion/04-score-sde.md) | **Forward SDE**: $dx = f(x, t)\,dt + g(t)\,dW$. **Reverse-time SDE** (Anderson 1982): $dx = [f(x, t) - g(t)^2 \nabla_x \log p_t(x)]\,dt + g(t)\,d\bar W$. **VP-SDE** $\Rightarrow$ DDPM, **VE-SDE** $\Rightarrow$ NCSN 을 limit 으로 증명. **Probability Flow ODE**: $\frac{dx}{dt} = f(x, t) - \frac{1}{2} g(t)^2 \nabla_x \log p_t(x)$ 로 결정론적 sampling, exact likelihood, latent manipulation. SDE 레포의 Itô calculus 로 유도 |
| [05. Classifier-Free Guidance · 현대 응용](./ch6-diffusion/05-classifier-free-guidance.md) | **Classifier guidance** (Dhariwal 2021): $\tilde\epsilon = \epsilon_\theta(x_t, t) - \sqrt{1-\bar\alpha_t} \cdot w \cdot \nabla_x \log p_\phi(y\|x_t)$. **Classifier-Free** (Ho & Salimans 2022): joint $\epsilon_\theta(x_t, t, y)$ 와 unconditional $\epsilon_\theta(x_t, t, \emptyset)$ 를 동시 훈련, $\tilde\epsilon = (1+w)\epsilon_\theta(x_t, t, y) - w\epsilon_\theta(x_t, t, \emptyset)$. **Stable Diffusion · DALL-E 3 · Imagen · SD3** 의 $w \approx 7.5$ 선택, DDIM (Song 2020) 의 non-Markovian sampling 으로 50 step 에 압축 |

</details>

<br/>

### 🔹 Chapter 7: 통합과 최신 동향

> **핵심 질문:** 5가지 생성 모델 계보를 어떤 기준으로 비교해야 하는가 (likelihood · speed · quality · stability)? Consistency Model 의 one-step generation 이 Diffusion 을 대체할 수 있는가? Energy-Based Model 의 부흥이 실전적인가? 현대 multimodal generation (text-to-image, text-to-video, text-to-3D) 은 어떤 수학적 조합인가?

<details>
<summary><b>5대 계보 비교 · Consistency Model · EBM · Frontier (4개 문서)</b></summary>

<br/>

| 문서 | 핵심 정리·재현 |
|------|--------------|
| [01. 5대 계보 통합 비교](./ch7-unification/01-five-families-comparison.md) | **비교 표**: Likelihood (Exact · Lower · None), Sampling (Fast · Medium · Slow), Quality (SOTA · Good · Blurry), Training (Stable · Unstable), Parameter efficiency, Inference cost. **의사결정 트리**: "exact likelihood 필요?" → Flow, "sample quality 우선?" → Diffusion, "real-time generation?" → GAN, "representation learning?" → VAE, "tractable sequence?" → AR. MNIST / CIFAR-10 에서 5모델 동시 훈련 후 FID/IS/NLL/sampling-time 직접 비교 |
| [02. Consistency Model · Rectified Flow](./ch7-unification/02-consistency-rectified-flow.md) | **Consistency Model** (Song 2023): $f_\theta(x_t, t) \to x_0$ (모든 $t$ 에 대해 일관성). **One-step generation** — diffusion sampling 을 1~4 step 으로 압축. **Distillation** vs **isolation** 훈련 방식. **Rectified Flow** (Liu 2022): $dx/dt = v_\theta(x, t)$, straight trajectory 로 ODE 적분 오차 최소화. SD3 가 Rectified Flow 기반으로 전환한 이유. 각 기법의 one-step FID 비교 재현 |
| [03. Generative Model 과 Energy-Based Model](./ch7-unification/03-ebm-revival.md) | **EBM**: $p_\theta(x) = \frac{1}{Z(\theta)} \exp(-E_\theta(x))$, partition function $Z$ intractable. **훈련**: MCMC (Langevin · HMC) 로 negative sample, contrastive divergence (Hinton 2002). **부흥**: JEM (Grathwohl 2020) — 분류기를 EBM 으로 재해석, Yang 2022 의 generative classifier. **Diffusion 과의 관계**: Score-based model 이 EBM 의 derivative 를 직접 학습하여 $Z$ 를 피함. 1D EBM 훈련으로 Langevin sampling 재현 |
| [04. Frontier — Multimodal · Video · 3D · 과학](./ch7-unification/04-frontier.md) | **Multimodal image**: DALL-E 3 (OpenAI), Imagen (Google), Stable Diffusion XL · 3 (Stability AI), Parti (autoregressive). **Video**: Sora (DiT-based), Runway Gen-3, Pika. **3D**: NeRF (Mildenhall 2020) + DreamFusion (SDS loss), Gaussian Splatting (Kerbl 2023) 생성. **과학적 응용**: AlphaFold 3 (Flow matching-based 단백질), RFdiffusion (단백질 설계), material discovery (MatterGen). 각 분야가 **어떤 생성 모델 조합** 을 사용하는지 지도 |

</details>

---

> 🆕 **2026-04 최신 업데이트**: Ch5-02 JSD 환원 증명에 Jensen 의 등호 조건 분석을 보강했고, Ch6-04 Score-SDE 의 Anderson 1982 reverse-time SDE 유도를 VP · VE · sub-VP 세 케이스로 분리했으며, Ch6-05 Classifier-Free Guidance 섹션에 SD3 (Stable Diffusion 3, MM-DiT + Rectified Flow) 분석을 추가했습니다. Ch7-02 Consistency Model 은 `diffusers==0.25` 기반 one-step FID 측정으로 리팩토링되었고, 11-섹션 문서 골격이 전체 33개 문서에서 일관됩니다.

## 🏆 핵심 정리 인덱스

이 레포에서 **완전한 증명** 또는 **원 논문 실험 재현** 을 제공하는 대표 결과 모음입니다. 각 챕터 문서에서 $\square$ 로 종결되는 엄밀한 증명 또는 `results/` 하의 플롯을 확인할 수 있습니다. (전체 76개 정리 중 핵심만 발췌)

| 정리·결과 | 서술 | 출처 문서 |
|----------|------|----------|
| **MLE ≡ min KL** | $\max_\theta \mathbb{E}_{p_d}[\log p_\theta] \equiv \min_\theta \text{KL}(p_d \| p_\theta)$ — 생성 모델의 통합 목표 | [Ch1-03](./ch1-taxonomy/03-kl-minimization-unification.md) |
| **FID 정의** | Fréchet distance $\|\mu_r - \mu_g\|^2 + \text{tr}(\Sigma_r + \Sigma_g - 2(\Sigma_r \Sigma_g)^{1/2})$ | [Ch1-04](./ch1-taxonomy/04-evaluation-metrics.md) |
| **Chain Rule Factorization** | $p(x_1, \ldots, x_n) = \prod_i p(x_i \| x_{<i})$ — AR 의 기반 | [Ch2-01](./ch2-autoregressive/01-chain-rule-factorization.md) |
| **WaveNet Dilated RF** | $d_l = 2^{l-1}$ 로 $L$ 층에서 $RF = 2^L$, 30 층 → 1초 오디오 | [Ch2-03](./ch2-autoregressive/03-wavenet.md) |
| **ELBO 분해** | $\log p(x) = \mathcal{L} + \text{KL}(q\|p(z\|x))$, $\mathcal{L} = \mathbb{E}_q[\log p(x\|z)] - \text{KL}(q\|p(z))$ | [Ch3-01](./ch3-vae/01-elbo-derivation.md) |
| **Reparameterization Gradient Variance** | Path-wise vs REINFORCE 의 variance $10^2 \sim 10^3$ 배 감소 | [Ch3-02](./ch3-vae/02-reparameterization.md) |
| **β-VAE Information Bottleneck** | $\beta$ 가 $I(X; Z)$ 의 channel capacity 상한, rate-distortion | [Ch3-03](./ch3-vae/03-beta-vae-ib.md) |
| **Change of Variables** | $\log p_X(x) = \log p_Z(z) - \sum_l \log\|\det J_{f_l}\|$ — exact likelihood | [Ch4-01](./ch4-flow/01-change-of-variables.md) |
| **RealNVP 삼각 Jacobian** | Coupling 으로 $\det$ 을 $O(D)$ 에 계산 | [Ch4-02](./ch4-flow/02-realnvp-coupling.md) |
| **MAF/IAF 이원성** | Density 평가 vs sampling 속도의 $O(1)$ / $O(D)$ 대칭 | [Ch4-04](./ch4-flow/04-maf-iaf.md) |
| **FFJORD Trace Integral** | $\log p(z_T) - \log p(z_0) = -\int \text{tr}(\partial f/\partial z)\,dt$, Hutchinson 추정 | [Ch4-05](./ch4-flow/05-cnf-neural-ode.md) |
| **GAN Minimax Equilibrium** | 최적 $D^*(x) = p_d / (p_d + p_g)$, $V = 2 \cdot JSD - \log 4$ | [Ch5-02](./ch5-gan/02-jsd-reduction.md) |
| **Mode Collapse — JSD Support** | Non-overlapping support 에서 JSD $= \log 2$, gradient $= 0$ | [Ch5-03](./ch5-gan/03-mode-collapse.md) |
| **Kantorovich-Rubinstein 쌍대** | $W_1 = \sup_{\|f\|_L \leq 1} \mathbb{E}_{p_d}[f] - \mathbb{E}_{p_g}[f]$ | [Ch5-04](./ch5-gan/04-wgan.md) |
| **Spectral Norm Lipschitz Bound** | $\|f\|_L \leq \prod_l \sigma(W_l)$, power iteration 으로 추정 | [Ch5-05](./ch5-gan/05-spectral-norm.md) |
| **DDPM Closed-form Forward** | $q(x_t \| x_0) = \mathcal{N}(\sqrt{\bar\alpha_t} x_0, (1-\bar\alpha_t) I)$ | [Ch6-01](./ch6-diffusion/01-ddpm-forward-reverse.md) |
| **$L_\text{simple}$ 유도** | ELBO $\to \mathbb{E}_{t, x_0, \epsilon}[\|\epsilon - \epsilon_\theta\|^2]$ (Ho 2020) | [Ch6-02](./ch6-diffusion/02-ddpm-simple-loss.md) |
| **DDPM ≡ Weighted DSM** | Noise prediction 과 score matching 의 동치 증명 | [Ch6-03](./ch6-diffusion/03-score-based-ncsn.md) |
| **Reverse-time SDE** | $dx = [f - g^2 \nabla \log p_t]\,dt + g\,d\bar W$ (Anderson 1982) | [Ch6-04](./ch6-diffusion/04-score-sde.md) |
| **Classifier-Free Guidance** | $\tilde\epsilon = (1+w)\epsilon_\theta(x_t, y) - w\epsilon_\theta(x_t, \emptyset)$ | [Ch6-05](./ch6-diffusion/05-classifier-free-guidance.md) |
| **5대 계보 Trade-off** | AR · VAE · Flow · GAN · Diffusion 의 likelihood/speed/quality 표 | [Ch7-01](./ch7-unification/01-five-families-comparison.md) |
| **Consistency Model One-Step** | $f_\theta(x_t, t) \to x_0$ 로 1 step sampling | [Ch7-02](./ch7-unification/02-consistency-rectified-flow.md) |

> 💡 **챕터별 문서·정리/정의 수**: Ch1(4문서, 23 정리·정의) · Ch2(4문서, 23) · Ch3(5문서, 30) · Ch4(5문서, 29) · Ch5(6문서, 36) · Ch6(5문서, 31) · Ch7(4문서, 22) — 합계 **33문서 + 194 정리·정의 + 30 엄밀한 $\square$ 증명 + 110+ PyTorch 실험**, 약 **12,000 라인** 분량.

---

## 💻 실험 환경

모든 챕터의 실험은 아래 환경에서 재현 가능합니다.

```bash
# requirements.txt
numpy==1.26.0
scipy==1.11.0
matplotlib==3.8.0
tqdm==4.66.0
torch==2.1.0
torchvision==0.16.0
diffusers==0.25.0     # Ch6 Diffusion 참조 (Stable Diffusion · DDIM · CFG)
normflows==1.7        # Ch4 Normalizing Flow 참조 (RealNVP · Glow · MAF)
torchdiffeq==0.2.3    # Ch4-05 Neural ODE · FFJORD
einops==0.7.0         # Ch2 Transformer · Ch6 DiT
torchmetrics==1.3.0   # FID · IS 계산
Pillow==10.0.0
jupyter==1.0.0
```

```bash
# 환경 설치
pip install numpy==1.26.0 scipy==1.11.0 matplotlib==3.8.0 tqdm==4.66.0 \
            torch==2.1.0 torchvision==0.16.0 diffusers==0.25.0 \
            normflows==1.7 torchdiffeq==0.2.3 einops==0.7.0 \
            torchmetrics==1.3.0 Pillow==10.0.0 jupyter==1.0.0

# 실험 노트북 실행
jupyter notebook
```

```python
# 대표 실험 ① — GAN JSD 환원 수치 검증 (Ch5-02, Goodfellow 2014 재현)
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

def compute_jsd(p, q):
    m = 0.5 * (p + q)
    kl_pm = (p * np.log((p + 1e-12) / (m + 1e-12))).sum()
    kl_qm = (q * np.log((q + 1e-12) / (m + 1e-12))).sum()
    return 0.5 * (kl_pm + kl_qm)

x = np.linspace(-5, 5, 200)
p_data = np.exp(-0.5 * (x - 1.0) ** 2); p_data /= p_data.sum()
p_gen  = np.exp(-0.5 * (x + 0.5) ** 2); p_gen  /= p_gen.sum()

D_star   = p_data / (p_data + p_gen + 1e-12)               # 최적 D
V_opt    = (p_data * np.log(D_star + 1e-12)).sum() \
         + (p_gen  * np.log(1 - D_star + 1e-12)).sum()
V_theory = 2 * compute_jsd(p_data, p_gen) - np.log(4)       # 2·JSD - log 4
print(f"V(D*, G) (직접)    = {V_opt:.6f}")
print(f"2·JSD(p_d ‖ p_g) - log4 = {V_theory:.6f}")
# → 두 값이 일치 (Goodfellow 2014 정리 수치 확인)

# 대표 실험 ② — 1D DDPM on 2-modal Gaussian mixture (Ch6-02)
class DDPM1D(nn.Module):
    def __init__(self, h=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2, h), nn.SiLU(),
            nn.Linear(h, h), nn.SiLU(),
            nn.Linear(h, 1),
        )
    def forward(self, x, t):
        return self.net(torch.cat([x, t.float().unsqueeze(-1) / T], dim=-1))

T = 200
betas  = torch.linspace(1e-4, 0.02, T)
alphas = 1 - betas
abar   = torch.cumprod(alphas, 0)

def q_sample(x0, t, eps):
    a = abar[t].unsqueeze(-1)
    return torch.sqrt(a) * x0 + torch.sqrt(1 - a) * eps

model = DDPM1D()
opt   = torch.optim.Adam(model.parameters(), lr=2e-3)

for step in range(3000):
    # target: 2-mode mixture
    x0 = torch.where(torch.rand(256, 1) > 0.5,
                     torch.randn(256, 1) * 0.5 + 2.0,
                     torch.randn(256, 1) * 0.5 - 2.0)
    t   = torch.randint(0, T, (256,))
    eps = torch.randn_like(x0)
    xt  = q_sample(x0, t, eps)
    loss = ((eps - model(xt, t)) ** 2).mean()   # L_simple (Ho 2020)
    opt.zero_grad(); loss.backward(); opt.step()

@torch.no_grad()
def sample(n=2000):
    x = torch.randn(n, 1)
    for t in reversed(range(T)):
        z   = torch.randn_like(x) if t > 0 else 0
        a, ac, b = alphas[t], abar[t], betas[t]
        eps = model(x, torch.full((n,), t))
        x   = (1 / torch.sqrt(a)) * (x - (b / torch.sqrt(1 - ac)) * eps) \
            + torch.sqrt(b) * z
    return x

plt.hist(sample().numpy().flatten(), bins=60, density=True, alpha=0.6)
plt.title("DDPM 1D — 2-modal mixture 학습 (L_simple)"); plt.show()

# 대표 실험 ③ — Reparameterization vs REINFORCE variance (Ch3-02)
def reparam_grad(mu, sigma, n=10000):
    eps = torch.randn(n)
    z = mu + sigma * eps                            # path-wise
    f = z ** 2
    return torch.autograd.grad(f.mean(), [mu, sigma])

def reinforce_grad(mu, sigma, n=10000):
    z = mu + sigma * torch.randn(n)                 # score function
    logq = -0.5 * ((z - mu) / sigma) ** 2 - torch.log(sigma)
    f = z ** 2
    return torch.autograd.grad((f.detach() * logq).mean(), [mu, sigma])
# 여러 seed 반복 → variance(reparam) / variance(reinforce) 비율 측정

# 대표 실험 ④ — RealNVP coupling layer on 2-moon (Ch4-02)
class CouplingLayer(nn.Module):
    def __init__(self, d=2, hidden=64):
        super().__init__()
        self.s = nn.Sequential(nn.Linear(d // 2, hidden), nn.ReLU(),
                               nn.Linear(hidden, d // 2))
        self.t = nn.Sequential(nn.Linear(d // 2, hidden), nn.ReLU(),
                               nn.Linear(hidden, d // 2))
    def forward(self, x):
        x1, x2 = x.chunk(2, dim=-1)
        s = torch.tanh(self.s(x1))
        y2 = x2 * torch.exp(s) + self.t(x1)
        logdet = s.sum(-1)
        return torch.cat([x1, y2], dim=-1), logdet
# 2-moon 데이터로 exact log-likelihood 훈련, Gaussian ← → moon transport 시각화
```

---

## 📖 각 문서 구성 방식

모든 문서는 다음 **11-섹션 골격**으로 작성됩니다.

| # | 섹션 | 내용 |
|:-:|------|------|
| 1 | 🎯 **핵심 질문** | 이 문서가 답하는 3~5개의 본질적 질문 |
| 2 | 🔍 **왜 이 생성 모델이 필요한가** | Likelihood tractability · sample quality · training stability 와의 연결 |
| 3 | 📐 **수학적 선행 조건** | Prob · Info · Bayesian · SDE · Opt · Convex · NN Theory 레포의 어떤 정리를 전제하는지 |
| 4 | 📖 **직관적 이해** | 샘플링 과정 · latent space · forward/reverse process 시각화 |
| 5 | ✏️ **엄밀한 정의·정리** | ELBO · change of variables · JSD 환원 · DDPM forward/reverse |
| 6 | 🔬 **증명 또는 수학적 유도** | JSD 환원 · reparameterization 정당성 · RealNVP Jacobian · $L_\text{simple}$ |
| 7 | 💻 **실험 재현** | MNIST/CIFAR 훈련 · FID/IS 측정 · sample grid · latent interpolation · diffusion 과정 시각화 |
| 8 | 🔗 **이론과 실전의 간극** | 이 결과가 실제 Stable Diffusion · DALL-E · Sora 를 얼마나 설명하는가 |
| 9 | ⚖️ **가정과 한계** | Mode collapse · posterior collapse · Jacobian trade-off · Markov 가정 |
| 10 | 📌 **핵심 정리** | 한 장으로 요약 |
| 11 | 🤔 **생각해볼 문제 (+ 해설)** | 손 계산·증명 재구성·구현·논문 비평 문제 |

> 📚 **연습문제 총 99개**: 모든 문서가 3문제씩 (기초 / 심화 / 논문 비평) — 33문서 × 3 = 99문제. 모든 문제에 `<details>` 펼침 해설 포함. ELBO 손 유도부터 JSD 환원 재증명, RealNVP Jacobian 계산, DDPM simple loss 유도, Score-SDE reverse 증명, Stable Diffusion CFG $w$ sweep 까지 단계적으로 심화됩니다.
>
> 🧭 **푸터 네비게이션**: 각 문서 하단에 `◀ 이전 / 📚 README / 다음 ▶` 링크가 항상 제공됩니다. 챕터 경계에서도 다음 챕터 첫 문서로 자동 연결됩니다.
>
> ⏱️ **학습 시간 추정**: 문서당 평균 약 360줄 (정의·증명·코드·연습문제 포함) 기준 **약 40분~1시간**. 전체 33문서는 약 **22~33시간** 상당 (증명 재구성·실험 재현 포함 시 40시간+).

---

## 🗺️ 추천 학습 경로

<details>
<summary><b>🟢 "생성 모델을 쓰지만 왜 작동하는지 이론적으로 이해하고 싶다" — 입문 투어 (1주, 약 10~12시간)</b></summary>

<br/>

```
Day 1  Ch1-01  Generative vs Discriminative
       Ch1-03  MLE ≡ KL 최소화 통합 목표
Day 2  Ch3-01  VAE ELBO 유도
       Ch3-02  Reparameterization Trick
Day 3  Ch4-01  Change of Variables
       Ch4-02  RealNVP Coupling
Day 4  Ch5-01  GAN Minimax 정식화
       Ch5-02  JSD 환원 증명
Day 5  Ch6-01  DDPM Forward/Reverse
       Ch6-02  L_simple 유도
Day 6  Ch6-05  Classifier-Free Guidance
       Ch2-04  GPT as Generative
Day 7  Ch7-01  5대 계보 통합 비교
       Ch7-04  Frontier 지도
```

</details>

<details>
<summary><b>🟡 "ELBO · JSD · Score 의 수학을 완전히 정복한다" — 이론 집중 (2주, 약 20~26시간)</b></summary>

<br/>

```
1주차 — Explicit Likelihood (명시적)
  Day 1    Ch1-01~04  분류 · 통합 목표 · 평가 지표 전반
  Day 2    Ch2-01~02  Chain rule + PixelCNN
  Day 3    Ch3-01~02  ELBO 유도 + Reparameterization 꼼꼼히
  Day 4    Ch3-03~04  β-VAE + Posterior Collapse 분석
  Day 5    Ch3-05     VQ-VAE + DALL-E 기반
  Day 6    Ch4-01~03  Change of Variables + RealNVP + Glow
  Day 7    Ch4-04~05  MAF/IAF + FFJORD

2주차 — Implicit + Diffusion
  Day 1    Ch5-01~02  GAN minimax + JSD 환원 손 증명
  Day 2    Ch5-03~04  Mode collapse + WGAN Kantorovich-Rubinstein
  Day 3    Ch5-05~06  Spectral Norm + StyleGAN
  Day 4    Ch6-01~02  DDPM + L_simple 유도
  Day 5    Ch6-03     Score-based NCSN
  Day 6    Ch6-04     Score-SDE 통합 프레임워크
  Day 7    Ch6-05 + Ch7-01~02  CFG + 5대 비교 + Consistency Model
```

</details>

<details>
<summary><b>🔴 "생성 모델의 수학을 완전 정복한다" — 전체 정복 (10주, 약 30~42시간 + 실험 재현 12~18시간)</b></summary>

<br/>

```
1주차   Chapter 1 전체 — 수학적 분류
         → Explicit vs Implicit 개념 정립
         → MLE ≡ min KL 증명
         → FID · IS · NLL 의 차이 측정

2주차   Chapter 2 전체 — Autoregressive
         → Chain rule 손 유도
         → PixelCNN masked conv 구현
         → WaveNet dilated RF 측정

3주차   Chapter 3 (1~2) — VAE 핵심
         → ELBO 3가지 유도 방식 (Jensen · KL split · importance weighted)
         → Reparameterization variance 측정 실험

4주차   Chapter 3 (3~5) — VAE 심화
         → β-VAE disentanglement 재현
         → Posterior collapse 진단과 해결
         → VQ-VAE로 DALL-E-style 생성

5주차   Chapter 4 전체 — Normalizing Flow
         → Change of variables 손 계산
         → RealNVP 2-moon density estimation
         → Glow로 고해상도 얼굴
         → FFJORD Hutchinson trace 구현

6주차   Chapter 5 (1~3) — GAN 기초
         → JSD 환원 손 증명
         → Mode collapse 재현 실험
         → DCGAN → WGAN 훈련 비교

7주차   Chapter 5 (4~6) — GAN 심화
         → WGAN-GP vs Spectral Norm 비교
         → StyleGAN2 얼굴 생성 재현
         → BigGAN-scale 훈련 시도

8주차   Chapter 6 (1~3) — Diffusion 기초
         → DDPM 1D · 2D 구현
         → L_simple vs L_ELBO 비교
         → NCSN Langevin sampling

9주차   Chapter 6 (4~5) — Diffusion 심화
         → Score-SDE 프레임워크 구현
         → Probability Flow ODE exact likelihood
         → CFG w sweep · DDIM 50-step 가속

10주차  Chapter 7 전체 — 통합과 Frontier
         → 5모델 동시 훈련 벤치마크
         → Consistency Model one-step
         → EBM Langevin 재현
         → Stable Diffusion / Sora / AlphaFold 3 분석
```

</details>

---

## 🔗 연관 레포지토리

| 레포 | 주요 내용 | 연관 챕터 |
|------|----------|-----------|
| [probability-theory-deep-dive](https://github.com/iq-ai-lab/probability-theory-deep-dive) | 분포 · 기댓값 · KL · 조건부 기댓값 | **전체 레포의 전제**, 특히 Ch1-03 (MLE ≡ KL) |
| [information-theory-deep-dive](https://github.com/iq-ai-lab/information-theory-deep-dive) | Entropy · KL · JSD · MI · rate-distortion | **Ch3-03** (β-VAE IB), **Ch5-02** (JSD 환원), Ch1-04 (metrics) |
| [bayesian-ml-deep-dive](https://github.com/iq-ai-lab/bayesian-ml-deep-dive) | VI · ELBO · MCMC · amortized inference | **Ch3 전체** (VAE), Ch7-03 (EBM MCMC) |
| [sde-deep-dive](https://github.com/iq-ai-lab/sde-deep-dive) | Itô · Forward/Reverse SDE · Fokker-Planck | **Ch6-04** (Score-SDE), Ch6-03 (Langevin) |
| [neural-network-theory-deep-dive](https://github.com/iq-ai-lab/neural-network-theory-deep-dive) | UAT · Backprop · Xavier·He init | **전체 레포의 전제**, 모든 모델의 NN 파라미터화 |
| [optimization-theory-deep-dive](https://github.com/iq-ai-lab/optimization-theory-deep-dive) | GD · SGD · Adam · minimax · saddle point | **Ch5** (GAN 훈련), Ch5-03 (Nash equilibrium) |
| [convex-optimization-deep-dive](https://github.com/iq-ai-lab/convex-optimization-deep-dive) | Dual · Lagrangian · Kantorovich-Rubinstein | **Ch5-04** (WGAN) |
| [transformer-deep-dive](https://github.com/iq-ai-lab/transformer-deep-dive) | Self-attention · Causal mask · Scaling | **Ch2-04** (GPT), Ch6-05 (DiT) |
| [cnn-deep-dive](https://github.com/iq-ai-lab/cnn-deep-dive) | Convolution · ResNet · 현대 아키텍처 | Ch2-02 (PixelCNN), Ch5-06 (StyleGAN 백본), Ch6 (U-Net in DDPM) |
| [multimodal-deep-dive](https://github.com/iq-ai-lab/multimodal-deep-dive) *(다음)* | Text-to-image · Text-to-video · 3D 생성 | **Ch7-04** 이후 직접 연결 |

> 💡 이 레포는 **"5가지 생성 모델이 모두 KL 최소화를 어떻게 다르게 근사하는가"** 에 집중합니다. Probability 에서 KL · JSD 를 익히고, Bayesian ML 에서 ELBO 와 variational inference 를 익힌 후 오면 Chapter 3 (VAE) 와 Chapter 5 (GAN) 의 증명이 훨씬 자연스럽습니다. Chapter 6 (Diffusion) 은 SDE Deep Dive 의 forward/reverse SDE 와 Itô calculus 를 전제로 시작합니다. Multimodal (다음 레포) 은 이 레포 Chapter 7-04 (Frontier) 을 전제로 이어집니다.

---

## 📖 Reference

### 🏛️ 고전·교과서
- **Deep Learning** (Goodfellow, Bengio, Courville, 2016) — Chapter 20 생성 모델 표준
- **Pattern Recognition and Machine Learning** (Bishop, 2006) — Chapter 9·10 mixture · VI
- **Probabilistic Machine Learning: Advanced Topics** (Murphy, 2023) — 생성 모델 현대적 treatment
- **Elements of Information Theory** (Cover & Thomas, 2006) — KL · JSD · rate-distortion

### 🔁 Autoregressive
- **Pixel Recurrent Neural Networks** (van den Oord, Kalchbrenner, Kavukcuoglu, 2016) — **PixelRNN/PixelCNN 원전**
- **Conditional Image Generation with PixelCNN Decoders** (van den Oord et al., 2016) — Gated PixelCNN
- **WaveNet: A Generative Model for Raw Audio** (van den Oord et al., 2016)
- **Generating Long Sequences with Sparse Transformers** (Child et al., 2019)
- **Language Models are Few-Shot Learners** (Brown et al., 2020) — **GPT-3**

### 🧩 VAE
- **Auto-Encoding Variational Bayes** (Kingma & Welling, 2013) — **VAE 원전**
- **Stochastic Backpropagation and Approximate Inference in Deep Generative Models** (Rezende, Mohamed, Wierstra, 2014) — DLGM
- **β-VAE: Learning Basic Visual Concepts with a Constrained Variational Framework** (Higgins et al., 2017)
- **Neural Discrete Representation Learning** (van den Oord, Vinyals, Kavukcuoglu, 2017) — **VQ-VAE**
- **An Information-Theoretic Analysis of Deep Latent-Variable Models** (Alemi et al., 2018) — Rate-distortion
- **Generating Diverse High-Fidelity Images with VQ-VAE-2** (Razavi, van den Oord, Vinyals, 2019)
- **Don't Blame the ELBO!** (Lucas et al., 2019) — Posterior collapse 분석

### 🌊 Normalizing Flow
- **NICE: Non-linear Independent Components Estimation** (Dinh, Krueger, Bengio, 2015)
- **Density Estimation using Real NVP** (Dinh, Sohl-Dickstein, Bengio, 2017) — **RealNVP**
- **Glow: Generative Flow with Invertible 1×1 Convolutions** (Kingma & Dhariwal, 2018)
- **Masked Autoregressive Flow for Density Estimation** (Papamakarios, Pavlakou, Murray, 2017) — **MAF**
- **Improving Variational Inference with Inverse Autoregressive Flow** (Kingma et al., 2016) — **IAF**
- **Neural Ordinary Differential Equations** (Chen, Rubanova, Bettencourt, Duvenaud, 2018)
- **FFJORD: Free-Form Continuous Dynamics for Scalable Reversible Generative Models** (Grathwohl et al., 2019)
- **Normalizing Flows for Probabilistic Modeling and Inference** (Papamakarios et al., 2021) — 서베이

### ⚔️ GAN
- **Generative Adversarial Nets** (Goodfellow et al., 2014) — **GAN 원전**
- **Unsupervised Representation Learning with Deep Convolutional GANs** (Radford, Metz, Chintala, 2016) — **DCGAN**
- **Wasserstein GAN** (Arjovsky, Chintala, Bottou, 2017)
- **Improved Training of Wasserstein GANs** (Gulrajani et al., 2017) — **WGAN-GP**
- **Spectral Normalization for Generative Adversarial Networks** (Miyato et al., 2018)
- **Progressive Growing of GANs for Improved Quality, Stability, and Variation** (Karras et al., 2018)
- **A Style-Based Generator Architecture for Generative Adversarial Networks** (Karras, Laine, Aila, 2019) — **StyleGAN**
- **Analyzing and Improving the Image Quality of StyleGAN** (Karras et al., 2020) — **StyleGAN2**
- **Large Scale GAN Training for High Fidelity Natural Image Synthesis** (Brock, Donahue, Simonyan, 2019) — **BigGAN**

### 🌫️ Diffusion · Score-Based
- **Deep Unsupervised Learning using Nonequilibrium Thermodynamics** (Sohl-Dickstein et al., 2015) — Diffusion 기원
- **Generative Modeling by Estimating Gradients of the Data Distribution** (Song & Ermon, 2019) — **NCSN**
- **Denoising Diffusion Probabilistic Models** (Ho, Jain, Abbeel, 2020) — **DDPM**
- **Denoising Diffusion Implicit Models** (Song, Meng, Ermon, 2020) — **DDIM**
- **Score-Based Generative Modeling through Stochastic Differential Equations** (Song et al., 2021) — **Score-SDE**
- **Diffusion Models Beat GANs on Image Synthesis** (Dhariwal & Nichol, 2021) — Classifier Guidance
- **Classifier-Free Diffusion Guidance** (Ho & Salimans, 2022)
- **High-Resolution Image Synthesis with Latent Diffusion Models** (Rombach et al., 2022) — **Stable Diffusion**
- **Photorealistic Text-to-Image Diffusion Models with Deep Language Understanding** (Saharia et al., 2022) — **Imagen**
- **Hierarchical Text-Conditional Image Generation with CLIP Latents** (Ramesh et al., 2022) — **DALL-E 2**
- **Scalable Diffusion Models with Transformers** (Peebles & Xie, 2023) — **DiT**
- **Consistency Models** (Song et al., 2023)
- **Flow Matching for Generative Modeling** (Lipman et al., 2023)
- **Rectified Flow** (Liu, Gong, Liu, 2023)
- **Scaling Rectified Flow Transformers for High-Resolution Image Synthesis** (Esser et al., 2024) — **Stable Diffusion 3**

### ⚡ Energy-Based · 이론
- **Training Products of Experts by Minimizing Contrastive Divergence** (Hinton, 2002) — CD
- **A Connection Between Score Matching and Denoising Autoencoders** (Vincent, 2011) — DSM
- **Implicit Generation and Modeling with Energy-Based Models** (Du & Mordatch, 2019)
- **Your Classifier is Secretly an Energy Based Model** (Grathwohl et al., 2020) — **JEM**
- **Reverse-time Diffusion Equation Models** (Anderson, 1982)

### 🌍 응용·Frontier
- **Zero-Shot Text-to-Image Generation** (Ramesh et al., 2021) — **DALL-E**
- **Video Diffusion Models** (Ho et al., 2022)
- **NeRF: Representing Scenes as Neural Radiance Fields for View Synthesis** (Mildenhall et al., 2020)
- **DreamFusion: Text-to-3D using 2D Diffusion** (Poole et al., 2023) — SDS loss
- **Highly Accurate Protein Structure Prediction with AlphaFold** (Jumper et al., 2021)
- **Accurate Structure Prediction of Biomolecular Interactions with AlphaFold 3** (Abramson et al., 2024)

---

<div align="center">

**⭐️ 도움이 되셨다면 Star 를 눌러주세요!**

Made with ❤️ by [IQ AI Lab](https://github.com/iq-ai-lab)

<br/>

*"`StableDiffusionPipeline.from_pretrained(...)` 으로 이미지를 뽑는 것과 — Goodfellow 2014 로 GAN 이 왜 $2 \cdot JSD - \log 4$ 로 환원되는지 · Kingma 2013 으로 ELBO 가 왜 reconstruction + regularization 으로 정확히 분해되는지 · Dinh 2017 로 coupling layer 가 왜 삼각 Jacobian 으로 $\det$ 을 $O(n)$ 에 만드는지 · Ho 2020 으로 ELBO 가 왜 $\|\epsilon - \epsilon_\theta\|^2$ 로 단순화되는지 · Song 2021 로 DDPM · NCSN · Probability Flow ODE 가 왜 하나의 SDE 프레임워크로 통합되는지 · Ho & Salimans 2022 로 CFG 의 $w$ 가 왜 quality-diversity knob 인지 — 이 모든 '왜' 를 직접 유도할 수 있는 것은 다르다"*

</div>
