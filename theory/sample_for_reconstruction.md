# Sampling and Reconstruction for Simple Fourier Signals
[[paper](https://dl.acm.org/doi/abs/10.1145/3313276.3316363?casa_token=yuCtUJeo_t8AAAAA:MKfhpMxo7Uw4CaVM4og90vb84Mh0BN6ktkDZ1OyEtpzqRjMTFW29U_q5vbSKKMARYVtNUCjocRw)]

## Problem

给定信号$y(t), t\in [0, T]$，和$\mathbb{R}$空间上的probability measure $\mu$，设计基于$\mu$的采样算法和重建算法，使得重建的$\tilde{y}$满足：
$$
\begin{equation}
    \|y-\tilde{y}\|_T^2 \leq \epsilon \|x\|_\mu^2+C\|n\|_T^2
\end{equation}
$$
其中$\epsilon$为精度系数。

## Theory

#### 优化Eq. (1)转化为Eq. (2)
为了解决eq. (1)，作者提出优化下式
$$
\begin{equation}
    \min_{g\in L_2(\mu)}\|\mathcal{F}_\mu^*g-(y+n)\|_T^2 + \epsilon \|g\|_\mu^2
\end{equation}
$$
其中$g\in L_2(\mu)$是一个频域信号，它的逆傅立叶变换$\mathcal{F}_\mu^*g$接近加了噪声的原信号$y+n$，并且作者证明了如果能找到一个$\tilde{g}$使得它是eq. (2)的$C\text{-approximation}$。
$$
\begin{equation}
    \begin{split}
        &\|\mathcal{F}_\mu^*\tilde{g}-(y+n)\|_T^2+\epsilon\|\tilde{g}\|_\mu^2 \\
        &\leq C\cdot \min_{g\in L_2(\mu)} \left[ \|\mathcal{F}_\mu^* g - (y+n)\|_T^2+\epsilon\|g\|_\mu^2 \right]
    \end{split}
\end{equation}
$$
那么它就相当于
$$
\begin{equation}
    \|\mathcal{F}_\mu^*\tilde{g}-y\|_T^2\leq 2C\epsilon \|x\|_\mu^2 + 2(C+1)\|n\|_T^2
\end{equation}
$$
到目前为止的分析，$g,\tilde{g}\in L_2(\mu)$都定义在连续空间上的函数。但是作者是希望从$s$个采样重建完整信号。

#### 从离散采样优化出定义在连续空间上的$\tilde{g}$

连续空间上的逆傅立叶变换：
$$
\begin{equation}
    \begin{split}
        &\left[\mathcal{F}_\mu^*g \right ](t)=\int_{\mathbb{R}}g(w)e^{2\pi iw t} d\mu(w)\\
        &\text{where } \mathcal{F}_\mu^*: L_2(\mu) \mapsto L_2(\mu)
    \end{split}
\end{equation}
$$
现在，首先我们要获得$s=c\cdot \tilde{s}_{\mu,\epsilon}\cdot(\log{\tilde{s}_{\mu, \epsilon}+1/\delta})$个采样点$\{t_1, ..., t_s\}$，从$[0,T]$上按照正比于$\tilde{\tau}_{\mu,\epsilon}$的概率采样。作者定义根据离散采样点进行的傅立叶变换：
$$
\begin{equation}
    \begin{split}
        &\left [\mathbf{F} g\right ](w)=\sum_{j=1}^s w_j\cdot g(j)\cdot e^{-2\pi i w t_j}\\
        &\text{where } w_j = \sqrt{\frac{1}{sT}\cdot \frac{\tilde{s}_{\mu,\epsilon}}{\tilde{\tau}_{\mu, \epsilon}(t_j)}},\mathbf{F}:\mathbb{C}^s\mapsto L_2(\mu)
    \end{split}
\end{equation}
$$
此时我理解的$g: \mathbb{N}\mapsto\mathbb{C}$是一个时域采样。$\mathbf{F}^*:L_2(\mu)\mapsto \mathbb{C}^s$是$\mathbf{F}$的共轭转置，但是我不知道为什么作者没给他的公式。

优化问题变成了优化：
$$
\begin{equation}
    \tilde{g}=\argmin_{g\in L_2(\mu)}\left [ \|\mathbf{F}^*g-(\mathbf{y}-\mathbf{n})\|_2^2+\epsilon\|g\|_\mu^2\right]
\end{equation}
$$
作者证明了求解eq. (7)得到的结果是eq. (2)的$\frac{1}{3}\mathrm{-approximation}$，并具有$\geq 1-\delta$的可信度。

#### 求解
|Symbol|Meaning|是否能在实际中算出来|
|---|---|---|
|$\mathcal{K}_\mu$|$\left[\mathcal{K}_\mu z\right](t)=\int_{w\in\mathbb{R}}e^{2\pi iwt}\left[ \frac{1}{T}\int_{s\in [0,T]}z(s)e^{-2\pi iwt} dt\right]d\mu(w)$||
|$s_{\mu, \epsilon}$| $s_{\mu, \epsilon}\stackrel{\mathrm{def}}{=}\sum_{i=1}^{\infty}\frac{\lambda_i(\mathcal{K}_\mu)}{\lambda_i(\mathcal{K}_\mu)+\epsilon}$<br>$s_{\mu, \epsilon}=\int_0^T\tau_{\mu, \epsilon}(t)dt$|不可以|
|$\tau_{\mu, \epsilon}(t)$|$\tau_{\mu, \epsilon}(t)\stackrel{\mathrm{def}}{=}\frac{1}{T}\cdot \max_{\{\alpha \in L_2(\mu):\|\alpha\|_\mu>0\}}\frac{\|\left[ \mathcal{F}_\mu^*\alpha \right](t)\|^2}{\|\mathcal{F}_\mu^*\alpha \|_T^2+\epsilon \|\alpha\|_\mu^2}$<br> $\tau_{\mu, \epsilon}(t)=\frac{1}{T}\cdot \left< \varphi_t, (\mathcal{G}_\mu + \epsilon \mathcal{I}_\mu)^{-1}\varphi_t\right>_\mu$<br>$\tau_{\mu, \epsilon}(t)=\frac{1}{T}\cdot \min_{\beta\in L_2(T)}\frac{\|\mathcal{F}_\mu\beta - \varphi_t\|_\mu^2}{\epsilon}+\|\beta\|_T^2$|不可以，因为这是$t$在所有$\mu$中的upperbound|
|$\varphi_t(w)$|$\varphi_t(w)\stackrel{\mathrm{def}}{=}e^{-2\pi iwt}$||

首先要解决的第一个问题是采样数目$s_{\mu, \epsilon}$的大小。作者定义的
$$
\begin{equation}
    \begin{split}
        &\tilde{s}_{\mu,\epsilon}=\int_{0}^T\tilde{\tau}_{\mu, \epsilon}(t)dt\\
        &\text{where }\tilde{\tau}_{\mu, \epsilon}(t) \geq \tau_{\mu, \epsilon}(t)
    \end{split}
\end{equation}
$$
为了使得$\tilde{s}_{\mu,\epsilon}$可解，作者在Theorem 17里定义可解的$\tilde{\tau}_\alpha(t)\geq \tau_{\mu, \epsilon}(t)$。


<!-- ## Algorithm

采样
- Sample $s=c\cdot \tilde{s}_{\mu, \epsilon} \cdot (\log{\tilde{s}_{\mu, \epsilon} + \frac{1}{\delta}})$个样本
  - 其中$\tilde{s}_{\mu, \epsilon}=\int_0^T \tilde{\tau}_{\mu, \epsilon}$
-  -->