本文采用statistical notation
# [Tweedie estimator](https://zhuanlan.zhihu.com/p/594007789)

Tweedie estimator希望能在不假设先验分布$p(\theta)$的情况下进行参数估计$p(\theta|X)$。


### 背景
贝叶斯估计是根据一组观测$X$预测参数$\theta$，即后验分布$p(\theta|X)$。

我们如果用神经网络$\hat{\theta}(X)$来预测$\theta$，并用MSE来作收敛：
$$
\begin{equation}
Loss = \mathbb{E}\left[ (\hat{\theta}(X)- \theta)^2\right]
\end{equation}
$$

最小化上式等价于求解下式：
$$
\begin{equation}
\hat{\theta}(X)=\mathbb{E}\left [ \theta|X\right] = \int \theta p(\theta|X) d\theta
\end{equation}
$$

> 证明如下
> $$
\begin{align*}
\mathbb{E}\left[(\hat{\theta}(X)-\theta)^2 \right] &= \mathbb{E}\left[(\hat{\theta}(X)-\mathbb{E}[\theta|X]+\mathbb{E}[\theta|X]-\theta)^2\right] \\
&= \mathbb{E}\left[(\hat{\theta}(X)-\mathbb{E}[\theta|X])^2 + 2(\hat{\theta}(X)-\mathbb{E}[\theta|X])(\mathbb{E}[\theta|X]-\theta) + (\mathbb{E}[\theta|X]-\theta)^2 \right] \\
&= \mathbb{E}\left[(\hat{\theta}(X)-\mathbb{E}[\theta|X])^2\right] + \mathbb{E}[2(\hat{\theta}(X)-\mathbb{E}\left[\theta|X])(\mathbb{E}[\theta|X]-\theta)\right] + \mathbb{E}\left[(\mathbb{E}[\theta|X]-\theta)^2\right] \\
&= \mathbb{E}\left[(\hat{\theta}(X)-\mathbb{E}[\theta|X])^2\right] + 2\mathbb{E}\left[(\hat{\theta}(X)-\mathbb{E}[\theta|X])\right]\mathbb{E}\left[\mathbb{E}\left[\theta|X\right]-\theta\right] + \mathbb{E}\left[(\mathbb{E}\left[\theta|X\right]-\theta)^2\right] \\
&= \mathbb{E}\left[(\hat{\theta}(X)-\mathbb{E}[\theta|X])^2\right] + 2\mathbb{E}\left[(\hat{\theta}(X)-\mathbb{E}[\theta|X])\right]\cdot0 + \mathbb{E}\left[(\mathbb{E}[\theta|X]-\theta)^2\right] \\
&= \mathbb{E}\left[(\hat{\theta}(X)-\mathbb{E}[\theta|X])^2\right] + \mathbb{E}\left[(\mathbb{E}[\theta|X]-\theta)^2\right] \\
&= \mathbb{E}\left[(\hat{\theta}(X)-\mathbb{E}[\theta|X])^2\right] + \mathbb{E}\left[\mathbb{E}[(\theta-\mathbb{E}[\theta|X])^2|X]\right] \\
&= \mathbb{E}\left[(\hat{\theta}(X)-\mathbb{E}[\theta|X])^2\right] + \mathbb{E}\left[(\theta-\mathbb{E}[\theta|X])^2\right]\\
&= \mathbb{E}\left[(\hat{\theta}(X)-\mathbb{E}[\theta|X])^2\right]
\end{align*}
$$



### Tweedie estimator
我们不假设先验分布$p(\theta)$，但我们仍需要似然分布$p(X|\theta)$，一般情况下似然分布能直接根据采样环境建模，我们这里不妨假设$p(X|\theta)\sim \mathcal{N}(\mu, \sigma)$遵循高斯分布。

$$
\begin{equation}
\begin{split}
\mathbb{E}\left[\theta|X\right] &=\int \theta p(\theta|X) d\theta\\
&=\int \theta\frac{p(X|\theta)p(\theta)}{p(x)}d\theta\\
&=\frac{\int \theta p(X|\theta)p(\theta) d\theta}{p(x)}\\
&=...\\
&=x+\sigma^2 \frac{d}{dx} p(x)
\end{split}
\end{equation}
$$

其中[Stanford的slides](https://efron.ckirby.su.domains/talks/2010TweediesFormula.pdf)标记$x$为当前采样的MLE（均值？）。

所以当似然分布$p(X|\theta)= \mathcal{N}(\mu, \sigma)$遵循高斯分布，我们发现后验分布期望$\mathbb{E}[\theta|X]$与先验分布$p(\theta)$无关。


### Case

从高斯分布$\mathcal{N}(\mu, \sigma)$中获得一组采样$X=\{x_1, x_2, ..., x_n\}$。这组采样其实可以理解为$X|\mu\sim \mathcal{N}(\mu_0, \sigma)$。

所以我们有$\mathbb{E}\left[ \theta|X\right]=x+\sigma^2 \nabla p(x)$。

