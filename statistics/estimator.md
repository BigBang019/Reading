# [参数估计](https://zhuanlan.zhihu.com/p/592423240)
*本文采用statistical notation。*

假设$\Theta$为代表未知参数的随机变量，$\mathbf{X}$为观测样本。

$\star$在所有的参数估计的问题里，$\hat{\theta}(\mathbf{X})=g(\mathbf{X})$都是关于观测$\mathbf{X}$的函数。

### Case
袋子里有黑白球，白球比例为$\theta$，我有采样$\mathbf{X}=(x_1, x_2, ..., x_n)^T$，其中$x_i$表示黑球或者白球。本次采样中7次白球，3次黑球。

$f_\Theta(\theta)$为先验分布：描述的是$\theta\in[0,1]$的概率分布。

$f_{\mathbf{X}|\Theta}(\mathbf{x}|\theta)$为似然：给定$\Theta=\theta$，采样的概率分布（从理论到实践：给定$\Theta=\theta$的认知，指导我们对采样$\mathbf{X}$的预测）

$f_{\Theta|\mathbf{X}}(\theta|\mathbf{x})$为后验分布：给定当前采样，$\Theta$的概率分布（从实践到理论：给定一组采样$\mathbf{X}$，我们对$\Theta$的重新认识）

### 最大似然估计
$$
\hat{\theta}=\argmax_\theta f_{\mathbf{X}|\Theta}(\mathbf{x}|\theta)
$$

最大似然估计的思路是选择一个$\theta$使得当前$f_{\mathbf{X}|\Theta}(\mathbf{x}|\theta)$最大。
$$
\begin{equation}
\begin{gathered}
L(\theta)=f_{\mathbf{X}|\Theta}(\mathbf{x}|\theta)=\prod_{i=1}^n f_{X_i|\Theta}(x_i|\theta)\\
l(\theta)=\log f_{\mathbf{X}|\Theta}(\mathbf{x}|\theta) = \sum_{i=1}^n \log f_{X_i|\Theta}(x_i| \theta)
\end{gathered}
\end{equation}
$$

> 例子1: 因为上面的例子里单个采样遵循伯努利分布，则$f_{X_i|\Theta}(x_i|\theta)=\theta^{x_i}(1-\theta)^{1-x_i}$
> $$
> \begin{split}
> l(\theta)=&\sum_{i=1}^n\log f_{X_i|\Theta}(x_i|\theta)\\
> =&\sum_{i=1}^n\log \theta^{x_i}(1-\theta)^{1-x_i}\\
> =&m\log\theta+(n-m)\log(1-\theta)\\
> \end{split}
> $$
> 当$\theta=\frac{1}{n}\sum_{i=1}^nx_i$，上式子取最大值。

> 例子2: 假设现在是对一个高斯分布$\mathcal{N}(\mu, \sigma^2)$作参数估计，我们的采样有$X=\{x_1, x_2, ..., x_{n}\}$，那么$f_{X_i|\Theta}(x_i|\theta)=\frac{1}{\sqrt{2\pi}\sigma}\exp(-\frac{(x_i-\mu)^2}{2\sigma^2})$
> $$
> \begin{split}
> l(\mu, \sigma)=&\sum_{i=1}^n\log f_{X_i|\Theta}(x_i|\theta)\\
> =&\sum_{i=1}^n[-\frac{1}{2}\log 2\pi-\log \sigma-\frac{(x_i-\theta)^2}{2\sigma^2}]\\
> &=-\frac{n}{2}\log 2\pi-\frac{n}{2}\log \sigma^2-\frac{(x_i-\theta)^2}{2\sigma^2}\\
> \end{split}
> $$
> 上式子分别对$\sigma$和$\mu$求导，可得当$\mu=\frac{1}{n}\sum_{i=1}^{n}x_i$并且$\sigma^2=\frac{\sum_{i=1}^n(x_i-\bar{X})}{n}$时，上式子求得最大值。

### 最大后验估计
$$
\hat{\theta}=\argmax_\theta f_{\Theta|\mathbf{X}}(\theta|\mathbf{x})
$$

最大后验分布的思路是既然我们已知当前观测$\mathbf{X}$，根据当前观测选择一个$\theta$使得$f_{\Theta|\mathbf{X}}(\theta|\mathbf{x})$最大。（为什么这个地方默认是$f_\mathbf{X}(\mathbf{x})$是常数）
$$
\begin{equation}
\argmax_\theta f_{\Theta|\mathbf{X}}(\theta|\mathbf{x})= \argmax_\theta \frac{f_{\Theta,\mathbf{X}}(\theta, \mathbf{x})}{f_\mathbf{X}(\mathbf{x})}=\argmax_\theta \frac{f_{\mathbf{X}|\Theta}(\mathbf{x}|\theta)f_\Theta(\theta)}{f_\mathbf{X}(\mathbf{x})}\propto f_{\mathbf{X}|\Theta}(\mathbf{x}|\theta)f_\Theta(\theta)
\end{equation}
$$

不同于MLE，MAE需要我们知道先验分布$f_\Theta(\theta)$

> 因为上面的例子里单个采样遵循伯努利分布，则$f_{X_i|\Theta}(x_i|\theta)=\theta^{x_i}(1-\theta)^{1-x_i}$，我们假设$f_\Theta(\theta)=2\theta$。 
> 
> $f_{\mathbf{X}|\Theta}(\mathbf{x}|\theta)f_\Theta(\theta)=\prod_{i=1}^{10}f_{X_i|\Theta}(x_i|\theta)*2\theta=2\theta^8(1-\theta)^3$
> 
> 上式最大时，$\theta=0.73$

### 贝叶斯估计

贝叶斯估计希望求得分布$f_{\Theta|\mathbf{X}}(\theta|\mathbf{x})$
$$
\begin{equation}
f_{\Theta|\mathbf{X}}(\theta|\mathbf{x})=\frac{f_{\Theta|\mathbf{X}}(\mathbf{x}|\theta)f_{\Theta}(\theta)}{\int f_{\mathbf{X}|\Theta}(\mathbf{x}|\theta)f_\Theta(\theta)d\theta}
\end{equation}
$$

上式得求解需要关于$f_{\mathbf{X}|\Theta}(\mathbf{x}|\theta)$和$f_\Theta(\theta)$的分析，在一开始的例子中，我们可以得到$f_{\mathbf{X}|\Theta}(\mathbf{x}|\theta)$遵循伯努利分布；而先验分布$f_\Theta(\theta)$在大多数情况下只能采用假设的方式。

### MMSE
$$
\hat{\theta}=\text{average of }f_{\Theta|\mathbf{X}}(\theta|\mathbf{x})
$$
是贝叶斯估计的一种。其目标函数为：
$$
\begin{equation}
\begin{gathered}
Loss=\left(\hat{\theta}-\theta\right)^2\\
\text{where} \quad \hat{\theta}(\mathbf{X})=g(\mathbf{X})
\end{gathered}
\end{equation}
$$
在任意的参数估计问题中，$\hat{\theta}$都是关于$\mathbf{X}$的函数。

上式子令人困惑的点在于：$\theta,\mathbf{X}$分别是$f_\Theta(\theta)$和$f_\mathbf{X}(\mathbf{X})$的采样，定义在他们上面的MSE没有意义（或者无法解读）。

我们跟随[Purdue的slides](https://probability4datascience.com/slides/Slide_8_04.pdf)将上述目标函数标准化：
$$
\begin{equation}
\begin{split}
Loss =&\left(\hat{\theta}-\theta\right)^2\\
=&\left(g(\mathbf{X})-\theta \right)^2\\
=&\mathbb{E}_{\mathbf{X},\Theta}\left[(g(\mathbf{X})-\Theta)^2\right]\\
\end{split}
\end{equation}
$$

最终的优化目标是
$$
\begin{equation}
\begin{split}
\hat{\theta}=&\argmin_{g(\cdot)} \mathbb{E}_{\Theta,\mathbf{X}} \left[ (\Theta-g(\mathbf{X}))^2\right]\\
=&\mathbb{E}_{\Theta|\mathbf{X}}\left[\Theta|\mathbf{X}=\mathbf{x}\right]
\end{split}
\end{equation}
$$
> 证明
> $$
> \begin{equation*}
> \begin{split}
> Loss =& \mathbb{E}_{\Theta,\mathbf{X}} \left[ (\Theta-g(\mathbf{X}))^2\right]\\
> &=\int \mathbb{E}_{\Theta|\mathbf{X}}\left[(\Theta-g(\mathbf{X}))^2|\mathbf{X}=\mathbf{x} \right] f_{\mathbf{X}}(\mathbf{x})\text{d}\mathbf{x}
> \end{split}
> \end{equation*}
> $$
> 因为$f_{\mathbf{X}}(\mathbf{x})\text{d}\mathbf{x}\geq0$，$\mathbb{E}_{\Theta|\mathbf{X}}\left[(\Theta-g(\mathbf{X}))^2|\mathbf{X}=\mathbf{x} \right]\geq 0$。
> 
> 所以最小化$Loss$要最小化$\mathbb{E}_{\Theta|\mathbf{X}}\left[(\Theta-g(\mathbf{X}))^2|\mathbf{X}=\mathbf{x} \right]$。
> 
> $$
> \begin{equation*}
> \begin{split}
> \mathbb{E}&_{\Theta|\mathbf{X}}\left[(\Theta-g(\mathbf{X}))^2|\mathbf{X}=\mathbf{x} \right]\\
> &=\mathbb{E}_{\Theta|\mathbf{X}}\left[\Theta^2-2\Theta g(\mathbf{X})+g(\mathbf{X})^2 \right]\\
> &=\underbrace{\mathbb{E}_{\Theta|\mathbf{X}}\left[\Theta^2| \mathbf{X}=\mathbf{x}\right]}_{V(\mathbf{x})}-\underbrace{2\mathbb{E}_{\Theta|\mathbf{X}}\left[\Theta|\mathbf{X}=\mathbf{x}\right]g(\mathbf{x})}_{u(\mathbf{x})} + g(\mathbf{x})^2\\
> &=V(\mathbf{x})-2u(\mathbf{x})g(\mathbf{x})+g(\mathbf{x})^2+u(\mathbf{x})^2-u(\mathbf{x})^2\\
> &=V(\mathbf{x})-u(\mathbf{x})^2+(u(\mathbf{x})-g(\mathbf{x}))^2\\
> &\geq V(\mathbf{x})-u(\mathbf{x})^2, \quad \forall g(\mathbf{x})
> \end{split}
> \end{equation*}
> $$
> 所以当$u(\mathbf{x})=g(\mathbf{x})$时，上式达到最小，即$g(\mathbf{x})=\mathbb{E}_{\Theta|\mathbf{X}}\left[\Theta|\mathbf{X}=\mathbf{x}\right]$。

# 无偏估计

