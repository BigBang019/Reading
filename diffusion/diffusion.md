# Diffusion

### 问题和思路

问题描述：$X$是一组采样，取自未知分布$p(x_0)$，我们希望获重建它的分布$p(x_0)$。

但是$p(x_0)$没有显式表达式，所以像[VAE](https://zhuanlan.zhihu.com/p/575984592)问题一样，diffusion引入一个隐变量$z$来辅助重建。

如果我们仅仅是重建$p_{\theta}(x_0|z)$的话，那么其实和VAE也没有什么区别。Diffusion的思想是用$T$个$p_{\theta}(x_{t}|x_{t-1})$去拟合$p_{\theta}(x_0|z)$。

$$
p(x_0|x_T)=\prod_{t}^T p(x_{t}|x_{t-1})
$$

我们最终希望获得的一个函数就是$p_{\theta}(x_0|x_T)$。其中$x_T\sim \mathcal{N}(0,1)$为高斯分布，$x_0\sim p(x_0)$为$X$所在的原始分布。

### 建模

<img src="pasteImage/diffusion.webp" style="display: block; margin-left: auto; margin-right: auto; width: 75%;"/>

前向传播和后向传播
$$
\begin{equation}
    \begin{gathered}
        q(x_{0:T})=p(x_0)\prod_{t=1}^T q(x_{t}|x_{t-1})\\
        p(x_{0:T})=p(x_T)\prod_{t=1}^T p_{\theta}(x_{t-1}|x_{t})
    \end{gathered}
\end{equation}
$$
其中每一段拟合$q(x_{t}|x_{t-1})$都用高斯分布，关于前向传播的两种表示方式：
$$
\begin{equation}
    \begin{gathered}
        q(x_{t}|x_{t-1}):=\mathcal{N}(x_t; a_{t}x_{t-1}, b_{t}\mathbf{I})
    \end{gathered}
\end{equation}
$$

$$
\begin{equation}
    \begin{gathered}
        q(x_{t}|x_{t-1}):=a_{t}x_{t-1} + b_{t}\epsilon_t, \quad \epsilon_t \sim \mathcal{N}(0, \mathbf{I})
    \end{gathered}
\end{equation}
$$

这个地方加一个约束$a_t^2+b_t^2=1$可以简化$x_t$的表达式。

$$
\begin{equation}
    x_t = (a_t...a_1)x_0 + \sqrt{1-(a_t...a_1)^2}\bar{\epsilon}, \quad \bar{\epsilon}\sim \mathcal{N}(0, \mathbf{I})
\end{equation}
$$

不妨用$\sqrt{\alpha_t}$代替$a_t$，并且而外定义$\beta_t:=1-\alpha_t$，$\bar{\alpha}_t=\prod_{t=1}^T \alpha_t$那么
$$
\begin{equation}
    \begin{gathered}
        x_t = \sqrt{\alpha_t}x_{t-1} + \sqrt{1-\alpha_t}\epsilon_t, \quad \epsilon_t \sim \mathcal{N}(0, \mathbf{I})\\
        x_t = \sqrt{\bar{\alpha}_t}x_0+\sqrt{1-\bar{\alpha}_t}\epsilon_t, \quad \epsilon_t\sim \mathcal{N}(0,\mathbf{I})
    \end{gathered}
\end{equation}
$$


### 求解

我们本质上是想要求解$p(x_0)$的分布，实践上我们是要重建$p_{\theta}(x_0|x_t)$，但是我们还是可以从$p(x_0)$的**最大似然估计**入手。


$$
\begin{equation}
    \begin{split}
        \log{p(x_0)} \geq & \mathbb{E}_{q(x_{1:T}|x_0)}\left[\log{\frac{p(x_{0:T})}{q(x_{1:T}|x_0)}}\right]\\
        = & \mathbb{E}_{q(x_{1:T}|x_0)}\left[\log{p(x_T)} + \sum_{t\geq 1}\log{\frac{p_{\theta}(x_{t-1}|x_t)}{q(x_t|x_{t-1})}}\right]=:L\\
        = & \mathbb{E}_{q(x_{1:T}|x_0)}\left[\underbrace{D_{KL}(q(x_T|x_0)\left|\right| p(x_T))}_{L_T} + \sum_{t\geq 1}\underbrace{D_{KL}(q(x_{t-1}|x_t,x_0) \left|\right| p_{\theta}(x_{t-1}|x_t))}_{L_{t-1}} - \underbrace{\log{p(x_0|x_1)}}_{L_0}\right]
    \end{split}
\end{equation}
$$

### 求解$L_T$
对于$L_T$，原论文里直接设$a_t$为常数，又因为
$$
\begin{equation}
    \begin{gathered}
        D_{KL}(\mathcal{N}(x; \mu_x, \Sigma_x) \left|\right| \mathcal{N}(y; \mu_y, \Sigma_y))\\
        =\frac{1}{2}\left[ \log{\frac{\left| \Sigma_y\right|}{\left| \Sigma_x\right|}} - d + tr(\Sigma_y^{-1}\Sigma_x)+(\mu_y-\mu_x)^T \Sigma_y^{-1}(\mu_y-\mu_x)\right]
    \end{gathered}
\end{equation}
$$
所以$L_T=D_{KL}(\mathcal{N}(\sqrt{\bar{\alpha}_t} x_0, (1-\bar{\alpha}_t)\mathbf{I}) \left|\right| \mathcal{N}(0, \mathbf{I}))$为学习无关的常数（式5&7）。

### 求解$L_{t-1}$

对于$L_{t-1}$，需要建模$q(x_{t-1}|x_t, x_0)$和$p_{\theta}(x_{t-1}|x_t)$。其中
$$
\begin{equation}
    \begin{gathered}
        q(x_{t-1}|x_t, x_0) = \mathcal{N}(\tilde{\mu}_t(x_t, x_0), \tilde{\beta}_t \mathbf{I})\\
        \text{where} \quad \tilde{\mu}_t(x_t, x_0):=\frac{\sqrt{\bar{\alpha}_{t-1}}\beta_t}{1-\bar{\alpha}_t}x_0+\frac{\sqrt{a_t}(1-\bar{\alpha}_{t-1})}{1-\bar{\alpha}_t}x_t \quad \text{and} \quad \tilde{\beta}_t:=\frac{1-\bar{\alpha}_{t-1}}{1-\bar{\alpha}_t}\beta_t
    \end{gathered}
\end{equation}
$$

因为$q(x_{t-1}|x_t, x_0)$遵循高斯分布（式8），不妨用高斯分布建模$p_{\theta}(x_{t-1}|x_t):=\mathcal{N}(\mu_\theta(x_t, t), \Sigma_\theta(x_t, t))$。原论文中对于$\Sigma_\theta(x_t, t)$的很简单，从$\tilde{\beta}_t$和$\beta_t$中选择了一个。不妨选择$\Sigma_\theta(x_t, t):=\Sigma(x_t, t)=\tilde{\beta}_t$。那么$L_{t-1}$就变成了如下

$$
\begin{equation}
    \begin{split}
        L_{t-1}&=D_{KL}(q(x_{t-1}|x_t, x_0) \left|\right| p_\theta(x_{t-1}|x_t))\\
        &=D_{KL}(\mathcal{N}(x_{t-1}; \tilde{\mu}_t(x_t, x_0), \tilde{\beta}_t\mathbf{I}) \left|\right| \mathcal{N}(x_{t-1}; \mu_\theta(x_t, t), \tilde{\beta}_t\mathbf{I}))\\
        &=\frac{1}{2\tilde{\beta}_t^2}\left [ \left|\right| \tilde{\mu}_t(x_t, x_0) - \mu_\theta(x_t, t) \left|\right|_2^2\right]
    \end{split}
\end{equation}
$$

##### 如何将上式作为优化目标写成能用神经网络来理解的形式
首先$\tilde{\mu}_t$可以又如下两种表示
$$
\begin{equation}
    \begin{split}
        \tilde{\mu}_t(x_t, x_0)&=\frac{\sqrt{\bar{\alpha}_{t-1}}\beta_t}{1-\bar{\alpha}_t}x_0+\frac{\sqrt{a_t}(1-\bar{\alpha}_{t-1})}{1-\bar{\alpha}_t}x_t\\
        &=\frac{1}{\sqrt{\alpha_t}}x_t-\frac{1-\alpha_t}{\sqrt{1-\bar{\alpha}_t}\sqrt{\alpha_t}}\epsilon_t
    \end{split}
\end{equation}
$$

$\mu_\theta(x_t, t)$是关于$x_t$和$t$的函数，所以我们需要学习一个映射$f_\theta:(x_t,t)\rightarrow x_0 \ \text{or}\ \epsilon_t$。

学习$x_0$，即$\hat{x_0}=f_\theta(x_t, t)$，式9会简化成如下形式：
$$
\begin{equation}
    \frac{\bar{\alpha}_{t-1}\beta_t^2}{2\tilde{\beta_t^2}(1-\bar{\alpha}_t)^2}\left[ \left|\right| f_\theta(x_t, t)-x_0\left|\right|_2^2\right]
\end{equation}
$$
学习$\epsilon_t$，即$\hat{\epsilon_t}=f_\theta(x_t, t)$，式9会简化成如下形式：
$$
\begin{equation}
    \frac{\beta_t^2}{2\tilde{\beta_t^2}(1-\bar{\alpha}_t)\alpha_t}\left[ \left|\right| f_\theta(x_t, t)-\epsilon_t\left|\right|_2^2\right]
\end{equation}
$$