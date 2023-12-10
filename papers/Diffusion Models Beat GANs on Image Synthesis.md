# Diffusion Models Beat GANs on Image Synthesis

*本文采用machine leanring notation*

### Method

首先定义Conditional Markovian noising process $\hat{q}$，并且假设给定任意图片$x_0$，它的label $y$是已知的：
$$
\begin{align}
\hat{q}(x_0)&:=q(x_0)\\
\hat{q}(y|x_0)&:=\text{Known labels per sample}\\
\hat{q}(x_{t+1}|x_t, y)&:=q(x_{t+1}|x_t)\\
\hat{q}(x_{1:T}|x_0, y)&:=\prod_{t=1}^T \hat{q}(x_t|x_{t-1},y)
\end{align}
$$

证明$\hat{q}(x_{t+1}|x_t,y)=\hat{q}(x_{t+1}|x_t)$，即Conditional Markovian noising process $\hat{q}$的前向过程和$y$无关：

证明$\hat{q}(x_{1:T}|x_0)=q(x_{1:T}|x_0)$：

证明$\hat{q}(x_t)=q(x_t)$：

证明$\hat{q}(y|x_t, x_{t+1})=\hat{q}(y|x_t)$：

求反向过程$\hat{q}(x_t| x_{t+1}, y)$：
$$
\begin{align}
\hat{q}(y|x_t|x_{t+1},y)&=\frac{\hat{q}(x_t, x_{t+1},y)}{\hat{q}(x_{t+1}, y)}\\
&=\frac{\hat{q}(x_t,x_{t+1},y)}{\hat{q}(y|x_{t+1})\hat{q}(x_{t+1})}\\
&=\frac{\hat{q}(x_t|x_{t+1})\hat{q}(y|x_t, x_{t+1})\hat{q}(x_{t+1})}{\hat{q}(y|x_{t+1})\hat{q}(x_{t+1})}\\
&=\frac{\hat{q}(x_t|x_{t+1})\hat{q}(y|x_t, x_{t+1})}{\hat{q}(y|x_{t+1})}\\
&=\frac{\hat{q}(x_t|x_{t+1})\hat{q}(y|x_t)}{\hat{q}(y|x_{t+1})}\\
&=\frac{q(x_t|x_{t+1})\hat{q}(y|x_t)}{\hat{q}(y|x_{t+1})}
\end{align}
$$
其中$\hat{q}(y|x_{t+1})$是常数，$q(x_t|x_{t+1})$已经被$p_\theta(x_t|x_{t+1})$拟合，只有$\hat{q}(y|x_t)$未知。我们只需要通过给$x_t$分配标签$y$的方式训练$p_\phi(y|x_t)$就可以完成$\hat{q}(y|x_t)$的拟合。这种情况下我们只需要从$Zp_\theta(x_t|x_{t+1})p_\phi(y|x_t)$中进行采样就可以完成反向过程。

### 反向过程采样
如何采样$p_\theta(x_t|x_{t+1})p_\phi(y|x_t)$？

已知$p_\phi(x_t|x_{t+1})$

$$
\begin{align}
p_\theta(x_t|x_{t+1})&=\mathcal{N}(\mu, \Sigma)=\sqrt{\frac{1}{(2\pi)^n\text{det}(\Sigma)}}\exp\left(-\frac{1}{2}(x-\mu)^T\Sigma^{-1}(x-\mu)\right)\\
\log p_\theta(x_t|x_{t+1})&=-\frac{1}{2}(x_t-\mu)^T\Sigma^{-1}(x_t-\mu)+C
\end{align}
$$

对于$\hat{q}(y|x_t)$，作者采用了在$x=\mu$泰勒展开的方式拟合$p_\phi(y|x_t)$（可以这么做的前置条件我还没看懂）：
$$
\begin{align}
\log p_\phi(y|x)\approx \log p_\phi(y|x_t)|_{x_t=\mu}
\end{align}
$$