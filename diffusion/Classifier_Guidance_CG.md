Diffusion Models Beat GANs on Image Synthesis. NeurIPS 2021.

# Classifier Guidance Takeaway

*本文采用machine leanring notation*


定义前向过程：
$$
\begin{align}
\hat{q}(x_{t}|x_{t-1}, y)&:=q(x_{t}|x_{t-1})=\mathcal{N}(\sqrt{\alpha_t}x_{t-1}, (1-\alpha_t)\mathbf{I})
\end{align}
$$

根据tweedie's formula和前向过程，我们有
$$
\begin{align}
\sqrt{\alpha_t}x_{t-1}&=x_t+(1-\alpha_t) \nabla_{x_t}\log q(x_t | y)\\
x_t&=\sqrt{\alpha_t}x_{t-1}+\sqrt{1-\alpha_t}\epsilon_t
\end{align}
$$
结合上式我们可以得到：
$$
\begin{align}
\epsilon_t&=-\sqrt{1-\bar{\alpha}_t}\nabla_{x_t}\log q(x_t|y)\\
&=-\sqrt{1-\bar{\alpha}_t}\nabla_{x_t}\log \frac{q(y|x_t)q(x_t)}{q(y)}\\
&=-\sqrt{1-\bar{\alpha}_t}\nabla_{x_t}\log q(y|x_t)q(x_t)\\
&=-\sqrt{1-\bar{\alpha}_t}\nabla_{x_t}\log q(y|x_t)-\sqrt{1-\bar{\alpha}_t}\nabla_{x_t}\log q(x_t)\\
&\approx-\sqrt{1-\bar{\alpha}_t}\nabla_{x_t}\log q(y|x_t)+\epsilon_\theta(x_t,t)
\end{align}
$$
所以求解拟合此时的真实噪声$\epsilon_t$只需要求解$\nabla_{x_t}\log q(y|x_t)$。那么$q(y|x_t)$是什么？给定任意（加噪）图片$x_t$，给出他的class label $y$，也就是说我们需要额外训练一个neural classifier $p_\phi(y|x_t)$。

所以我们可以定义拟合的噪声
$$
\begin{align}
\hat{\epsilon}(x_t, t, y):=\epsilon_\theta(x_t,t)-\sqrt{1-\bar{\alpha}_t}\nabla_{x_t}\log p_{\phi}(y|x_t)
\end{align}
$$
可以直接将上述拟合噪声带入[DDPM](DDPM.md)/[DDIM](DDIM.md)的反向过程。

# QA

### 如何理解$\nabla_{x_t}\log q(x_t|y)$

$$
\begin{align}
\nabla_{x_t}\log q(x_t|y)&=\nabla_{x_t}\log \frac{q(y|x_t)q(x_t)}{q(y)}\\
&=\underbrace{\nabla_{x_t}\log q(y|x_t)}_{\text{classifier gradient}}+\underbrace{\nabla_{x_t}\log q(x_t)}_{\text{unconditional score}}\\
\end{align}
$$
原文里有提到过guidance scale版的预测噪声本质是在增大$\text{classifier gradient}$
$$
\nabla_{x_t}\log q(x_t|y)=s\nabla_{x_t}\log q(y|x_t)+\nabla_{x_t}\log q(x_t)
$$


<!-- ### Reference

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
\hat{q}(x_t|x_{t+1},y)&=\frac{\hat{q}(x_t, x_{t+1},y)}{\hat{q}(x_{t+1}, y)}\\
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
\log p_\phi(y|x)&\approx \log p_\phi(y|x_t)|_{x_t=\mu}+(x_t-\mu)\nabla_{x_t}\log p_\phi(y|x_t)|_{x_t=\mu}\\
&=(x_t-\mu)g+C_1
\end{align}
$$
这里$g=\nabla_{x_t}\log p_\phi(y|x_t)|_{x_t=\mu}$，所以
$$
\begin{aligned}
\log (p_\theta(x_t|x_{t+1})p_\phi(y|x_t))&\approx-\frac{1}{2}(x_t-\mu)^T\Sigma^{-1}(x_t-\mu)+(x_t-\mu)g+C_2\\
&=-\frac{1}{2}(x_t-\mu-\Sigma g)^T\Sigma^{-1}(x_t-\mu-\Sigma g)+\frac{1}{2}g^T\Sigma g+C_2\\
&=-\frac{1}{2}(x_t-\mu-\Sigma g)^T\Sigma^{-1}(x_t-\mu-\Sigma g)+C_3\\
&=\log p(z)+C_4,\ z\sim \mathcal{N}(\mu+\Sigma g, \Sigma)
\end{aligned}
$$ -->

