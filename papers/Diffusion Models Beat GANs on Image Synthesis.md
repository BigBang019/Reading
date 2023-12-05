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