# Diffusion (with multiple noise perturbations)
*本文采用machine leanring notation*

### 思路
对于一个图片分布$p(\mathbf{x})$，$p(\mathbf{x})$越大则意味着越接近一个真实图片，当我们采样$\mathbf{x}\sim p(\mathbf{x})$时候，得到的$\mathbf{x}$大概率会有比较高的$p(\mathbf{x})$值（即$\mathbf{x}$为真实图片）。但是假如我们采样$\mathbf{x}\sim \mathcal{N}(0,\mathbf{I})$，$p(\mathbf{x})$高的概率就很低（即$\mathbf{x}$不太可能是真实图片）。

所以diffusion的思路是，能不能从任意高斯采样$x\sim \mathcal{N}(0,1)$，通过迭代的方式求得一个局部极值（即一个真实图片）：
$$
\begin{equation}
\mathbf{x}_{max}=\argmax_x p(\mathbf{x})
\end{equation}
$$
迭代的方式采用的是langevin dynamic：
$$
\begin{equation}
\mathbf{x}_{i+1}\gets \mathbf{x}_{i} + \delta \nabla_{\mathbf{x}} \log p(\mathbf{x})+\sqrt{2\delta} \mathbf{z}_i, \quad i=0,1,...,K
\end{equation}
$$
其中$\mathbf{z}_i\sim \mathcal{N}(0, \mathbf{I})$，当$K\to\inf,\delta\to 0$的时候，我们能找到这样的一个极值。


### 问题定义及目标函数

A dataset $\mathbf{X}=\{\mathbf{x}_1, \mathbf{x}_2, ..., \mathbf{x}_N\}$ is independetly sampled from a distribution $p(x)$.

任意分布都可以用神经网络$f_\theta$表示成如下形式
$$
\begin{equation}
    p_\theta(\mathbf{x})=\frac{e^{-f_\theta(\mathbf{x})}}{Z_\theta}
\end{equation}
$$
其中$Z_\theta=\int e^{-f_\theta(\mathbf{x})} \text{d}\mathbf{x}$为正则项，目的是保证$\int p_\theta(\mathbf{x})d\mathbf{x}=1$，为intractable function。

优化过程还是根据最大似然估计
$$
\begin{equation}
    \max_\theta \sum_{i=1}^N \log{p_\theta(\mathbf{x}_i)}
\end{equation}
$$
由于$Z_\theta$是intractable function，导致式(2)也是intractable。但是我们可以通过如下方式消除$Z_\theta$
$$
\begin{equation}
    s_\theta(\mathbf{x})=\nabla_{\mathbf{x}}\log p_\theta(\mathbf{x})=-\nabla_{\mathbf{x}} f_\theta(\mathbf{x}) - \nabla_{\mathbf{x}} \log Z_\theta=-\nabla_{\mathbf{x}} f_\theta(\mathbf{x}).
\end{equation}
$$
那么问题就转化成了如何优化下式
$$
\begin{equation}
    \mathbb{E}_{p(\mathbf{x})}\left[ \lVert \nabla_{\mathbf{x}} \log{p(\mathbf{x})} - s_\theta(\mathbf{x}) \rVert_2^2 \right]
\end{equation}
$$

### 求解方向$\nabla_{\mathbf{x}} \log p(\mathbf{x})$

根据tweedie's formular，我们有
$$
\begin{equation}
\sqrt{\bar{\alpha}_t}x_0 = x_t+(1-\bar{\alpha}_t)\nabla_{x_t}\log p(x_t)
\end{equation}
$$
又因为$x_t\sim \mathcal{N}(\sqrt{\bar{\alpha}_t}x_0, \sqrt{1-\bar{\alpha}}_t \mathbf{I})$等价于：
$$
\begin{equation}
x_t=\sqrt{\bar{\alpha}_t} x_0+\sqrt{1-\bar{\alpha}_t}\epsilon_t
\end{equation}
$$
结合上两式子可得：
$$
\begin{equation}
\nabla_{x_t}\log p(x_t)=-\frac{\epsilon_t}{\sqrt{1-\bar{\alpha}_t}}
\end{equation}
$$
即$\nabla_{x_t}\log p(x_t)$是前向过程加的噪声乘以一个常数。

<img src="https://yang-song.net/assets/img/score/langevin.gif" width="250">


### 解决$\nabla_{\mathbf{x}} \log p(\mathbf{x})$在稀疏区域不准确的问题


<img src="https://yang-song.net/assets/img/score/smld.jpg" width="500">
<img src="https://yang-song.net/assets/img/score/pitfalls.jpg" width="500">

解决方案是给当前数据$p(\mathbf{y})$加高斯噪声$\mathcal{N}(\mathbf{y}, \sigma_i^2)$，以此起到类似数据增强的作用。随着$\sigma_i$的增大，数据会离原先的分布越遥远。理想情况下，我们希望不管在离原先分布多远的地方都有增强的数据点来提供$\nabla_{\mathbf{y}} \log p(\mathbf{y})$。所以diffusion采用了分层加噪的手段，并且对于每一层加噪数据$p_{\sigma_i}(\mathbf{x})$，我们都希望求得其$\nabla_{\mathbf{x}} \log p_{\sigma_i}(\mathbf{x})$。

因为$\mathbf{x}|\mathbf{y}\sim \mathcal{N}(\mathbf{y}, \sigma_i^2)$，所以边际分布$p_{\sigma_i}(\mathbf{x})$为：
$$
\begin{equation}
    p_{\sigma_i}(\mathbf{x})=\int p(\mathbf{y})p(\mathbf{x}|\mathbf{y}) \text{d}\mathbf{y}
\end{equation}
$$
每个边际分布的采样都可以这样获得：$\mathbf{x}+\sigma_i \mathbf{z}$。

**优化目标**：我们用$s_\theta(\mathbf{x}, i)$拟合每个分布的$\nabla_{\mathbf{x}} \log p_{\sigma_i}(\mathbf{x})$。
$$
\begin{equation}
\mathbb{E}\left[\lVert \nabla_{\mathbf{x}}\log p_{\sigma_i}(\mathbf{x})-s_\theta(\mathbf{x},i)\rVert _2^2\right]
\end{equation}
$$
当完成以上优化目标以后，使用langevin dynamics作梯度下降：
$$
\begin{equation}
\mathbf{x}_{i+1}\gets \mathbf{x}_{i} + \delta s_\theta(\mathbf{x},i)+\sqrt{2\delta} \mathbf{z}_i, \quad i=0,1,...,K
\end{equation}
$$


# Diffusion (with stochastic differential equations)

*本文采用machine leanring notation*

### 思路

SDE的diffusion可以认为是把加噪行为定义在连续空间上，而不是像上文一样离散地加噪：
$$
\begin{equation}
\begin{split}
    \text{discrete: }&\quad\mathbf{x}_{t+1}\gets \mathbf{x}_{t} + \delta \nabla_{\mathbf{x}} \log p(\mathbf{x}_t)+\sqrt{2\delta} \mathbf{z}_i, \quad i=0,1,...,K\\
    \text{continuous:}&\quad\text{d}\mathbf{x}=\mathbf{f}(\mathbf{x}, t)\text{d}t+g(t)\text{d}\mathbf{w},\quad t\in [0, T]
\end{split}
\end{equation}
$$
其中$\mathbf{w}$是随机vector，$\text{d}\mathbf{w}$可以理解为无限小的高斯噪声，$g(t)$为diffusion coefficient，$\mathbf{f}(\mathbf{x}, t)$为drift coefficient。我们标记$p_t(\mathbf{x})=\mathbf{x}(t)$为$t$时刻的数据分布的话，显然$p_0(\mathbf{x})$是原始分布，$p_T(\mathbf{x})=\mathcal{N}(0,1)$遵循高斯分布当$T$足够大时。