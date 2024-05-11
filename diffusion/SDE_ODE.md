SCORE-BASED GENERATIVE MODELING THROUGH STOCHASTIC DIFFERENTIAL EQUATIONS. ICLR 2021.

# SDE/ODE Takeaway

SDE表达如下
$$
\begin{equation}
\text{d}\mathbf{x}=\mathbf{f}(\mathbf{x}, t)\text{d}t+g(t)\text{d}\mathbf{w},\quad t\in [0, T]
\end{equation}
$$
其中$\mathbf{w}$是随机vector，$\text{d}\mathbf{w}$可以理解为无限小的高斯噪声，$g(t)$为diffusion coefficient，$\mathbf{f}(\mathbf{x}, t)$为drift coefficient。我们标记$p_t(\mathbf{x})$为$\mathbf{x}(t)$为marginal distribution，显然$p_0(\mathbf{x})$是原始分布，$p_T(\mathbf{x})=\mathcal{N}(0,1)$为高斯分布当$T$足够大时。

任何SDE都有以下反向过程：
$$
\begin{equation}
\text{d}\mathbf{x}=\left[\mathbf{f}(\mathbf{x},t)-g^2(t)\nabla_\mathbf{x}\log p_t(\mathbf{x})\right]\text{d}t+g(t)\text{d}\mathbf{w}
\end{equation}
$$
需要训练$s_\theta(\mathbf{x},t)$拟合$\nabla_\mathbf{x}\log p_t(\mathbf{x})$：
$$
\mathbb{E}\left[\lVert\nabla_\mathbf{x}\log p_t(\mathbf{x})-s_\theta(\mathbf{x},t)\rVert_2^2\right]
$$
SDE反向：
$$
\begin{equation}
\Delta\mathbf{x}\leftarrow\left[\mathbf{f}(\mathbf{x},t)-g^2(t)s_\theta(\mathbf{x},t)\right]\Delta t + g(t)\sqrt{|\Delta t|} \mathbf{z}_t, \quad \mathbf{z}_t\sim \mathcal{N}(0, \mathbf{I})
\end{equation}
$$

下式ODE与Eq.(1)拥有相同的marginal distribution $p_t(\mathbf{x})$：
$$
\begin{equation}
\text{d}\mathbf{x}=\left[\mathbf{f}(\mathbf{x},t)-\frac{1}{2}g^2(t)\nabla_\mathbf{x}\log p_t(\mathbf{x})\right]\text{d}t
\end{equation}
$$
当使用$s_\theta(\mathbf{x},t)$替换$\nabla_\mathbf{x}\log p_t(\mathbf{x})$以后，Eq.(4)为neural ODE，可以使用neural ODE solver求解

# QA

### SDE框架下的理解生成问题

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

方向$\nabla_{\mathbf{x}} \log p(\mathbf{x})$不可求解，但是上式子还是可以通过score matching的方式进行优化


### SDE框架下理解前向过程

<img src="https://yang-song.net/assets/img/score/smld.jpg" width="500">
<img src="https://yang-song.net/assets/img/score/pitfalls.jpg" width="500">

Score function $\nabla_{\mathbf{x}} \log p(\mathbf{x})$在$p_{data}(x)$稀疏区域不准确。

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

### SMLD和DDPM在SDE框架下的统一
DDPM的前向过程为：
$$
\begin{equation}
\mathbf{x}_t = \sqrt{1-\beta_t}\mathbf{x}_{t-1} + \sqrt{\beta_t}\mathbf{\epsilon}_t, \quad \mathbf{\epsilon}_t \sim \mathcal{N}(0, \mathbf{I})
\end{equation}
$$
上式可以被写成一个SDE：
$$
\begin{equation}
\text{d}\mathbf{x}=-\frac{\beta(t)}{2}\mathbf{x}\ \text{d}t+\sqrt{\beta(t)}\text{d}\mathbf{w},\quad \mathbf{f}(\mathbf{x},t)=-\frac{\beta(t)}{2}\mathbf{x},\quad g(t)=\sqrt{\beta(t)}
\end{equation}
$$
离散化后的逆向过程为：
$$
\begin{equation}
\mathbf{x}_{i-1}=\frac{1}{\sqrt{1-\beta_i}}\left[\mathbf{x}_i+\frac{\beta_i}{2}\nabla_\mathbf{x}\log p_i(\mathbf{x}_i)\right]+\sqrt{\beta_i}\mathbf{z}_i, \quad \mathbf{z}_i\sim \mathcal{N}(0, \mathbf{I})
\end{equation}
$$

SMLD的前向过程：
$$
\begin{equation}
\mathbf{x}_i=\mathbf{x}_{i-1}+\sqrt{\sigma_i^2-\sigma^2_{i-1}}\mathbf{z}_{i}
\end{equation}
$$
因此其前向过程的SDE：
$$
\begin{equation}
\text{d}\mathbf{x}=\sqrt{\frac{\text{d}\left[\sigma^2(t)\right]}{\text{d}t}}\text{d}\mathbf{w},\quad \mathbf{f}(\mathbf{x},t)=0,\quad g(t)=\sqrt{\frac{\text{d}\left[\sigma^2(t)\right]}{\text{d}t}}
\end{equation}
$$
离散化后的逆向过程为：
$$
\mathbf{x}_{i-1}=\mathbf{x}_i+(\sigma_i^2-\sigma_{i-1}^2)\nabla_\mathbf{x}\log p_i(\mathbf{x})+\sqrt{(\sigma_i^2-\sigma_{i-1}^2)}\mathbf{z}_i, \quad \mathbf{z}_i\sim \mathcal{N}(0,\mathbf{I})
$$

### 与DDPM/DDIM中$\epsilon$的关系
DDPM/DDIM的前向为$q(x_t|x_0)$

从式(5)中可以看到$x_t\sim \mathcal{N}(\sqrt{\bar{\alpha}_t}x_0, \sqrt{1-\bar{\alpha}}_t \mathbf{I})$。根据tweedie's formular，我们有
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
所以学习$\epsilon_t$等效于学习score function。

