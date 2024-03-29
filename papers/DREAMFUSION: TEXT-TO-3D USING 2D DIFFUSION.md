# DREAMFUSION: TEXT-TO-3D USING 2D DIFFUSION
*本文采用machine learning notation*

### Notation Lookup
*解释文章中的notation*

$\mathbf{x}\sim q(\mathbf{x})$为原始数据分布，$\mathbf{z}_t$为加噪数据，并且有$\mathbf{z_t}|\mathbf{x}\sim \mathcal{N}(\alpha_t\mathbf{x}, \sigma_t^2\mathbf{I})$。

$q(\mathbf{z}_t|\mathbf{x})$为前向传播过程，$p_\phi(\mathbf{z}_{t-1}|\mathbf{z}_{t})$为反向传播过程（denoising），$\hat{\mathbf{x}}_\phi(\mathbf{z}_t)$为denoiser的拟合。

文章中用$s_\phi(\mathbf{z}_t)$拟合$\nabla_{\mathbf{z}_t}\log p(\mathbf{z}_t)$，并且标记$\epsilon_\phi(\mathbf{z}_t)=-\sigma_t s_\phi(\mathbf{z}_t)$。

结合tweedie's formula，他们的关系可以这么写（详见diffusion.md解法2）
$$
\begin{equation}
\begin{split}
    \alpha_t\mathbf{x}=\mathbb{E}\left[\mathbf{x}|\mathbf{z}_t\right]&=\mathbf{z}_t+\sigma_t^2\nabla_{\mathbf{z}_t}\log p(\mathbf{z}_t)\\
    \approx\alpha_t \hat{\mathbf{x}}_\phi(\mathbf{z}_t)&= \mathbf{z}_t-\sigma_t\epsilon_\phi(\mathbf{z}_t)
\end{split}
\end{equation}
$$

本文中diffusion的优化目标写作：
$$
\begin{equation}
    \mathcal{L}_{diff}(\phi, \mathbf{x})=\mathbb{E}_{t\sim \mathcal{U}(0,1),\epsilon\sim \mathcal{N}(0, \mathbf{I})}\left[w(t)\lVert \epsilon_t(\alpha_t \mathbf{x}+\sigma_t\epsilon) - \epsilon\rVert _2^2 \right]
\end{equation}
$$

### Pipeline
<img src="../pasteImage/dreamfusion.png">

*先讲一下pipeline，讲清楚这篇文章到底在做什么，怎么做的，技术难点和核心贡献我们下一小节讨论。*

本文是给定text，生成NeRF表示的3D模型：
$$
\begin{equation}
\mathbf{x}=g(\theta)
\end{equation}
$$
文中采用了freeze LDM参数$\phi$，梯度下降$\theta$的手段：
$$
\begin{equation}
\theta^*=\argmin_{\theta} \mathcal{L}_{diff}(\phi, \mathbf{x}=g(\theta)| y)=\argmin_\theta \mathbb{E}_{t\sim \mathcal{U}(0,1),\epsilon\sim \mathcal{N}(0, \mathbf{I})}\left[w(t)\lVert \epsilon_t(\alpha_t \mathbf{x}+\sigma_t\epsilon|y) - \epsilon\rVert _2^2 \right]
\end{equation}
$$
所以我们的目标就是构建$\epsilon_t(\alpha_t \mathbf{x}+\sigma_t \epsilon|y)-\epsilon$：
- Step0：对于给定的viewport和光线，生成NeRF渲染的图片$\mathbf{x}=g(\theta)$。
- Step1：对$\mathbf{x}$加噪声获得$\mathbf{z}_t$，再给定带方向的文本"a DSLR photo of a peacock on a surf board (front view)"，获得$\hat{\mathbf{x}}_\phi(\mathbf{z}_t|y)$。根据式(1)推得$\hat{\epsilon}_\phi(\mathbf{z}_t|y)=(\mathbf{z}_t-\alpha_t\hat{\mathbf{x}}_\phi(\mathbf{z}_t|y))/\sigma_t$。
- Step2：计算$\hat{\epsilon}_\phi(\mathbf{z}_t|y)-\epsilon$，根据式(4)对$\theta$作梯度下降。

### 核心Contribution
作者发现根据上面的pipeline得到的$\theta^*$并不优秀，并通过计算对$\theta$梯度的方式展开分析：
$$
\begin{equation}
\begin{split}
\nabla_\theta \mathcal{L}_{diff}(\phi,\mathbf{x}=g(\theta)|y)&=\mathbb{E}_{t\sim \mathcal{U}(0,1),\epsilon\sim \mathcal{N}(0,\mathbf{I})}\left[C(t)\underbrace{\left(\hat{\epsilon}_\phi(\mathbf{z}_t|y)-\epsilon \right)}_{\text{Noise Residual}}\underbrace{\frac{\partial \hat{\epsilon}(\mathbf{z}_t|y)}{\partial \mathbf{z}_t}}_{\text{U-Net Jacobian}}\underbrace{\frac{\partial \mathbf{x}}{\partial \theta}}_{\text{Generator Jacobian}}\right]
\end{split}
\end{equation}
$$
作者发现$\text{U-Net Jacobian}$经常出现梯度消失的情况，因为这一项等同于一个分布的Hessian Matrix：
$$
\begin{equation}
\frac{\partial \hat{\epsilon}(\mathbf{z}_t|y)}{\partial \mathbf{z}_t}=-\sigma_t\nabla_{\mathbf{z}_t}^2\log p(\mathbf{z}_t|y)
\end{equation}
$$
解决方案是丢弃$\text{U-Net Jacobian}$，得到新的优化目标（$\text{Score Distillation Sampling}$）：
$$
\begin{equation}
\begin{split}
\nabla_\theta \mathcal{L}_{SDS}(\phi,\mathbf{x}=g(\theta)|y)&=\mathbb{E}_{t\sim \mathcal{U}(0,1),\epsilon\sim \mathcal{N}(0,\mathbf{I})}\left[C(t)\underbrace{\left(\hat{\epsilon}_\phi(\mathbf{z}_t|y)-\epsilon \right)}_{\text{Noise Residual}}\underbrace{\frac{\partial \mathbf{x}}{\partial \theta}}_{\text{Generator Jacobian}}\right]
\end{split}
\end{equation}
$$
作者发现SDS的结果非常好，是因为优化SDS等效于优化下式（证明详见附录A.4节。）：
$$
\begin{equation}
\nabla_\theta\mathcal{L}_{SDS}(\phi, \mathbf{x}=g(\theta))=\nabla_\theta\mathbb{E}_t\left[\sigma_t/\alpha_tw(t)\text{KL}\left(q(\mathbf{z}_t|g(\theta))\lVert p_\phi(\mathbf{z}_t|y) \right)\right]
\end{equation}
$$
其中$\text{KL}\left(q(\mathbf{z}_t|g(\theta))\lVert p_\phi(\mathbf{z}_t|y) \right)$的意义是优化$\theta$使得分布$q(\mathbf{z}_t|g(\theta))$和目标分布$p_\phi(\mathbf{z}_t|y)$一致，这个过程叫做$\text{probability
density distillation}$。

### 我们实际上可以总结出这篇文章究竟做了什么
这篇文章首次提出了在LDM上作梯度下降的方案，本质是在作分布蒸馏：将$p_\phi(\mathbf{z}_t|y)$的知识蒸馏到$q(\mathbf{z}_t|g(\theta))$中。每当我们产生类似的想法，都可以用类似的pipeline实现相同的结果。