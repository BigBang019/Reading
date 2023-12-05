*本文采用statistical notation*
# Tweedie estimator

经验贝叶斯的一种，Tweedie estimator希望能在不假设先验分布$p_\Theta(\theta)$的情况下求解$\hat{\theta}(\mathbf{X})$。

目标仍然是：
$$
\begin{equation}
\hat{\theta}(\mathbf{X})=\argmin_{\hat{\theta}}\mathbf{E}_{\Theta,X}\left[(\hat{\theta}(\mathbf{X})-\Theta)^2\right]=\mathbb{E}_{\Theta|\mathbf{X}}\left[\Theta|\mathbf{X}=\mathbf{x}\right]
\end{equation}
$$


### 背景
先验$p_\Theta(\theta)$未知，他其中的$N$个采样$\{\theta_1, \theta_2, ..., \theta_N\}$也未知，但是我们知道每个采样都作为随机种子生成了一个随机数$x_i\sim \mathcal{N}(\theta_i, \sigma_0^2)$。也就是说$x_i|\theta_i\sim \mathcal{N}(\theta_i, \sigma_0^2)$。

如果采用MLE的思路那么容易得出$\theta_i=\argmax_{\theta} p_{\mathbf{X}|\Theta}(\mathbf{x}|\theta)=x_i$。

$?$那么在这种情况下能否得出$\hat{\theta}(\mathbf{X})=\argmin_{\hat{\theta}}\mathbb{E}\left[(\hat{\theta}(\mathbf{X})-\Theta)^2\right]=\argmin_{\hat{\theta}}\mathbb{E}\left[(\hat{\theta}(\mathbf{X})-\mathbb{E}_\Theta[\Theta])^2\right]$，即$\hat{\theta}(\mathbf{X})=\mathbb{E}_\Theta[\Theta]=\frac{1}{N}\sum_i^N \theta_i=\frac{1}{N}x_i$。

### Tweedie estimator
我们不假设先验分布$p_\Theta(\theta)$，后验分布$p_{X|\Theta}(x_i|\theta_i)= \mathcal{N}(\theta_i, \sigma_0)$遵循高斯分布。那么后验分布的期望为：

$$
\begin{equation}
\begin{split}
\mathbb{E}_{\Theta|X}\left[\Theta|X=x_i\right] &=\int \theta_i p_{\Theta|X}(\theta_i|x_i) \text{d}\theta_i\\
&=\int \theta_i\frac{p_{X|\Theta}(x_i|\theta_i)p_\Theta(\theta_i)}{p_{X}(x_i)}\text{d}\theta_i\\
&=\frac{\int \theta_i p_{X|\Theta}(x_i|\theta_i)p_\Theta(\theta_i) \text{d}\theta_i}{p_{X}(x_i)}\\
&=...\\
&=x_i+\sigma_0^2 \frac{\text{d}}{\text{d}x} \log p_X(x)
\end{split}
\end{equation}
$$

后验分布期望$\mathbb{E}_{\Theta|X}[\Theta|X]$与先验分布$p_\Theta(\theta)$无关。

### 用处

如果$z$取自一个未知的高斯采样$z\sim \mathcal{N}(\mu_z, \sigma_0^2)$，我们有估计$\hat{\mu}_z=z+\sigma_0^2\nabla_z \log p_Z(z)$。



## Reference
[Tweedie estimator](https://zhuanlan.zhihu.com/p/594007789)

[Stanford的slides](https://efron.ckirby.su.domains/talks/2010TweediesFormula.pdf)