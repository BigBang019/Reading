# Diffusion (Score-based SDE Perspective)
A dataset $X=\{x_1, x_2, ..., x_N\}$ is independetly sampled from a distribution $p(x)$.

任意分布都可以用神经网络$f_\theta$表示成如下形式
$$
\begin{equation}
    p_\theta(x)=\frac{e^{-f_\theta(x)}}{Z_\theta}
\end{equation}
$$
其中$Z_\theta=\int e^{-f_\theta(x)} dx$为正则项，目的是保证$\int p_\theta(x)dx=1$，为intractable function。

优化过程还是根据最大似然估计
$$
\begin{equation}
    \max_\theta \sum_{i=1}^N \log{p_\theta(x_i)}
\end{equation}
$$
由于$Z_\theta$是intractable function，导致式(2)也是intractable。但是我们可以通过如下方式消除$Z_\theta$
$$
\begin{equation}
    s_\theta(x)=\nabla_x\log p_\theta(x)=-\nabla_x f_\theta(x) - \nabla_x \log Z_\theta=-\nabla_x f_\theta(x).
\end{equation}
$$
那么问题就转化成了如何优化下式
$$
\begin{equation}
    \mathbb{E}_{p(x)}\left[ \left|\right| \nabla_x \log{p(x)} - s_\theta(x)\left|\right|_2^2 \right]
\end{equation}
$$
上式的新问题是
> Directly computing this $\nabla_x\log{p(x)}$ is infeasible.
>
> 但是有解决方案：score-matching



