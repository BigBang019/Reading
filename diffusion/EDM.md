# EDM Takeaway

回忆以前的ODE框架如下：
$$
\begin{equation}
\text{d}\mathbf{x}=\left[\mathbf{f}(\mathbf{x},t)-\frac{1}{2}g^2(t)\nabla_\mathbf{x}\log p_t(\mathbf{x})\right]\text{d}t
\end{equation}
$$

在EDM的框架中，希望只用scaling factor $s_t$和噪声$\sigma_t$表示加噪，而不再用t，因此EDM希望用$\sigma,s_t$的表达式替换掉$\mathbf{f}(\mathbf{x},t),g(t),\nabla_\mathbf{x}\log p_t(\mathbf{x})$。

并且要满足下面的marginal distribution 
$$
\begin{equation}
p(\mathbf{x}_t|\mathbf{x}_0)=\mathcal{N}(s_t\mathbf{x}_0, s_t^2\sigma_t^2\mathbf{I})
\end{equation}
$$
即$\mathbf{x}_t=s_t\mathbf{x}_0+s_t\sigma_t\epsilon$。

为了满足这个要求，必须要：
$$
\begin{equation}
\begin{aligned}
\mathbf{f}(\mathbf{x},t)&=\frac{\dot{s_t}}{s_t}\mathbf{x}\\
g(t)&=s_t\sqrt{2\dot{\sigma_t}\sigma_t}\\
\nabla_\mathbf{x}\log p_t(\mathbf{x})&=\nabla_\mathbf{x}\log p(\frac{\mathbf{x}}{s_t};\sigma_t)
\end{aligned}
\end{equation}
$$
此时的ODE为：
$$
\begin{equation}
\text{d}\mathbf{x}=\left[\frac{\dot{s_t}}{s_t}\mathbf{x}-s^2_t\dot{\sigma_t}\sigma_t\nabla_\mathbf{x}\log p(\frac{\mathbf{x}}{s_t};\sigma_t)\right]\text{d}t
\end{equation}
$$

优化目标：
$$
\begin{equation}
\mathbb{E}_{y\sim p_{data},n\sim\mathcal{N}(0,\sigma^2\mathbf{I})}\left[\lambda_\sigma\lVert D_\theta(y+n;\sigma)-y\rVert_2^2\right]
\end{equation}
$$
并且训练好的$D_\theta$满足
$$
\begin{equation}
\begin{aligned}
\nabla_\mathbf{x}\log p(\mathbf{x};\sigma_t)&=\frac{1}{\sigma_t^2}\left[D_\theta(\mathbf{x};\sigma_t)-\mathbf{x}\right]\\
\nabla_\mathbf{x}\log p(\frac{\mathbf{x}}{s_t}; \sigma_t)&=\frac{1}{s_t\sigma_t^2}\left[D_\theta(\frac{\mathbf{x}}{s_t};\sigma_t)-\frac{\mathbf{x}}{s_t}\right]
\end{aligned}
\end{equation}
$$

那么此时的ODE为：
$$
\begin{equation}
\text{d}\mathbf{x}=\left[(\frac{\dot{\sigma_t}}{\sigma_t}+\frac{\dot{s_t}}{s_t})\mathbf{x}-\frac{\dot{\sigma_t}s_t}{\sigma_t}D_\theta(\frac{\mathbf{x}}{s_t};\sigma_t)\right]\text{d}t
\end{equation}
$$

# QA

### $\sigma(t)$和$s(t)$选什么最好
当$\sigma(t)=t,s(t)=1$时最好，此时ODE为：$\text{d}\mathbf{x}/\text{d}t=(\mathbf{x}-D(\mathbf{x};t))/t$