Elucidating the Design Space of Diffusion-Based Generative Models. NeurIPS 2022.

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

# Code怎么写的

先看```EDMLoss```
```
rnd_normal = torch.randn([images.shape[0], 1, 1, 1], device=images.device)
sigma = (rnd_normal * self.P_std + self.P_mean).exp()
weight = (sigma ** 2 + self.sigma_data ** 2) / (sigma * self.sigma_data) ** 2
y, augment_labels = augment_pipe(images) if augment_pipe is not None else (images, None)
n = torch.randn_like(y) * sigma
D_yn = net(y + n, sigma, labels, augment_labels=augment_labels)
loss = weight * ((D_yn - y) ** 2)
```
$$
\begin{aligned}
\sigma&=\exp(P_{std}\epsilon+P_{mean})\\
W&=\frac{\sigma^2+\sigma_{data}^2}{\sigma^2\sigma_{data}^2}\\
\mathbf{n}&=\sigma\mathbf{\epsilon}\\
\mathcal{L}&=W\left[\lVert D_\theta(\mathbf{y}+\mathbf{n})-\mathbf{y}\rVert _2^2\right]
\end{aligned}
$$
```EDMPrecond```
```
c_skip = self.sigma_data ** 2 / (sigma ** 2 + self.sigma_data ** 2)
c_out = sigma * self.sigma_data / (sigma ** 2 + self.sigma_data ** 2).sqrt()
c_in = 1 / (self.sigma_data ** 2 + sigma ** 2).sqrt()
c_noise = sigma.log() / 4

F_x = self.model((c_in * x).to(dtype), c_noise.flatten(), class_labels=class_labels, **model_kwargs)
D_x = c_skip * x + c_out * F_x.to(torch.float32)
```
$$
\begin{aligned}
c_{skip}&=\frac{\sigma_{data}^2}{\sigma^2+\sigma_{data}^2}\\
c_{out}&=\frac{\sigma\sigma_{data}}{\sqrt{\sigma^2+\sigma_{data}^2}}\\
c_{in}&=\frac{1}{\sqrt{\sigma_{data}^2+\sigma^2}}\\
c_{noise}&=\frac{\log{\sigma}}{4}\\
D_\theta(\mathbf{x})&=c_{skip}\mathbf{x}+c_{out}F_\theta(c_{in}\mathbf{x},c_{noise})
\end{aligned}
$$

采样```edm_sampler```
```
t_steps = (sigma_max ** (1 / rho) + step_indices / (num_steps - 1) * (sigma_min ** (1 / rho) - sigma_max ** (1 / rho))) ** rho
```
$$
t_i=\left(\sigma_{max}^{\frac{1}{\rho}}+\frac{t}{T}(\sigma_{min}^{\frac{1}{\rho}}-\sigma_{max}^{\frac{1}{\rho}})\right)^{\rho}
$$
```
gamma = min(S_churn / num_steps, np.sqrt(2) - 1)
t_hat = net.round_sigma(t_cur + gamma * t_cur)
x_hat = x_cur + (t_hat ** 2 - t_cur ** 2).sqrt() * S_noise * randn_like(x_cur)
denoised = net(x_hat, t_hat, class_labels).to(torch.float64)
d_cur = (x_hat - denoised) / t_hat
x_next = x_hat + (t_next - t_hat) * d_cur
if i < num_steps - 1:
    denoised = net(x_next, t_next, class_labels).to(torch.float64)
    d_prime = (x_next - denoised) / t_next
    x_next = x_hat + (t_next - t_hat) * (0.5 * d_cur + 0.5 * d_prime)
```
$$
\begin{aligned}
\gamma&=\min(\sqrt{2}-1,\frac{S_{churn}}{T})\\
\hat{t}_i&=t_i+\gamma t_i\\
\hat{x}_i&=x_i+\sqrt{\hat{t}_i^2-t_i^2}S_{noise}\epsilon\\
d_i&=\frac{\hat{x}_i-D_\theta(\hat{x}_i,\hat{t}_i)}{\hat{t}_i}\\
x_{i+1}&=\hat{x}+(t_{i+1}-\hat{t})*d_i\\
d'_i&=\frac{x_{i+1}-D_\theta(x_{i+1},t_{i+1})}{t_{i+1}}\\
x_{i+1}&=\hat{x}_i+(t_{i+1}-\hat{t}_i)*\frac{d_i+d'_i}{2}
\end{aligned}
$$