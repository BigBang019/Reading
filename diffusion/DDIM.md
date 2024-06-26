# DDIM Takeaway

*本文采用machine leanring notation*

DDIM定义的forward和reverse都不是Markov过程，DDIM先定义了以下inference distribution：
$$
\begin{equation}
\begin{gathered}
q_\sigma(\mathbf{x}_{1:T}|\mathbf{x}_0):=q_\sigma(\mathbf{x}_T|x_0)\prod_{t=2}^T q_\sigma(\mathbf{x}_{t-1}|\mathbf{x}_t, \mathbf{x}_0)\\
q_\sigma(\mathbf{x}_T|\mathbf{x}_0):=\mathcal{N}(\sqrt{\bar{\alpha}_t}\mathbf{x}_0, (1-\bar{\alpha}_T)\mathbf{I})\\
\forall t>1, \quad q_\sigma(\mathbf{x}_{t-1}|\mathbf{x}_t,\mathbf{x}_0):=\mathcal{N}\left(\sqrt{\bar{\alpha}_{t-1}}\mathbf{x}_0+\sqrt{1-\bar{\alpha}_{t-1}-\sigma_t^2}\cdot \frac{\mathbf{x}_t-\sqrt{\bar{\alpha}_t}\mathbf{x}_0}{\sqrt{1-\bar{\alpha}_t}}, \sigma_t^2\mathbf{I}\right)
\end{gathered}
\end{equation}
$$
注意$q_\sigma(\mathbf{x}_{t-1}|\mathbf{x}_t, \mathbf{x}_0)$并不是反向过程，因为$\mathbf{x}_0$在条件位表示已知量，真正的reverse过程是不知道$\mathbf{x}_0$。

以上定义满足边际分布（marginal distribution）：
$$
\begin{equation}
q_\sigma(\mathbf{x}_t|\mathbf{x}_0)=\mathcal{N}(\sqrt{\bar{\alpha}_t}\mathbf{x}_0, (1-\bar{\alpha}_t)\mathbf{I})
\end{equation}
$$
基于以上定义，forward过程可以通过bayes' rule推导：
$$
\begin{equation}
q_\sigma(\mathbf{x}_t|\mathbf{x}_{t-1},\mathbf{x}_0)=\frac{q_\sigma(\mathbf{x}_{t-1}|\mathbf{x}_t,\mathbf{x}_0)q_\sigma(\mathbf{x}_t|\mathbf{x}_0)}{q_\sigma(\mathbf{x}_{t-1}|\mathbf{x}_0)}
\end{equation}
$$
如果已知$\mathbf{x}_0$，那么reverse过程$p_\theta(\mathbf{x}_{t-1}|\mathbf{x}_t)$可以归约成为$q(\mathbf{x}_{t-1}|\mathbf{x}_t, \mathbf{x}_0)$。DDIM用了猜一个$\mathbf{x}_0$的方式来作归约：
$$
\begin{equation}
\mathbf{x}_0\approx f_\theta(\mathbf{x}_t):=\frac{\mathbf{x}_t-\sqrt{1-\bar{\alpha}_t}\cdot \epsilon_\theta(\mathbf{x}_t,t)}{\sqrt{\bar{\alpha}_t}}
\end{equation}
$$
那么
$$
\begin{equation}
    p_\theta(\mathbf{x}_{t-1}|\mathbf{x}_t)=
    \left \{
    \begin{aligned}
    &\mathcal{N}(f_\theta(\mathbf{x}_1), \sigma_1^2\mathbf{I}) \quad \text{if}\ t=1\\
    &q_\sigma(\mathbf{x}_{t-1}|\mathbf{x}_t,f_\theta(\mathbf{x}_t))\quad \text{otherwise}
    \end{aligned}
    \right.
\end{equation}
$$

采样过程：
$$
\begin{equation}
\mathbf{x}_{t-1}=\sqrt{\bar{\alpha}_{t-1}}\left(\frac{\mathbf{x}_t-\sqrt{1-\bar{\alpha}_t}\epsilon_\theta(\mathbf{x}_t)}{\sqrt{\bar{\alpha}_t}}\right)+\sqrt{1-\bar{\alpha}_{t-1}-\sigma_t^2}\cdot \epsilon_\theta(\mathbf{x}_t)+\sigma_t\epsilon_t
\end{equation}
$$

优化目标：（暂略）


# QA
### 与DDPM的关系
当$\sigma_t=0$的时候，整个reverse过程为deterministic的；当$\sigma_t=\sqrt{\frac{(1-\bar{\alpha}_{t-1})(1-\bar{\alpha}_t)}{\bar{\alpha}_{t-1}(1-\bar{\alpha}_t)}}$的时候，整个forward和reverse都变为Markovian，此时DDIM变为DDPM。

### DDIM Inversion是什么
回忆DDIM reverse：$\mathbf{x}_{t-1}=\sqrt{\bar{\alpha}_{t-1}}\left(\frac{\mathbf{x}_t-\sqrt{1-\bar{\alpha}_t}\epsilon_\theta(\mathbf{x}_t)}{\sqrt{\bar{\alpha}_t}}\right)+\sqrt{1-\bar{\alpha}_{t-1}-\sigma_t^2}\cdot \epsilon_\theta(\mathbf{x}_t)+\sigma_t\epsilon_t$

考虑以下场景：用户A采样$\mathbf{x}_T\sim \mathcal{N}(\mathbf{0},\mathbf{I})$，并且用DDIM reverse得到$\mathbf{x}_0$。那么给定$\mathbf{x}_0$，能不能推回当初采样的$\mathbf{x}_T$？

我们重写一下DDIM reverse的公式：
$$
\begin{equation}
\mathbf{x}_t=\sqrt{\alpha_t}\left[\mathbf{x}_{t-1}-\sqrt{1-\bar{\alpha}_{t-1}}\epsilon_\theta(\mathbf{x}_t,t)\right]+\sqrt{1-\bar{\alpha}_t}\epsilon_\theta(\mathbf{x}_t, t)
\end{equation}
$$
上面就是DDIM inverse的递推式，但是公式右边还是有关于$\mathbf{x}_{t}$的函数$\epsilon_\theta(\mathbf{x}_t,t)$，在实际应用中，DDIM inverse用$\epsilon_\theta(\mathbf{x}_{t-1},t)$替换了$\epsilon_\theta(\mathbf{x}_t,t)$。我们会在后续code的分析里验证这一点。

# Code怎么写的

### DDIM Forward

```diffusers.DDPMScheduler.add_noise```
```
sqrt_alpha_prod = alphas_cumprod[timesteps] ** 0.5
sqrt_one_minus_alpha_prod = (1 - alphas_cumprod[timesteps]) ** 0.5
noisy_samples = sqrt_alpha_prod * original_samples + sqrt_one_minus_alpha_prod * noise
```
$$
\sqrt{\bar{\alpha}_t}x_0+\sqrt{1-\bar{\alpha}_t}\epsilon
$$

### DDIM Reverse

```diffuser.DDIMScheduler.step```
```
prev_timestep = timestep - self.config.num_train_timesteps // self.num_inference_steps
```
$$
\tau_{i-1},\tau_{i}
$$

```
pred_original_sample = (sample - beta_prod_t ** (0.5) * model_output) / alpha_prod_t ** (0.5)
```
$$
\frac{x_{\tau_i}-\sqrt{1-\bar{\alpha}_{\tau_i}}\epsilon_\theta(x_{\tau_{i}},\tau_{i})}{\sqrt{\bar{\alpha}_{\tau_i}}}
$$
```
pred_sample_direction = (1 - alpha_prod_t_prev - std_dev_t**2) ** (0.5) * pred_epsilon
```
$$
\sqrt{1-\bar{\alpha}_{\tau_{i-1}}-\sigma_{\tau_i}^2}\epsilon_\theta(x_{\tau_i}, \tau_i)
$$
```
prev_sample = alpha_prod_t_prev ** (0.5) * pred_original_sample + pred_sample_direction
```
$$
\sqrt{\bar{\alpha}_{\tau_{i-1}}}\left(\frac{x_{\tau_i}-\sqrt{1-\bar{\alpha}_{\tau_i}}\epsilon_\theta(x_{\tau_{i}},\tau_{i})}{\sqrt{\bar{\alpha}_{\tau_i}}}\right)+\sqrt{1-\bar{\alpha}_{\tau_{i-1}}-\sigma_{\tau_i}^2}\epsilon_\theta(x_{\tau_i}, \tau_i)
$$

### DDIM Inverse
```diffuser.DDIMInverseScheduler.step```


```
prev_timestep = timestep
timestep = min(
    timestep - self.config.num_train_timesteps // self.num_inference_steps, self.config.num_train_timesteps - 1
)
```
注意这里的$(\text{prev\_timestep}, \text{timestep})=(\tau_{i},\tau_{i-1})$和DDIM reverse的$(\text{prev\_timestep}, \text{timestep})=(\tau_{i-1},\tau_{i})$不同。
```
pred_original_sample = (sample - beta_prod_t ** (0.5) * model_output) / alpha_prod_t ** (0.5)
```
$$
\frac{\mathbf{x}_{\tau_{i-1}}-\sqrt{1-\bar{\alpha}_{\tau_{i-1}}}\epsilon_\theta(\mathbf{x}_{\tau_{i-1}},\tau_i)}{\sqrt{\bar{\alpha}_{\tau_{t-1}}}}
$$
```
pred_sample_direction = (1 - alpha_prod_t_prev) ** (0.5) * pred_epsilon
```
$$
\sqrt{1-\bar{\alpha}_{\tau_i}}\epsilon_\theta(\mathbf{x}_{\tau_{i-1}},\tau_{i})
$$
```
prev_sample = alpha_prod_t_prev ** (0.5) * pred_original_sample + pred_sample_direction
```
$$
\sqrt{\bar{\alpha}_{\tau_i}}\left(\frac{\mathbf{x}_{\tau_{i-1}}-\sqrt{1-\bar{\alpha}_{\tau_{i-1}}}\epsilon_\theta(\mathbf{x}_{\tau_{i-1}},\tau_i)}{\sqrt{\bar{\alpha}_{\tau_{i-1}}}}\right)+\sqrt{1-\bar{\alpha}_{\tau_{i-1}}}\epsilon_\theta(\mathbf{x}_{\tau_{i-1}},\tau_i)
$$
当$(\tau_{i},\tau_{i-1})=(t,t-1)$时，上式等效于Eq. (7)；当不满足时，后果还未探索。