LucidDreamer: Towards High-Fidelity Text-to-3D Generation via Interval Score Matching. CVPR 2023.

# 核心思路
首先作者观察到了以下现象。

The optimazation process of score distallition sampling (SDS) leads $\mathbf{x}=g_\theta(\mathbf{c})$ towards the averaged pseudo-ground-truth (i.e. $\hat{\mathbf{x}}_0^t$):

<!-- $$
\begin{equation}
\begin{aligned}
\nabla_\theta \mathcal{L}_{SDS}(\theta)&=\mathbb{E}_{t,\epsilon,\mathbf{c}}\left[w(t)\left(\epsilon_\phi(\mathbf{x}_t,t,y)-\epsilon\right)\frac{\partial \mathbf{x}}{\partial \theta}\right]\\
&=\mathbb{E}_{t,\epsilon,\mathbf{c}}\left[\frac{\sqrt{\bar{\alpha}_t}w(t)}{\sqrt{1-\bar{\alpha}_t}}\left(\frac{\sqrt{1-\bar{\alpha}_t}}{\sqrt{\bar{\alpha}}_t}(\epsilon_\phi(\mathbf{x}_t,t,y)-\epsilon)+\frac{\mathbf{x}_t-\mathbf{x}_t}{\sqrt{\bar{\alpha}_t}}\right)\frac{\partial \mathbf{x}}{\partial \theta}\right]\\
&=\mathbb{E}_{t,\epsilon,\mathbf{c}}\left[\frac{\sqrt{\bar{\alpha}_t}w(t)}{\sqrt{1-\bar{\alpha}_t}}\left(\frac{\mathbf{x}_t-\sqrt{1-\bar{\alpha}_t}\epsilon}{\sqrt{\bar{\alpha}_t}}-\frac{\mathbf{x}_t-\sqrt{1-\bar{\alpha}_t}\epsilon_{\phi}(\mathbf{x}_t,t,y)}{\sqrt{\bar{\alpha}_t}}\right)\right]\\
&=\mathbb{E}_{t,\epsilon,\mathbf{c}}\left[\frac{\sqrt{\bar{\alpha}_t}w(t)}{\sqrt{1-\bar{\alpha}_t}}(g_{\theta}(\mathbf{c})-\hat{\mathbf{x}}^t_0)\frac{\partial \mathbf{x}}{\partial \theta}\right]\\
\end{aligned}
\end{equation}
$$ -->
$$
\begin{equation}
\begin{aligned}
\mathcal{L}_{SDS}(\theta)&=\mathbb{E}_{t,\epsilon,\mathbf{c}}\left[w(t)\left\lVert\epsilon_\phi(\mathbf{x}_t,t,y)-\epsilon\right\rVert _2^2\right]\\
&=\mathbb{E}_{t,\epsilon,\mathbf{c}}\left[\frac{\sqrt{\bar{\alpha}_t}w(t)}{\sqrt{1-\bar{\alpha}_t}}\left\lVert\frac{\sqrt{1-\bar{\alpha}_t}}{\sqrt{\bar{\alpha}}_t}(\epsilon_\phi(\mathbf{x}_t,t,y)-\epsilon)+\frac{\mathbf{x}_t-\mathbf{x}_t}{\sqrt{\bar{\alpha}_t}}\right\rVert _2^2\right]\\
&=\mathbb{E}_{t,\epsilon,\mathbf{c}}\left[\frac{\sqrt{\bar{\alpha}_t}w(t)}{\sqrt{1-\bar{\alpha}_t}}\left\lVert\frac{\mathbf{x}_t-\sqrt{1-\bar{\alpha}_t}\epsilon}{\sqrt{\bar{\alpha}_t}}-\frac{\mathbf{x}_t-\sqrt{1-\bar{\alpha}_t}\epsilon_{\phi}(\mathbf{x}_t,t,y)}{\sqrt{\bar{\alpha}_t}}\right\rVert _2^2\right]\\
&=\mathbb{E}_{t,\epsilon,\mathbf{c}}\left[\frac{\sqrt{\bar{\alpha}_t}w(t)}{\sqrt{1-\bar{\alpha}_t}}\left\lVert g_{\theta}(\mathbf{c})-\hat{\mathbf{x}}^t_0\right\rVert _2^2\right]\\
\end{aligned}
\end{equation}
$$
其中
$$
\begin{equation}
\hat{\mathbf{x}}_0^t=\frac{\mathbf{x}_t-\sqrt{1-\bar{\alpha}_t}\epsilon_\phi(\mathbf{x}_t,t,y)}{\sqrt{\bar{\alpha}_t}}
\end{equation}
$$
是从$t$时刻预测的pseudo-ground-truth，有意思的是这个预测的方式和[DDIM](../diffusion/DDIM.md)里预测$\mathbf{x}_0$的方式完全一致。
