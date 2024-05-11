*Gaussian Shading: Provable Performance-Lossless Image Watermarking for Diffusion Models. CVPR 2024.*

# 核心思想
这篇文章的水印只能面向stable diffusion类别的text-to-image models。文章中用$z_T$描述纯高斯噪声，$z_T^s$描述水印过后的高斯噪声。

- 加水印：作者根据secret message $m$采样$z_T^s$，通过DPMSolver得到去噪的水印latent $z_0^s$。水印图片$X^s$通过latent decoder映射得到：$X^s=\mathcal{D}(z_0^s)$。
- 验证水印：给定一个图片$X'^s$，先将其映射回latent space：$z_0'^s=\mathcal{E}(X'^s)$，然后通过DDIM的inverse process得到原先加噪的$z_T^s$，从$z_T^s$上验证水印$m$。

怎么根据$m$采样$z_T^s$？$l$-bit的binary message$m$可以被表示为一个整型数$y\in[0,2^l)$。那么我们有：
$$
\begin{aligned}
p(y)&=\frac{1}{2^l}\\
\end{aligned}
$$
