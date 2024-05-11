Tree-Rings Watermarks: Invisible Fingerprints for Diffusion Images. NeurIPS 2023.

# 核心思路
- 加水印：作者在$x_T$的频域上加入特定的pattern得到水印过后的$x_T'$，得到的$x_T'$可以通过DDIM的reverse process转化为水印过后的图片$x_0'$。
- 验证水印：需要将$x_0'$通过DDIM的inverse process重新转化回$x_T'$。标记$y=\mathcal{F}(x_T')$为噪声图的频域图，作者在$y$上通过假设检验的方式检测水印的pattern是否存在。

设计的pattern是什么？文章中提出了三类pattern：$TreeRing_{zero},TreeRing_{rand},TreeRing_{rings}$，他们有通用的范式：作者选定一个binary mask $M$，并且生成一个key $k^*\in \mathbb{C}^{|M|}$，用$k^*_i$替换mask内的$\mathcal{F}(x_T)_i$，即：
$$
\mathcal{F}(x_T)_i\leftarrow
\left\{
    \begin{aligned}
    &k_i^* &\text{if} \ i\in M\\
    &\mathcal{F}(x_T)_i&\text{otherwise}\\
    \end{aligned}
\right.
$$
不同的pattern只是影响了$k^*$的生成。

那么怎么通过假设检验检测？作者衡量以下指标
$$
d_\text{detection distance}=\frac{1}{|M|}\sum_{i\in M}\left|k_i^*-\mathcal{F}(x'_T)_i\right|
$$
如果$\mathcal{F}(x_T')$未加水印，即遵循多维高斯分布的话，那么显然$d_\text{detection distance}$遵循$\mathcal{X}^2$分布。既然知道了检验指标$d_\text{detection distance}$的分布，剩下的就是简单的hypothesis test。