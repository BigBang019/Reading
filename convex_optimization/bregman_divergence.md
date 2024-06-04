# Bregman Divergence

Bregman divergence定义的是特定空间里两个点的距离。

Given a differentiable, strictly convex function $h:\mathcal{M}\mapsto \mathbb{R}$ on a convex set $\mathcal{M}\subset \mathbb{R}^n$.

Bregman divergence with respect to $h$:
$$
D_h(y||x)=h(y)-h(x)-\left<\nabla h(x), y-x\right>
$$

目前对$h$的理解是衡量一个点$x$的metrics，比如在$\mathbb{R}^n$空间中：
$$
h(x)=\lVert x\rVert_2^2
$$
此时
$$
D_h(y||x)=\lVert y-x\rVert _2^2
$$

# Bregman Projection
The Bregman Projection of point $x'$ onto convex set $\mathcal{M}$ is:
$$
x^+=\argmin_{x\in \mathcal{M}} D_h(x||x')
$$

# Reference
https://www.zhihu.com/question/22426561/answer/209945856