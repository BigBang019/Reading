# Mirror Descent Takeaway
Motivation：Mirror Descent是解决convex optimization的first-order methods的统一框架。

### 解决什么问题

$$
\min_{x\in K} \left\{f(x)+h(x)\right\}
$$

We first define a closed convex set $K\subset \mathbb{R}^n$, and differentiable function $f(x):K\mapsto \mathbb{R}$ is G-Lipschitz with respect to $\lVert \cdot\rVert$ if:
$$
\lVert \nabla f(x)\rVert _* \leq G,\ \forall x\in \mathbb{R}^n
$$

Convex and differentiable funciton $h(x):\mathbb{R}^n\mapsto \mathbb{R}$ is a $\alpha$-strongly convex with respect to $\lVert \cdot \rVert$ if:
$$
h(y)\geq h(x)+\left <\nabla h(x), y-x\right>+\frac{\alpha}{2}\lVert y-x\rVert _2^2
$$

一般来讲$f(x)$作为目标函数，$h(x)$作为限制函数。

### Mirror Descent
- Map $x_t$ to dual space point $\theta_t$ through mirror map $\nabla h$: $\theta_t = \nabla h(x_t)$
- Take a step in dual space: $\theta_{t+1}=\theta_t-\eta\nabla f(\theta_t)$
- Map $\theta_t$ to primal space point $x_{t+1}'$: $x_{t+1}'=(\nabla h)^{-1}(\theta_{t+1})$
- Do [Bregman Projection](./bregman_divergence.md) $x_{t+1}'$ to a "close" $x_{t+1}\in K$.


迭代式：
$$
x_{t+1}\leftarrow \argmin_{x\in K} \left\{\eta \left<\nabla f(x_t), x\right>+D_h(x||x_t)\right\}
$$


### 其他常见迭代式

$$
x_{t+1}=\nabla h^*(\nabla h(x_t)-\eta \nabla f(x_t))
$$
where $h^*(\theta)=\sup_{x\in K}\left[\left<x,\theta\right>-h(x)\right]$


# QA

### 和Gradient Descent什么关系
当$h_1(x)=\frac{1}{2}\lVert x\rVert _2^2$，此时$D_h(x||x_t)=\frac{1}{2}\lVert x-x_t\rVert _2^2$。求解$x_{t+1}$等效于求解：
$$
\begin{aligned}
\nabla_x \left(\eta \left<\nabla f(x_t), x\right> +\frac{1}{2}\lVert x-x_t\rVert _2^2\right)&=0\\
\eta \nabla f(x_t)+(x-x_t)&=0\\
x&=x_t-\eta \nabla f(x_t)\\
\end{aligned}
$$
这时候Mirror Descent退化为Gradient Descent。

### 和Proximal Gradient Descent什么关系

TODO

# Reference
https://www.cs.cmu.edu/~anupamg/advalgos17/scribes/lec15.pdf

https://tlienart.github.io/posts/2018/10/27-mirror-descent-algorithm/