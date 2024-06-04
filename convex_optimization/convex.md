# Convex Set

### Simplex

$$
\left\{\mathbf{x}\in\mathbb{R}^n_+: \sum_{i=1}^n x_i=1\right\}
$$
一维情况下为point， 二维情况下为line segment，三维为triangle，四维为tetrahedron，...

# Convex Function

A function $f:\mathcal{M}\mapsto \mathbb{R}$ on a convex $\mathcal{M}\subset \mathbb{R}^n$ is a convex function if:
$$
f(tx+(1-t)y)\leq tf(x)+(1-t)f(y),\ \forall t\in [0,1], x,y\in \mathcal{M}
$$

Strictly convex:
$$
f(tx+(1-t)y)<tf(x)+(1-t)f(y), \ \forall t\in (0,1), x\neq y
$$

$\text{strongly convex}\Rightarrow \text{strictly convex}\Rightarrow \text{convex}$

### Equaivalent Condition
A function $f$ is convex:
- $f(y)\geq f(x)+\nabla f(x)^T(y-x), \ \forall x,y$ 

# Strongly Convex

A differentable function $f$ is strongly convex if
$$
f(y)\geq f(x)+\nabla f(x)^T(y-x)+\frac{\mu}{2}\lVert y-x\rVert^2
$$
for some $\mu>0$ and all $x,y$.

A function is n-strongly convex (or strongly convex modulus n) if $\mu=n$.

### Equivalent conditions.
A function $f$ is strongly-convex with constant $\mu>0$ if:
- $f(y)\geq f(x)+\nabla f(x)^T(y-x)+\frac{\mu}{2}\lVert y-x\rVert^2,\ \forall x,y$
- $g(x)=f(x)-\frac{\mu}{2}\lVert x\rVert^2, \ \forall x$ is convex
- $(\nabla f(x)-\nabla f(y))^T(x-y)\geq \mu \lVert x-y\rVert^2, \ \forall x,y$
- $f(\alpha x+(1-\alpha)y)\leq \alpha f(x)+(1-\alpha)f(y)-\frac{\alpha(1-\alpha)\mu}{2}\lVert x-y\rVert^2,\ \alpha\in[0,1]$

### Implications

If $f$ is differentiable, the following conditions are all implied by strong convexity conditions:
- $\frac{1}{2}\lVert\nabla f(x)\rVert^2\geq \mu(f(x)-f^*), \ \forall x$
- $\lVert \nabla f(x)-\nabla f(y)\rVert \geq \mu\lVert x-y\rVert, \ \forall x,y$
- $f(y)\leq f(x)+\nabla f(x)^T(y-x)+\frac{1}{2\mu}\lVert \nabla f(y)-\nabla f(x)\rVert^2, \ \forall x,y$
- $(\nabla f(x)-\nabla f(y))^T(x-y)\leq \frac{1}{\mu}\lVert \nabla f(x)-\nabla f(y)\rVert^2, \ \forall x,y$

### Application
- Let $h=f+g$ where $f$ is strongly convex function and $g$ is a convex function, then $h$ is strongly convex function.

- If $f$ is a strongly convex function, then $\min_{x} f(x)$ has unique solution.