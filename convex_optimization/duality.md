# Duality
Consider general minimization problem:
$$
\min_{x} f(x)\text{ subject to}\\
h_i(x)\leq 0,i=1,...,m\\
l_j(x)=0,j=1,...,r
$$
定义Lagrangian:
$$
L(x,u,v)=f(x)+\sum_{i=1}^m u_ih_i(x)+\sum_{j=1}^r v_jl_j(x),\\
\text{with }u\geq 0
$$
$$
f^*\geq \min_{x\in C}L(x,u,v)\geq \min_{x} L(x,u,v):=g(u,v)
$$
因此$g^*$是$f^*$的一个lower bounds，求解原问题（Primal Problem）在特定情况下（Slater's Condition）等效于求解以下对偶问题（Duality Problem）
$$
\max_{u,v} g(u,v)\text{ subject to}\\
u\geq 0
$$

### Strong Duality
$$
\begin{aligned}
\text{(weak duality) }&f^*\geq g^*\\
\text{(strong duality) }&f^*=g^*
\end{aligned}
$$
Slater's condition: if the primal is a convex problem (i.e. $f$ and $h_1,...,h_m$ are convex, $l_1,...,l_r$ are affine), and there exists at least one strictly feasible $x\in \mathbb{R}^n$, meaning: $h_1(x)<0,...,h_m(x)<0, l_1(x)=0,...,l_r(x)=0$, then strong duality holds.

### Duality Gap
Given primal feasible $x$ and dual feasible $u,v$, the quantity:
$$
f(x)-g(u,v)
$$
is called the duality gap between $x$ and $u,v$. Note that
$$
f(x)-f^*\leq f(x)-g(u,v)
$$
这有什么用？
- If gap is zero, $x$ is primal optimal
- If we are guaranteed $f(x)-g(u,v)\leq \epsilon$, then we are guaranteed $f(x)-f^*\leq \epsilon$

### KKT Conditions
- $0\in \partial \left(f(x)+\sum_{i=1}^m u_i h_i(x)+\sum_{j=1}^r v_j l_j(x)\right)$
- $u_i\cdot h_i(x)=0,\ \forall i$
- $h_i(x)\leq 0, l_j(x)=0,\ \forall i,j$
- $u_i\geq 0,\ \forall i$

$$
\begin{aligned}
&\text{strong duality}\\
\Leftrightarrow&x^*\text{ and }u^*,v^*\text{ are primal and dual solutions}\\
\Leftrightarrow&x^*\text{ and }u^*,v^*\text{ satisfy the KKT conditions}
\end{aligned}
$$

这有啥用？至少可以证明**Uniqueness in $l_1$ penalized problems**

### Uniqueness in $l_1$ penalized problems

Let $f$ be differentiable and strictly convex, let $X\in \mathbb{R}^{n\times p}, \lambda >0$. Consider
$$
\min_\beta f(X\beta)+\lambda \lVert \beta\rVert _1
$$
If the entries of $X$ are drawn from a continuous probability distribution (on $\mathbb{R}^{n\times p}$), then w.p. 1 there is a unique solution and it has at most $\min\{n,p\}$ nonzero components.


### Dual Norm

Dual norm $\lVert x\rVert _*$:
$$
\lVert x\rVert _*=\max_{\lVert z\rVert \leq 1} z^Tx
$$

# Conjugate Function
假设
$$
f:\mathbb{R}^n\mapsto \mathbb{R}
$$
他的Conjugate Function $f^*:\mathbb{R}^n\mapsto \mathbb{R}$可以通过Legendre-Fenchel变换得到：
$$
\begin{aligned}
f^*(y)=\max_{x\in \mathbb{R}^n}\left\{\left<x,y\right>-f(x)\right\}\\
-f^*(u)=\min_{x\in\mathbb{R}^n} \{f(x)-u^Tx\}
\end{aligned}
$$
Conjugate function的属性：
- $f(x)+f^*(y)\geq x^Ty$
- $f^{**}\leq f$
- If $f$ is closed and convex, then $f^{**}=f$
- If $f$ is closed and convex, then $\forall x,y,\ x\in \partial f^*(y)\Leftrightarrow y\in \partial f(x)\Leftrightarrow f(x)+f^*(y)=x^Ty$
- If $f(u,v)=f_1(u)+f_2(v)$, then $f^*(w,z)=f^*_1(w)+f^*_2(z)$

几个例子：
- $f(x)=\frac{1}{2}x^TQx$, $f^*(y)=\frac{1}{2}y^T Q^{-1}y$
- $f(x)=I_C(x)$, $f^*(y)=I_C^*(y)=\max_{x\in C}y^Tx$
- $f(x)=\lVert x\rVert$, $f^*(y)=I_{\{z:\lVert z\rVert _*\leq 1\}}(y)$



有什么用？Conjugate function可以把问题转化成dual problem

### Conjugates and dual problems
Consider:
$$
\begin{aligned}
&\min_{x}f(x)+g(x) &\text{ (Original Problem)}\\
\Leftrightarrow&\min_{x,z}f(x)+g(z)\text{ subject to } x=z &\text{ (Add a dummy variable)}\\
&g(u)=\min_{x}\{f(x)+g(z)+u^T(z-x)=-f^*(u)-g^*(-u)\} &\text{ (Get the dual function)}
\end{aligned}
$$
因此我们可以得到：
$$
\min_{x}f(x)+g(x)\Leftrightarrow \max_{u}-f^*(u)-g^*(-u)
$$
