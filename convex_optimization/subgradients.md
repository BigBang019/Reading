# Subgradients

**Subgradients**的定义主要是为了引出**First-order Optimality**。

### Subgradients

A subgradient of a convex function $f$ at $x$ is a vector $g\in \mathbb{R}^n$ such that:
$$
f(y)\geq f(x)+g^T(y-x), \ \forall y
$$
If $f$ is differentiable at $x$, then $g=\nabla f(x)$ uniquely.

### Subdifferential

Set of all subgradients of convex $f$ is called the subdifferential:
$$
\partial f(x)=\left\{g\in \mathbb{R}^n:g\text{ is a subgradient of }f\text{ at }x\right\}
$$

- If $f$ is convex, $\partial f$ is non-empty.
- $\partial f(x)$ is closed and convex
- If $f$ is differentiable at $x$, $\partial f(x)=\left\{\nabla f(x)\right\}$
- If $\partial f(x)=\left\{g\right\}$, then $f$ is differentiable at $x$ and $\nabla f(x)=g$

### Subgradients & convex geometry

Indicator function $I_C:\mathbb{R}^n\mapsto\mathbb{R}$,
$$
I_C(x)=\left\{
    \begin{aligned}
    0&\quad\text{if }x\in C\\
    \inf&\quad\text{if }x\notin C\\
    \end{aligned}
\right.
$$
For $x\in C$, $\partial I_C(x)=\mathcal{N}_C(x)$, the normal cone of $C$ at $x$ is:
$$
\mathcal{N}_C(x)=\left \{g\in \mathbb{R}^n:g^T(x-y)\geq 0, \ \forall y\in C\right\}
$$

### Optimal Condition

$$
\begin{aligned}
x^*=\argmin_{x}f(x)&\Leftrightarrow 0\in \partial f(x^*)\\
\text{(proof)}\quad0\in \partial f(x^*)&\Leftrightarrow f(y)\geq f(x^*)+0^T(y-x^*),\ \forall y\\
&\Leftrightarrow f(x^*)=\min_x f(x) 
\end{aligned}
$$

### First-order Optimality

对于问题
$$
\min_{x} f(x)\text{ subject to }x\in C
$$
其中$C$为convex domain。

$$
\begin{aligned}
x^*=\argmin_{x} f(x)+I_C(x)&\Leftrightarrow \nabla f(x)^T(y-x)\geq 0 \ \forall y\in C\\
\text{(proof)}\quad 0\in \partial(f(x)+I_C(x)) &\Leftrightarrow 0\in \{\nabla f(x)\}+\mathcal{N}_C(x)\\
&\Leftrightarrow -\nabla f(x)\in \mathcal{N}_C(x)\\
&\Leftrightarrow -\nabla f(x)^Tx\geq -\nabla f(x)^Ty \ \forall y\in C\\
&\Leftrightarrow \nabla f(x)^T (y-x)\geq 0\ \forall y\in C
\end{aligned}
$$
