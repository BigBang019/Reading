# Proximal Gradient Descent Takeaway

### 解决什么问题

$$
\min_{x} f(x)=\min_{x}\left\{g(x)+h(x)\right\}
$$

$g$ convex，differentiable
$h$ convex，not differentiable，一般作为限制函数。

### Proximal Mapping

$$
Prox_{h,t}(x)=\argmin_z \frac{1}{2t} \lVert x-z\rVert _2^2 + h(z)
$$
即给定一个$x$，找到最优点$z=prox_{h,t}(x)$，使得$\frac{1}{2t} \lVert x-z\rVert _2^2 + h(z)$最小，这样的$z$能够使得$h$足够小，而且接近不可微点$x$。

### Proximal Gradient Descent

$$
x_{k+1}=Prox_{h,t_k}(x_k-t_k\nabla g(x_k))
$$
即给定起点$x_k$，首先沿着$g$的负梯度方向更新一个值$x_k-t_k\nabla g(x_k)$，然后用近端映射寻找一个$z$，这个$z$能使得不可微函数$h$足够小，而且接近这个$x_k-t_k\nabla g(x_k)$，就用这个$z$作为本次迭代的更新值。

# QA

### 和Gradient Descent什么关系

常规梯度下降：$h(x)=0$。

近端映射$prox_{t}(x)=\argmin_{z} \frac{1}{2t}\lVert x-z\rVert _2^2 + 0=x$

这个时候的更新策略：$x_{k+1}=x_{k}-t_k\nabla g(x_k)$

# 和Projection Descent什么关系

投影梯度下降：$h(x)=I_C(x)$

近端映射：$prox_{t}=\argmin_{z}\frac{1}{2t}\lVert x-z\rVert _2^2+I_C(z)=\argmin_{z\in C}\frac{1}{2t}\lVert x-z\rVert _2^2=\pi_{C}(x)$

更新策略：$x_{k+1}=\pi_C(x_k-t_k \nabla g(x_k))$
