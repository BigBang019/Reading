
*本文采用statistical notation*

# 条件概率

$$
\begin{equation}
p_{A,B}(a,b)=p_{A|B}(a|b)p_B(b)
\end{equation}
$$
全概率公式
$$
\begin{equation}
p_A(a)=\int p_{A,B}(a,b) \text{d}b= \int p_{A|B}(a|b)p_B(b) \text{d}b
\end{equation}
$$



# 期望
期望$\mathbb{E}_{X}\left[ X\right]$表示随机变量$X$遵循$p_X(x)$的分布下，$X$的均值：
$$
\begin{equation}
\mathbb{E}_{X}\left[ X\right] = \int_{\Omega} x p(x) dx
\end{equation}
$$
期望$\mathbb{E}_{X}\left[ g(X)\right]$表示随机变量$X$遵循分布$p_X(x)$的情况下$g(X)$的均值：
$$
\begin{equation}
\mathbb{E}_{X}\left[ g(X)\right] = \int_{\Omega} g(x) p_X(x) dx
\end{equation}
$$
条件期望$\mathbb{E}_{X|Y}\left[ X|Y=y\right]$表示随机变量$X$遵循$p_{X|Y}(x|y)$下，$X$的均值：
$$
\begin{equation}
\mathbb{E}_{X|Y}\left[ X|Y=y\right] = \int_{\Omega} x p(x|y) \text{d}x
\end{equation}
$$
全期望公式：
$$
\begin{equation}
\mathbb{E}_{X}\left[ X\right] = \int_{\Omega} \mathbb{E}_{X|Y}\left[X|Y=y\right] f_Y(y)\text{d}y
\end{equation}
$$


# 联合分布
$$
f_\mathbf{X}(\mathbf{x})=f_{X_1,...,X_n}(x_1, ..., x_N)
$$

假设随机变量$X_1, X_2, ..., X_n$拥有联合密度$f_{X_1, X_2, ..., X_n}(x_1, x_2, ..., x_n)$，那么
$$
\begin{equation}
f_{X_1}(x_1)=\int f_{X_1, X_2, ..., X_n}(x_1, x_2, ..., x_n) \text{d}x_2\text{d}x_3...\text{d}x_n
\end{equation}
$$


假设随机变量$X_1, X_2, ..., X_n$拥有联合密度$f_{X_1, X_2, ..., X_n}(x_1, x_2, ..., x_n)$，如果$X_i$是$i.i.d$的那么我们有
$$
\begin{equation}
    \begin{gathered}
    f_{X_1, X_2, ..., X_n}(x_1, x_2, ..., x_n) = \prod_{i=1}^n f_{X_i}(x_i)\\
    f_{X_1, X_2, ..., X_n|\Theta}(x_1, x_2, ..., x_n|\theta) = \prod_{i=1}^n f_{X_i|\Theta}(x_i|\theta)\\
    \end{gathered}
\end{equation}
$$

