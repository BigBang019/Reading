# Fourier Transform

我们从一个猜想开始，任意的连续信号可不可以用下面的表达式表示：
$$
\begin{equation}
f(t)=\sum_{k=-n}^n C_k e^{2\pi ki}
\end{equation}
$$
那么我们可以做以下推导：
$$
\begin{equation}
\begin{aligned}
C_me^{2\pi mit}&=f(t)-\sum_{k\neq m}C_ke^{2\pi kit}\\
C_m&=e^{-2\pi mit}f(t)-\sum_{k\neq m}C_ke^{-2\pi mit}e^{2\pi kit}\\

\end{aligned}
\end{equation}
$$


<!-- #### 问题：从sample重建连续函数
定义
$$
\begin{equation}
    \begin{gathered}
        \Psi_p(x)=\sum_{k=-\infty}^{\infty}\delta(x-kp)\\
        \left [\mathcal{F}\Psi_p \right ](w)=\frac{1}{p}\Psi_{\frac{1}{p}}(w)\\
        \left [\mathcal{F}^{-1}\Psi_p \right ](t)=\frac{1}{p}\Psi_{\frac{1}{p}}(t)
    \end{gathered}
\end{equation}
$$
那么sampling就是
$$
f(x)\Psi_p(x)=\sum_{k=-\infty}^{\infty}f(kp)\Psi_p(x-kp)
$$
如果我们假设$f$的频域限定在$[-p/2,p/2]$，

$$
\begin{equation}
    \begin{gathered}
        \pi_p\left[ (\mathcal{F}f)*\Psi_p\right]=(\mathcal{F}f)\\
    \end{gathered}
\end{equation}
$$
$$
\begin{equation}
    \begin{split}
        f(t)&=\mathcal{F}^{-1}\left [ \pi_p\left[ (\mathcal{F}f)*\Psi_p\right] \right]\\
        &=(\mathcal{F}^{-1}\pi_p) * \mathcal{F}^{-1}[(\mathcal{F}f) * \Psi_p]\\
        &=(\mathcal{F}^{-1}\pi_p) * [\mathcal{F}^{-1}(\mathcal{F}f) \cdot \mathcal{F}^{-1}\Psi_p]\\
        &=p\cdot sinc(pt) * [\frac{1}{p}\sum_{k=-\infty}^\infty f(\frac{k}{p})\delta(t-\frac{k}{p})]\\
        &=\sum_{k=-\infty}^{\infty}f(\frac{k}{p})[sinc(pt)*\delta(t-\frac{k}{p})]\\
        f(t)&=\sum_{k=-\infty}^{\infty}(\frac{k}{p})sinc(pt-k)
    \end{split}
\end{equation}
$$
可见需要无穷的采样 -->
