# Fourier Transform (Extension)

#### 问题：从sample重建连续函数
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
可见需要无穷的采样

#### 问题：从有限sample重建连续函数
根据[1]的信息，Prolate Spheroidal Wave Functions


## Reference
[1] Avron, Haim, et al. "A universal sampling method for reconstructing signals with simple fourier transforms." Proceedings of the 51st Annual ACM SIGACT Symposium on Theory of Computing. 2019.