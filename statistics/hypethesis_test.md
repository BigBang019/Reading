
# Hoteling T^2 Test

### 双样本检验
假设有样本集$\mathbf{X}=\{\mathbf{X}_0, \mathbf{X}_1, ..., \mathbf{X}_{n-1}\}$是一组$k$维正态分布的独立随机变量，具有均值$\mathbf{\mu}_X$和方差$\Sigma$，样本集$\mathbf{Y}=\{\mathbf{Y}_0,\mathbf{Y}_1, ..., \mathbf{Y}_{m-1}\}$是一组$k$维正态分布的独立随机变量，具有均值$\mathbf{\mu}_Y$和方差$\Sigma$，如何检验$\mathbf{X}$和$\mathbf{Y}$是否存在显著性差异？

上述描述有如下限制：
- $\mathbf{X}_i,\mathbf{Y}_j$必须遵循$k$维正态分布
- $\mathbf{X}$的采样互相独立，$\mathbf{Y}$的采样互相独立
- 方差矩阵相同

设立假设：
$$
H_0:\text{样本集}\mathbf{X}和\mathbf{Y}不存在显著性差异
$$

$T^2$和$F$值计算如下：
$$
\mathbf{S}_X=\frac{1}{n-1}\sum_{i=0}^n(\mathbf{X}_i-\mathbf{\mu}_X)(\mathbf{X}_i-\mathbf{\mu}_X)^T\\
\mathbf{S}_Y=\frac{1}{m-1}\sum_{i=0}^m(\mathbf{Y}_i-\mathbf{\mu}_Y)(\mathbf{Y}_i-\mathbf{\mu}_Y)^T\\
\mathbf{S}=\frac{(n-1)\mathbf{S}_X+(m-1)\mathbf{S}_Y}{n+m-2}\\
T^2=(\mu-X-\mu_Y)^T\left\{\mathbf{S}(\frac{1}{n}+\frac{1}{m})\right\}^{-1}(\mu_X-\mu_Y)\\
F=\frac{n+m-p-1}{p(n+m-2)}T^2\sim F_{p,n+m-p-1}
$$
注意上式的$p$并不是*p-value*，而是自由度degree-of-freedom（$df_1$），Hoteling T square Test里的$df_2=n+m-p-1$，所以在置信度$\alpha$下的critical *F-value*可以被标记为$F_{\alpha,df_1,df_2}$。[查表](http://www.socr.ucla.edu/Applets.dir/F_Table.html)。


*p-value*取值范围在$[0,1]$，当*p-value*越大，意味着采样数据与$H_0$越一致，或者说我们无法拒绝$H_0$。

当$F>F_{\alpha,df_1,df_2}$，我们可以拒绝$H_0$

```
# X: (n,p)
# Y: (m,p)
from hotelling.stats import hotelling_t2
T_value, F_value, p_value, cov = hotelling_t2(X, Y)
```