# Gaussian Splatting

*3D Gaussian Splatting for Real-Time Radiance Field Rendering. TOG 2023.*

- 输入是set of calibrated images和SfM生成的sparse点云


### Geometry

3D gaussian $\mathbf{P}\sim \mathcal{N}(\mu, \Sigma)$:
$$
\begin{equation}
\mathcal{G}(\mathbf{P}-\mu)=\frac{1}{2\pi \Sigma(\mathbf{P})^{\frac{1}{2}}}\exp\left(-\frac{1}{2}(\mathbf{P}-\mu)^T\Sigma^{-1}(\mathbf{P})\right)
\end{equation}
$$
如果存在一个$\text{affine transformation}$，$\phi(\mathbf{P})=\mathbf{V}\mathbf{P}+\mathbf{b}$，那么$\phi(\mathbf{P})\sim \mathcal{N}(\mathbf{V}\mu, \mathbf{V}\Sigma \mathbf{V}^T)$。

从world coordinate调整到camera coordinate是一个$\text{affine transformation}$：$\mathbf{P}_{cam}=\mathbf{R}\mathbf{P}_{world}$。此时$\mathbf{P}_{cam}$的covariance matrix为$\mathbf{R}\Sigma\mathbf{R}^T$。

但是2D projection $\mathbf{p}=\mathbf{K}\mathbf{P}$并不是一个$\text{affine transformation}$，其中$\mathbf{P}$为3D space的点，$\mathbf{p}$是一个image plane上的点，$\mathbf{K}$为camera intrinsic matrix。本文用$\text{first-order approximation}$的方式来拟合$\mathbf{p}=\mathbf{K}\mathbf{P}$：
$$
\begin{equation}
\mathbf{p}=\mathbf{p}_0+\mathcal{J}_{\mathbf{P}_0}(\mathbf{P}-\mathbf{P}_0)
\end{equation}
$$
上式子为$\text{affine transformation}$，所以变量$\mathbf{u}$的covariance matrix为$\mathcal{J}\Sigma\mathcal{J}^T$，其中$\mathcal{J}$为$\mathbf{p}=\mathbf{K}\mathbf{P}$变换的$\text{Jacobian matrix}$。

如何求得$\mathcal{J}$？
$$
\mathcal{J}=
\begin{pmatrix}
\frac{\partial u}{\partial x}&\frac{\partial u}{\partial y}&\frac{\partial u}{\partial z}\\
\frac{\partial v}{\partial x}&\frac{\partial v}{\partial y}&\frac{\partial v}{\partial z}\\
\end{pmatrix}=
\begin{pmatrix}
\frac{\alpha_x}{z}&0&-\frac{\alpha_x x}{z^2}\\
0&\frac{\alpha_y}{z}&-\frac{\alpha_y y}{z^2}\\
\end{pmatrix}
$$
综合来看世界坐标投影到image plane的covariance matrix为：
$$
\Sigma'=JR\Sigma R^TJ^T
$$

### Covariance Matrix
假设存在2D Covariance Matrix
$$
\begin{equation}
\Sigma=\begin{pmatrix}
\sigma_{xx}&\sigma_{xy}\\
\sigma_{xy}&\sigma_{yy}\\
\end{pmatrix}
\end{equation}
$$
对矩阵进行特征值分解得到的特征值$\lambda_1$和$\lambda_2$以及对应的特征向量$\mathbf{v}_1$和$\mathbf{v}_2$，那么
- $\lambda_1$和$\lambda_2$分别代表了数据在$\mathbf{v}_1$和$\mathbf{v}_2$上的方差
- 协方差矩阵可以通过椭圆来表示数据分本，中心是数据的均值，椭圆的主轴分别是$\mathbf{v}_1$和$\mathbf{v}_2$的方向，其中长轴、短轴长度分别为$\sqrt{\lambda_1}$、$\sqrt{\lambda_2}$
- 当前协方差矩阵的影响范围半径为$3*\max(\sqrt{\lambda_1}, \sqrt{\lambda_2})$，因为99.7\%的高斯分布的数据在$3\sigma$之内

协方差矩阵的逆
$$
\Sigma^{-1}=\frac{1}{\det(\Sigma)}\begin{pmatrix}
\sigma_{yy}&-\sigma_{xy}\\
-\sigma_{xy}&\sigma_{xx}\\
\end{pmatrix}
$$


# Related

### Structure from Motion
做什么：multi-view images => camera poses + critital points (sparse)

SfM可作为MVS的前置任务。

### [Multi-view Stero](https://github.com/walsvid/Awesome-MVS#2023-1)

做什么：multi-view calibrated images => 3D point cloud (dense)


- MVSNet: Depth Inference for Unstructured Multi-view Stereo. *ECCV 2022.*


### Point-based Nerf
下面讲的基本都是splatting流派

$$
\begin{equation}
\begin{gathered}
C=\sum_{i=1}^N T_i (1-exp(-\sigma_i\delta_i))\mathbf{c}_i\\
T_i=exp(-\sum_{j=1}^{i-1}\sigma_j\delta_j)
\end{gathered}
\end{equation}
$$
其中，点i是沿着一个ray上按步长$\delta$采样一次获得，获得点的密度$\sigma$，透明度$T$，颜色$\mathbf{c}$。


上式的变体：
$$
\begin{equation}
\begin{gathered}
C=\sum_{i=1}^N T_i \alpha_i\mathbf{c}_i\\
\alpha_i=(1-exp(-\delta_i \sigma_i))\quad \text{and}\quad T_i=\prod_{j=1}^{i-1}(1-\alpha_j)
\end{gathered}
\end{equation}
$$

-  Neural Point Catacaustics for Novel-View Synthesis of Reflections. *TOG 2022.*
    - 解决的什么问题：nerf重建反光物体效果不好
    - 怎么解决的：input wrapper先构建反射出来的的点云
    - 仍存在的问题：依赖MVS
- Differentiable Point-Based Radiance Fields for Efficient View Synthesis. *Siggraph Asia 2022*.
    - 解决的什么问题：训练加速（30min），inference加速（40ms），不依赖MVS
    - 怎么解决的：initialize random set of points
    - 仍存在的问题：依赖initialization mask，而且是单物体重建。