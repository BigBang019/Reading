# Gaussian Splatting

**3D Gaussian Splatting for Real-Time Radiance Field Rendering.** *TOG 2023.*

- 输入是set of calibrated images和SfM生成的sparse点云





# Related

## Structure from Motion
做什么：multi-view images => camera poses + critital points (sparse)

SfM可作为MVS的前置任务。

## [Multi-view Stero](https://github.com/walsvid/Awesome-MVS#2023-1)

做什么：multi-view calibrated images => 3D point cloud (dense)


- MVSNet: Depth Inference for Unstructured Multi-view Stereo. *ECCV 2022.*


## Point-based Nerf
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