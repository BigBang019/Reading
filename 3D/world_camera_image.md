# World, Camera, Image Plane

### World2Camera

<img src="../pasteImage/nerf_axis.png" width="400">


相机在世界坐标$\mathbf{c}=(c_x,c_y,c_z)$下。

$$
\begin{equation}
\begin{split}
\mathbf{v}_{f}=\mathbf{v}_{forward}&=\frac{\mathbf{c}}{\lVert\mathbf{c}\rVert _2^2}\\
\mathbf{v}_{r}=\mathbf{v}_{right}&=\mathbf{v}_{f}\times\mathbf{y}\\
\mathbf{v}_{u}=\mathbf{v}_{up}&=\mathbf{v}_{r}\times \mathbf{v}_{forward}\\
\end{split}
\end{equation}
$$

基于以上，我们可以定义相机坐标转换世界坐标的矩阵$\mathbf{M}_{pose}$：
$$
\begin{equation}
\mathbf{M}_{pose}=
\begin{pmatrix}
| & | & | & |\\
\mathbf{v}_{r} & \mathbf{v}_{u} & \mathbf{v}_{f} & \mathbf{c}\\
| & | & | & |\\
0 & 0 & 0 & 1\\
\end{pmatrix}
\end{equation}
$$


### Camera2ImagePlane

<img src="../pasteImage/camera_image.avif" width="400">

此时$M(X,Y,Z)$为camera coordinate下的坐标，$f$为focal length，因为$m$和$M$在一条线上，所以满足
$$
\begin{equation}
u=f\frac{X}{Z}\quad v=f\frac{Y}{Z}
\end{equation}
$$

<img src="../pasteImage/camera_image2.avif" width="400">

因为像素坐标是原点在边角，所以需要平移原点：
$$
\begin{equation}
u=f\frac{X}{Z}+p_u\quad v=f\frac{Y}{Z}+p_v
\end{equation}
$$

假设真实的画布长宽$(H\times W)$：
$$
\begin{equation}
Hx=f\frac{X}{Z}+p_u\quad Wy=f\frac{Y}{Z}+p_v
\end{equation}
$$

等价于：
$$
\begin{equation}
x=\alpha_x\frac{X}{Z}+p_x\quad y=\alpha_y\frac{Y}{Z}+p_y
\end{equation}
$$

整理成矩阵形式：
$$
\begin{equation}
Zm=
Z\begin{pmatrix}
x\\
y\\
1
\end{pmatrix}
=\begin{pmatrix}
\alpha_x & 0 & p_x\\
0 & \alpha_y & p_y\\
0 & 0 & 1 \\
\end{pmatrix}
\begin{pmatrix}
X\\
Y\\
Z
\end{pmatrix}
\end{equation}
$$

更加通用的表达式，s表示由于传感器没有垂直于光轴，传感器轴之间可能发生的倾斜：
$$
\begin{equation}
KM=
\begin{pmatrix}
\alpha_x & s & p_x\\
0 & \alpha_y & p_y\\
0 & 0 & 1 \\
\end{pmatrix}
\begin{pmatrix}
X\\
Y\\
Z
\end{pmatrix}
\end{equation}
$$
得到$Zm=KM$

### Ray生成

<img src="../pasteImage/ray_generate.png">