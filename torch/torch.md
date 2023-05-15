# Pytorch

v1.8.2

### nn.BatchNorm2d
Args: num_features $C$
Input: $(N, C, H, W)$
Output: $(N, C, H, W)$
$$y = \frac{x - \mathrm{E}[x]}{ \sqrt{\mathrm{Var}[x] + \epsilon}} * \gamma + \beta$$
$C$ input size


### nn.Conv1d
Args: in_channels $C_{in}$, out_channels $C_{out}$, kernel_size, padding, stride, dilation
Input: $(N, C_{in}, L)$
Output: $(N, C_{out}, L_{out})$

### nn.Conv2d
Args: in_channels $C_{in}$, out_channels $C_{out}$, kernel_size, padding, stride, dilation
Input: $(N, C_{in}, H, W)$
Output: $(N, C_{out}, H_{out}, W_{out})$
$$
\begin{align*}
\text{out}(N_i, C_{\text{out}_j}) = \text{bias}(C_{\text{out}_j}) +
        \sum_{k = 0}^{C_{\text{in}} - 1} \text{weight}(C_{\text{out}_j}, k) \star \text{input}(N_i, k)\\
    H_{out} = \left\lfloor\frac{H_{in}  + 2 \times \text{padding}[0] - \text{dilation}[0]
                        \times (\text{kernel\_size}[0] - 1) - 1}{\text{stride}[0]} + 1\right\rfloor\\
    W_{out} = \left\lfloor\frac{W_{in}  + 2 \times \text{padding}[1] - \text{dilation}[1]
                        \times (\text{kernel\_size}[1] - 1) - 1}{\text{stride}[1]} + 1\right\rfloor
\end{align*}
$$

$C$ number of channels
$N$ batch size
目前来看好像一般都是
kernel_size=3, padding=stride=dilation=1
kernel_size=1, padding=0, stride=dilation=1
这种情况下$H_{in}=H_{out}$

kernel_size=7, padding=3, stride=2, dilation=1

### nn.MaxPool2d
Args: kernel_size, padding, stride, dilation
Input: $(N, C, H, W)$
Output: $(N, C, H_{out}, W_{out})$
$$
\begin{align*}
H_{out} = \left\lfloor\frac{H_{in} + 2 * \text{padding[0]} - \text{dilation[0]}
                    \times (\text{kernel\_size[0]} - 1) - 1}{\text{stride[0]}} + 1\right\rfloor\\
W_{out} = \left\lfloor\frac{W_{in} + 2 * \text{padding[1]} - \text{dilation[1]}
                    \times (\text{kernel\_size[1]} - 1) - 1}{\text{stride[1]}} + 1\right\rfloor
\end{align*}
$$
在kernel窗口中选取最大的一个数，每次移动步长为stride

### nn.AdaptiveAvgPool2d
Args: output_size $S$
Input: $(N, C, H_{in}, W_{in})$
Output: $(N, C, S_{0}, S_{1})$
在kernel窗口中取平均，但是这个是一个自适应层，你的kernel和步长可以自行选择，只要你满足了我的输出shape