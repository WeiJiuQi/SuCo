---
name: SRHT Wrap-Around Fix
overview: 用"坐标回绕 (wrap-around)"策略替代零填充，使 SRHT 变换质量不再受 d_padded 与 d_orig 差异影响，当 d 为 2 的幂时行为完全不变。
todos:
  - id: update-srht-header
    content: 在 SRHTContext 中添加 wrap_scale 字段
    status: completed
  - id: update-srht-init
    content: 在 init_srht 中计算 wrap_scale
    status: completed
  - id: update-srht-apply
    content: 在 apply_srht_batch 中将零填充替换为坐标回绕
    status: completed
  - id: verify-build
    content: 编译验证 + 运行保距性测试
    status: completed
isProject: false
---

# SRHT Wrap-Around Fix

## 问题分析

当前实现在 `d_orig` 不是 2 的幂时，将向量从 d_orig 零填充到 d_padded：

```63:68:src/srht.cpp
        for (int j = 0; j < d_orig; j++) {
            buf[j] = data[i][j] * ctx.signs[j];
        }
        for (int j = d_orig; j < d_padded; j++) {
            buf[j] = 0.0f;
        }
```

零填充的问题：FWHT 输出的每个分量是 d_padded 个输入的加权和，但其中 `d_padded - d_orig` 项恒为零，不携带信息。这导致：

- 子采样 m 个分量时，每个分量的有效信号占比降低
- 距离估计的方差增大，召回率下降
- 当 d_orig = d_padded（2 的幂）时无此问题

## 解决方案：坐标回绕 (Wrap-Around)

**核心思想**：将 d_orig 维的坐标以取模方式回绕填满 d_padded 个位置，不留任何零。每个位置乘以独立随机符号，并通过缩放因子保证等距性。

**数学表述**：

定义嵌入矩阵 W (d_padded x d_orig)：

- W[j][k] = wrap_scale[k] 当 j % d_orig == k，否则为 0
- count_k = d_padded 中满足 j % d_orig == k 的 j 的个数
- wrap_scale[k] = 1 / sqrt(count_k)

则 W^T W = I（等距嵌入），即 ||Wx||^2 = ||x||^2。

变换后的缓冲区：`buf[j] = x[j % d_orig] * signs[j] * wrap_scale[j % d_orig]`

当 d_padded == d_orig 时，count_k = 1，wrap_scale[k] = 1.0，退化为当前实现，行为完全不变。

## 修改清单

### 1. [src/srht.h](src/srht.h)

在 `SRHTContext` 中新增 `wrap_scale` 字段：

```cpp
struct SRHTContext {
    int d_orig;
    int d_padded;
    int m;
    std::vector<float> signs;
    std::vector<int> sample_idx;
    std::vector<float> wrap_scale;  // 新增：每个原始坐标的缩放因子，长度 d_orig
    unsigned int seed;
};
```

### 2. [src/srht.cpp](src/srht.cpp)

**init_srht 中**：计算每个原始坐标在回绕中出现的次数，并生成 wrap_scale：

```cpp
// 计算回绕缩放因子
ctx.wrap_scale.resize(d_orig);
for (int k = 0; k < d_orig; k++) {
    int count = ctx.d_padded / d_orig + (k < ctx.d_padded % d_orig ? 1 : 0);
    ctx.wrap_scale[k] = 1.0f / sqrtf((float)count);
}
```

**apply_srht_batch 中**：将零填充替换为回绕填充：

```cpp
// 当前（零填充）：
for (int j = 0; j < d_orig; j++) {
    buf[j] = data[i][j] * ctx.signs[j];
}
for (int j = d_orig; j < d_padded; j++) {
    buf[j] = 0.0f;
}

// 改为（坐标回绕）：
for (int j = 0; j < d_padded; j++) {
    int k = j % d_orig;
    buf[j] = data[i][k] * ctx.signs[j] * ctx.wrap_scale[k];
}
```

其余代码（FWHT、子采样、缩放因子 `1/sqrt(m)`）均不变。

## 正确性验证

- **保范性**：||buf||^2 = sum_k count_k * wrap_scale[k]^2 * x_k^2 = sum_k x_k^2 = ||x||^2
- **保距性**：对差向量 z = x - y，同理 ||buf^x - buf^y||^2 = ||x - y||^2
- **缩放因子**：FWHT 后 ||H * buf||^2 = d_padded * ||x||^2，子采样 m 个后期望为 m * ||x||^2，乘 1/sqrt(m) 后期望 ||x||^2，不变
- **d_padded = d_orig 时**：count_k = 1, wrap_scale = 1.0，完全退化为现有实现

## 为什么回绕优于零填充

- 零填充：d_padded - d_orig 个位置恒为零，FWHT 输出中对应的信号分量被"稀释"，距离估计方差大
- 回绕：所有 d_padded 个位置都携带信号，FWHT 充分混合，距离估计方差与无填充的 2 的幂情况相当

