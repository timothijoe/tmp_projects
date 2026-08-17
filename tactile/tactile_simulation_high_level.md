# 机器人触觉仿真：High-level 技术路线分析

## 1. 问题定义

机器人触觉仿真的核心问题不是“选择哪个软件”，而是：

> 如何将物理引擎中的接触过程转换为可供机器人策略或学习模型使用的 tactile observation？

一个完整的数据链路可以写成：

```text
robot and object geometry
          ↓
collision detection
          ↓
contact physics
          ↓
virtual tactile sensor model
          ↓
tactile observation
          ↓
policy / representation model
```

物理引擎通常能够提供接触点、法向、冲量、力和穿透深度，但这些数据还不是最终的触觉表示。触觉仿真的主要差异在于：如何对底层 contact data 进行空间组织、物理建模和观测编码。

目前可以将方法概括为三类：

```text
Tactile Simulation
│
├── 1. Contact / Force Tactile
├── 2. Taxel / Pressure-map Tactile
└── 3. Vision-based Tactile
```

三类方法的关系可以理解为逐步增加空间信息和传感器建模深度：

```text
接触了吗、力有多大？
            ↓
力分布在表面什么位置？
            ↓
柔软触觉表面产生了什么形变和光学图像？
```

## 2. Contact / Force Tactile

### 2.1 核心思想

Contact / Force Tactile 直接使用刚体物理引擎的接触结果，将一个 link 或一个预定义区域上的接触聚合为低维信号。

```text
collision geometry
       ↓
contact solver
       ↓
contact points / forces / impulses
       ↓
body-level or region-level aggregation
       ↓
contact observation
```

典型输出包括：

- 是否接触；
- 区域接触强度；
- 三维净接触力；
- 三维净接触冲量；
- 少量接触点及其位置和法向。

典型张量为：

$$
\mathbf s\in\mathbb R^R
$$

或：

$$
\mathbf F\in\mathbb R^{R\times3}
$$

其中 $R$ 是触觉区域或被监测 rigid body 的数量。

### 2.2 这种方法模拟到了哪一层？

它主要模拟：

```text
接触物理
    +
区域级信号聚合
```

它通常不模拟：

- 连续压力分布；
- 柔性表面形变；
- 传感器内部结构；
- marker displacement；
- 光学成像。

因此，这类方法应称为 **physics-grounded contact abstraction**，而不是高保真传感器渲染。

### 2.3 代表方案

| 方案 | 区域如何定义 | 主要输出 |
|---|---|---|
| ManiSkill3 Allegro Touch | 16 个独立 FSR collision links | 16 个区域的 impulse magnitude |
| Isaac Sim Experimental Contact Sensor | rigid body 上的空间 sensor/filter | scalar force reading 或 raw contacts |
| Isaac Lab ContactSensor | 匹配一组 rigid-body prims | batched 3D net contact force |
| MuJoCo Touch Sensor | site volume | site 内法向力总和 |
| Gazebo Contact Sensor | collision element | 变长 contact messages |

### 2.4 ManiSkill3 Allegro Touch 的位置

ManiSkill3 Allegro Touch 明确属于：

```text
Contact / Force Tactile
└── Region-level impulse tactile
    └── FSR-layout-inspired collision regions
```

其原始输出为：

$$
\mathbf J\in\mathbb R^{N_{env}\times16\times3}
$$

随后对每个三维冲量向量取模：

$$
s_r=\lVert\mathbf J_r\rVert_2
$$

形成：

$$
\mathbf s\in\mathbb R^{N_{env}\times16}
$$

这里的 16 个数表示 16 个区域的接触冲量强度，不表示区域内部的压力分布，也不是 tactile RGB image。

其区域与数据链路可以概括为：

```text
16 个独立 FSR collision links
        ↓
PhysX 计算每个区域参与的接触
        ↓
按 FSR link 聚合三维 contact impulse
        ↓
对三维向量取模
        ↓
固定长度 16 维 observation
```

因此，FSR link 不是原始手指 mesh 上的一个数学采样点，而是附着在手上的、具有碰撞几何和有限面积的虚拟感知区域。它的优点是传感器布局和输出维度已经确定；代价是每个区域内部的接触位置、形状和方向信息会在聚合及取模后丢失。

### 2.5 Isaac Sim Experimental Contact Sensor

Isaac Sim Experimental Contact Sensor 属于：

```text
Contact / Force Tactile
└── Sensor-level contact reporting
    ├── fixed-size summarized reading
    └── variable-length raw contact records
```

传感器 prim 附着在具有 rigid-body ancestor 的场景层级中。其空间位置、半径或过滤配置用于决定哪些 PhysX contact reports 被传感器接收；这个 sensor/filter 本身不等同于一个新的碰撞体，也不会因为画了一个半径就自动生成物理接触。

高层接口可为单个 sensor 提供：

$$
f\in\mathbb R,\qquad b\in\{0,1\},\qquad n_c\in\mathbb N
$$

其中：

- $f$ 是该传感器当前报告的接触力强度标量；它适合表示“接触有多强”，但单独不能恢复力的方向或法向、切向分量；
- $b$（`in_contact`）表示当前是否至少存在一个满足过滤条件的接触；
- $n_c$（`number_of_contacts`）表示本帧返回的接触记录数量，而不是接触面积。更多 contact records 可能来自接触流形离散、多个碰撞体或求解器行为，不能直接解释为面积更大。

若读取 raw data，当前帧可表示为：

$$
\mathbf P,\mathbf N,\mathbf J\in\mathbb R^{M\times3}
$$

其中 $M$ 随帧变化；每条记录通常包含参与接触的 body、世界坐标位置 $\mathbf p_i$、接触法向 $\mathbf n_i$、冲量 $\mathbf J_i$ 和时间步长 $\Delta t$。近似平均接触力可以由冲量除以时间步理解：

$$
\mathbf F_i\approx\frac{\mathbf J_i}{\Delta t}
$$

数据链路为：

```text
rigid-body collisions
        ↓
PhysX contact reports
        ↓
sensor spatial/body filters
        ├── summarized scalar force + contact flag + count
        └── M 条 position/normal/impulse raw records
```

这里的 $M$ 条记录不是预先固定在 link 表面的 $M$ 个传感点，而是本帧由碰撞求解器产生的接触记录。若要输入神经网络，可以直接使用固定大小的高层读数，或自行把 raw contacts 按 region 聚合、采样/填充为点集，或投影成 taxel map。

### 2.6 Isaac Lab ContactSensor

Isaac Lab ContactSensor 属于：

```text
Contact / Force Tactile
└── Batched rigid-body contact-force tactile
    └── RL-oriented tensor observation
```

它是建立在 Isaac Sim / PhysX 接触信息之上的张量化接口。配置中的 prim path 可以匹配每个并行环境里的一个或多个 rigid-body prim。这里的基本 region 通常是整个被匹配刚体的碰撞表面；如果希望一个指腹具有多个独立区域，就需要把这些区域建成可分别识别的刚体/触觉 prim，或在 raw contact 后处理中进行空间划分。

若每个环境匹配 $B$ 个刚体，主要输出为世界坐标系中的净接触力：

$$
\mathbf F^{W}_{net}\in\mathbb R^{N_{env}\times B\times3}
$$

其中每个向量：

$$
\mathbf F^{W}_{net,b}=\sum_{i\in\mathcal C_b}\mathbf F_i^W
$$

表示刚体 $b$ 当前所有相关接触力的向量和。对最后一维取模可形成固定大小的区域强度：

$$
s_b=\left\lVert\mathbf F^{W}_{net,b}\right\rVert_2,\qquad
\mathbf s\in\mathbb R^{N_{env}\times B}
$$

启用对象过滤时，还可得到按被监测 body 与过滤对象组织的 force matrix：

$$
\mathbf F^W_{matrix}\in\mathbb R^{N_{env}\times B\times F\times3}
$$

其中 $F$ 是过滤对象数量；具体支持的匹配关系受接口版本限制。启用历史缓存后，输出还会增加长度为 $H$ 的时间维。

数据链路为：

```text
matched rigid-body prims in many environments
        ↓
PhysX contact forces
        ↓
按 body（可再按 filter）聚合
        ↓
batched [N_env, B, 3] tensors
        ↓
直接输入策略，或取模/变换到局部坐标系
```

与 ManiSkill3 Allegro Touch 的主要区别不是底层物理完全不同，而是抽象层不同：ManiSkill 已经为 Allegro 手预定义了 16 个 FSR 区域并给出 16 维触觉 observation；Isaac Lab 提供通用的、批量化的刚体净接触力接口，区域布局和最终 observation 通常由用户根据机器人结构定义。

#### Isaac Sim 与 Isaac Lab 的关系

Isaac Sim 是负责场景、机器人、碰撞、物理求解、渲染和底层传感器的通用仿真平台；Isaac Lab 是建立在 Isaac Sim 之上的机器人学习框架，进一步提供并行环境、GPU tensor、observation/action、reward、termination 和训练工作流。

两种 Contact Sensor 更准确的关系是：它们共享 Isaac Sim/PhysX 的底层 contact reporting 能力，但 Isaac Lab 通常不是先创建一个 Isaac Sim Experimental Contact Sensor，再处理它的 `force` 或 raw-contact 输出。

```text
                 physics backend / contact reporting
                              │
             ┌────────────────┴────────────────┐
             ↓                                 ↓
Isaac Sim Experimental Contact        Isaac Lab ContactSensor
场景中的 sensor prim                  RigidContactView / tensor API
摘要 + raw contacts                   批量聚合的固定张量
```

因此可以将两者理解为同一类底层接触信息之上的两条封装路线：

- Isaac Sim Experimental Contact Sensor 面向单个虚拟传感器、场景调试和原始接触读取；
- Isaac Lab ContactSensor 面向大量并行环境和可以直接送入策略网络的固定形状数据。

#### Body-level 聚合造成的信息损失

Isaac Lab 的标准输出回答的是“哪个 rigid body/link 正在接触，以及该 body 的净接触力是多少”，而不是“这个 link 表面的哪个位置正在接触”。假设同一个指尖的前部和后部分别受到相同方向、相同大小的力：

```text
front contact                  back contact
┌────────────────┐            ┌────────────────┐
│             ●  │            │  ●             │
└────────────────┘            └────────────────┘
          \                         /
           └── 可能得到相同的 [Fx, Fy, Fz] ──┘
```

聚合过程为：

$$
\left\{(\mathbf p_i,\mathbf F_{n,i})\right\}_{i=1}^{M}
\longrightarrow
\mathbf F^W_{net,b}=\sum_{i=1}^{M}\mathbf F^W_{n,i}
$$

接触位置 $\mathbf p_i$ 在求和后被丢弃，所以多个不同的接触分布可以对应同一个净力向量。`ContactSensorData.pos_w` 表示 sensor/body origin 的世界坐标位置，并不是 contact position。

还需要注意，Isaac Lab 文档将 `net_forces_w` 定义为世界坐标系中的**净法向接触力**：三维分量是法向力向量在世界坐标系 $x,y,z$ 上的表达，不应自动解释为“法向力 + 两个切向摩擦力”。

如果需要区分指尖前部、后部或更细的空间位置，需要提高感知区域的分辨率：

1. 将表面设计成多个可以分别报告的 tactile bodies/regions；
2. 读取更底层的 contact position，并在指尖局部坐标系中做 region aggregation；
3. 将接触位置投影到表面网格，构建 taxel map。

因此，Isaac Lab 的主要取舍可以概括为：

```text
丢弃 link 内的详细接触位置
              ↓
获得低维、固定形状、GPU batched observation
              ↓
适合大规模 RL
```

### 2.7 MuJoCo Touch Sensor

MuJoCo 传统 Touch Sensor 属于：

```text
Contact / Force Tactile
└── Site-volume normal-force tactile
    └── one scalar per touch site
```

传感区域由一个 `site` 的空间体积定义。MuJoCo 查找涉及该 site 所属 body、且接触点落入 site 体积内的接触，然后聚合其法向接触力。site 在这里主要是感知和空间选择区域，不必是一个独立 collision body。

对第 $r$ 个 touch site，其读数可写成：

$$
s_r=\sum_{i\in\mathcal C_r}\max(0,F_{n,i})
$$

若定义 $R$ 个 touch sensors，单个仿真实例得到：

$$
\mathbf s\in\mathbb R^R
$$

外部并行化 $N_{env}$ 个环境后，可以组织为：

$$
\mathbf S\in\mathbb R^{N_{env}\times R}
$$

数据链路为：

```text
contacts involving the site's body
        ↓
contact point lies inside site volume?
        ↓
sum normal contact forces
        ↓
one nonnegative scalar per site
```

这个标量不是三维合力，也不包含切向摩擦力方向；它更像一个理想化的局部压力/触碰强度读数。多个小 site 可以组成稀疏区域阵列，但一个大 site 的单一读数仍不会提供区域内部的压力分布。

site 是附着在 body 上的虚拟感知体积，通常不参与碰撞。真正产生接触的是 `geom`。与 Isaac Lab 默认按整个 rigid body 聚合相比，MuJoCo 可以在同一个 body 上布置多个局部 sites：

```text
one fingertip body
┌──────────┬──────────┐
│ front    │ back     │
│ site     │ site     │
└──────────┴──────────┘
       ↓          ↓
  s_front      s_back
```

因此，定义 $R$ 个 sites 后可以自然形成 region-level tactile：

$$
\mathbf s=[s_1,\ldots,s_R]\in\mathbb R^R
$$

若 sites 按 $H\times W$ 规则排列，还可将其 reshape 成低分辨率 taxel map。不过相邻 site 若重叠，同一个接触可能被多个传感器同时计入；若存在间隙，又可能漏掉边界接触，因此 site 的覆盖、深度和曲面排布需要专门设计。

#### 传统 Touch Sensor 只保留法向力

传统 `<touch>` 的结论需要明确限定：

```text
MuJoCo contact solver
├── normal contact constraints
└── tangential/friction constraints
              ↓
       traditional <touch>
              ↓
只输出 site 内 normal-force scalars 的总和
```

也就是说，MuJoCo 底层会求解摩擦，但传统 Touch Sensor 不输出切向摩擦力。它不能从单一读数恢复：

- 接触法向方向；
- 两个切向摩擦力分量；
- 接触位置和接触点数量；
- 区域内部的压力分布。

这里还应把 `<touch>` 与 MuJoCo 的 `<force>` sensor 区分开：后者输出 child body 与 parent body 之间的三轴相互作用力，描述结构传力，不等价于某个表面 site 内的局部触觉。

#### 新版 Contact Sensor

MuJoCo 新版通用 `<contact>` sensor 用于把动力学管线中原本变长的 contacts 转换成固定大小数组。它可按 geom、body、subtree 或 site 匹配接触，从每条记录中选择：

```text
found, force, torque, dist, pos, normal, tangent
```

若最多保留 $K$ 条接触，每条选择的字段共占 $D$ 维，则输出为：

$$
\mathbf C\in\mathbb R^{K\times D}
$$

不足 $K$ 条的部分使用空槽；超过时可通过 `none`、`mindist`、`maxforce` 或 `netforce` 等 reduction 规则选择或合并。它比传统 Touch Sensor 保留更多信息，也更方便获取法向和切向接触力、位置及接触坐标系，同时解决变长 contact list 不便直接输入神经网络的问题。

如果需要完全自行控制处理流程，也可遍历 `data.contact` 并调用 `mj_contactForce`。对每条接触可以读取局部接触 wrench，再结合 contact frame 和 position 转换坐标、筛选对象并聚合成自定义 regions 或 taxels。

#### 新版 Tactile Sensor

MuJoCo 新版 `<tactile>` sensor 是另一种空间化触觉接口。它在与传感器关联的 mesh 顶点上采样：

$$
\mathbf T_v=[p_v,v_{t1,v},v_{t2,v}]
$$

若 mesh 有 $V$ 个顶点，输出为：

$$
\mathbf T\in\mathbb R^{V\times3}
$$

其中：

- $p_v$ 是由 penetration depth 映射得到的 penetration-pressure proxy；
- $v_{t1,v}$ 和 $v_{t2,v}$ 是两个切向方向的相对滑动速度。

因此，它可以表示接触位置、接触形状和局部滑动。如果 mesh 顶点具有 $H\times W$ 规则拓扑，可以 reshape 为：

$$
\mathbf T\in\mathbb R^{H\times W\times3}
$$

但这三个通道不是 RGB，也不是 $[F_n,F_{t1},F_{t2}]$：第一个通道并非严格意义上的局部力/面积，后两个通道是滑动速度而不是摩擦力。对于弯曲的非规则三角 mesh，更准确的称呼是 `per-vertex tactile field`，需要 UV 参数化或插值才能显示成规则二维 map。此外，该接口当前只接受传感 geom 与 SDF geoms 接触产生的贡献，适用范围受建模条件限制。

#### MuJoCo 中常见的使用层级

现有 MuJoCo 项目往往按需求选择以下层级，而不是一律使用新版 Tactile Sensor：

| 需求 | 常用数据来源 | 输出形式 |
|---|---|---|
| 接触判断、reward、termination | `data.contact` + geom pair/阈值 | bool/contact flags |
| 某个 body 的总体外部作用 | `cfrc_ext` 等 body-level 数据 | 6D body wrench |
| 每个指尖或区域的接触强度 | 传统 `<touch>` sites | $R$ 个法向力标量 |
| 位置、摩擦和完整接触细节 | `data.contact` + `mj_contactForce` | 变长 contact records |
| 定长的详细接触 observation | 新版 `<contact>` | $K\times D$ slots |
| 空间压力代理量与滑动场 | 新版 `<tactile>` | $V\times3$ per-vertex field |

对灵巧手触觉研究，一个渐进式路线是：先用少量 touch sites 建立快速 region-level baseline；需要切向摩擦和位置时改用 raw contacts 或新版 Contact Sensor；只有在明确需要空间分布时，再使用自定义 taxel mapping 或新版 Tactile Sensor。

### 2.8 Gazebo Contact Sensor

Gazebo Contact Sensor 属于：

```text
Contact / Force Tactile
└── Collision-bound contact reporting
    └── variable-length contact messages
```

传感器在 SDF 中绑定到一个 collision element。物理引擎为该 collision 计算接触后，传感器通过 Gazebo Transport 发布 contact message。它更像一个接触数据出口，而不是已经整理好的神经网络 observation。

若当前更新周期产生 $M$ 条接触记录，可抽象为：

$$
\mathbf P,\mathbf N\in\mathbb R^{M\times3},\qquad
\mathbf d\in\mathbb R^M
$$

并可能包含每条接触对应的 force/torque wrench：

$$
\mathbf W\in\mathbb R^{M\times6}
$$

其中 $M$ 随时间变化；消息通常还包含两个 collision 的标识。具体字段与数值可用性会受到 Gazebo 版本和所选 physics backend 的影响。

数据链路为：

```text
SDF collision element
        ↓
physics backend generates contacts
        ↓
Gazebo Contact Sensor
        ↓
variable-length Contacts message
        ↓
user-defined aggregation / encoding
```

为了输入固定结构的神经网络，通常需要自行定义 $R$ 个区域，并将消息转换为以下一种形式：

$$
\mathbf s\in\mathbb R^R
\quad\text{或}\quad
\mathbf F\in\mathbb R^{R\times3}
\quad\text{或}\quad
\mathbf W\in\mathbb R^{R\times6}
$$

因此，Gazebo Contact Sensor 默认并没有统一的 tactile tensor。它与 Isaac Sim raw contact 类似，保留了较丰富但变长的接触记录；与 ManiSkill Allegro Touch 相比，则缺少预定义的触觉区域布局和最终 16 维 observation。

### 2.9 Contact / Force Tactile 的整体优势与局限

优势：

- 输出维度低；
- 计算开销小；
- 容易进行 GPU 批量仿真；
- 容易与关节状态拼接；
- 适合作为策略的接触状态输入；
- 物理含义相对清晰。

局限：

- 区域内部空间信息丢失；
- 标量输出通常丢失方向；
- 接触点数量不等于接触面积；
- 刚体接触参数会影响信号；
- 不能表达完整接触形状和柔性形变。

## 3. Taxel / Pressure-map Tactile

### 3.1 核心思想

Taxel 方法不再把整个指腹或触觉区域压缩成一个数，而是将传感表面离散为空间网格：

```text
sensor surface
┌─────────────────┐
│ t11 t12 t13 t14 │
│ t21 t22 t23 t24 │
│ t31 t32 t33 t34 │
└─────────────────┘
```

每个 taxel 表示一个局部表面单元。输出可以是压力标量：

$$
P\in\mathbb R^{H\times W}
$$

也可以是三维力分布：

$$
T\in\mathbb R^{H\times W\times3}
$$

其中：

$$
T_{ij}=[F_n,F_{t,1},F_{t,2}]
$$

这里的三个通道是法向力和两个切向力分量，不是 RGB。

### 3.2 Taxel 输出究竟表示什么？

`taxel map` 只说明数据具有空间索引，不保证每个通道都是严格物理意义上的力。常见输出包括：

| 输出 | 典型维度 | 物理含义 |
|---|---:|---|
| binary contact map | $H\times W$ | 每个单元是否接触 |
| normal-force map | $H\times W$ | 每个单元的法向力或法向力代理量 |
| contact-force map | $H\times W\times3$ | 法向与两个切向分量 |
| pressure map | $H\times W$ | 单位面积上的法向载荷或压力代理量 |
| depth/indentation map | $H\times W$ | 几何穿透或表面压入深度 |
| slip-velocity map | $H\times W\times2$ | 局部切平面内的相对滑动速度 |

因此必须同时注明：网格拓扑、坐标系、单位、通道语义和聚合时间窗。形状同为 $H\times W\times3$ 的两个传感器，一个可能输出力向量，另一个可能输出 pressure + sliding velocity，它们不能直接视为相同 observation。

### 3.3 四种主要实现机制

#### 路线 A：多个独立小区域

直接在机器人表面创建大量小 collision regions、sites 或可分别识别的 sensor bodies，每个区域输出一个读数：

```text
many small physical/sensing regions
                 ↓
one scalar/vector per region
                 ↓
[R] or [R, 3]
                 ↓ reshape if topology is regular
[H, W] or [H, W, 3]
```

优点是每个 taxel 的物理归属清晰；缺点是大量 collision shapes 或 rigid bodies 会增加模型和接触求解成本，还可能改变原有表面的碰撞行为。MuJoCo 的多个传统 `<touch>` sites 可以形成这种低分辨率阵列；在 Isaac Lab 中则需要让各区域能够被 contact reporter 分别识别，单纯在同一个 body 上增加很多 shapes 不一定会自动得到逐 shape 输出。

#### 路线 B：Raw-contact spatial binning

保留一个正常的 fingertip collision body，从物理引擎读取变长 raw contacts：

```text
position + contact frame + force/impulse
                    ↓
transform to fingertip local frame
                    ↓
project to surface coordinates (u, v)
                    ↓
assign or splat into taxel bins
                    ↓
accumulate normal/shear components
```

设第 $i$ 个接触在指尖局部表面坐标中为 $(u_i,v_i)$，其局部接触力为 $[F_{n,i},F_{t1,i},F_{t2,i}]$，则 taxel $(h,w)$ 可以定义为：

$$
\mathbf T_{hw}
=
\sum_{i=1}^{M}k\big((u_i,v_i),(u_{hw},v_{hw})\big)
\begin{bmatrix}
F_{n,i}\\F_{t1,i}\\F_{t2,i}
\end{bmatrix}
$$

$k(\cdot)$ 可以是硬分箱、双线性 splatting 或高斯核。这条路线不需要真的创建数百个 sensor bodies，taxel 只是 observation builder 中的虚拟网格。它可以建立在 Isaac Sim raw contacts、MuJoCo `data.contact + mj_contactForce` 或 Gazebo contact messages 之上。

#### 路线 C：SDF / penetration sampling

在传感表面预定义 $V$ 个采样点，并查询接触物体的 signed distance field：

```text
tactile surface sample points
              ↓
query object SDF and gradient
              ↓
penetration depth + surface normal
              ↓
penalty / stress mapping
              ↓
per-point normal and shear field
```

典型的法向 penalty model 为：

$$
F_{n,v}=k_n\delta_v+c_n\dot\delta_v
$$

切向量可以使用弹簧—阻尼或速度模型，并由 Coulomb friction 限制：

$$
\lVert\mathbf F_{t,v}\rVert
\leq \mu F_{n,v}
$$

这种方法的输出密度由预定义采样点决定，不受刚体求解器当前只产生几个 contact points 的直接限制，适合 GPU 并行。但它依赖 SDF 质量、penalty 参数和接触对象的预处理，所得 force/stress 可能是传感器模型的代理量，而不是主动力学求解器真正施加的逐点接触力。

#### 路线 D：连续接触面压力场或软体离散

更物理化的路线不是把几个点扩散到网格，而是先计算一个具有面积的 contact patch：

```text
compliant/hydroelastic surfaces
              ↓
contact surface or deformed mesh
              ↓
continuous/discretized pressure field p(x)
              ↓
sample or integrate over taxel areas
```

第 $r$ 个 taxel 的法向载荷可以定义为：

$$
F_{n,r}=\int_{A_r}p(\mathbf x)\,dA
$$

Drake Hydroelastic Contact 是代表性例子：它构造离散 contact surface，并在其上给出连续分片线性的 pressure field，再积分得到合力与力矩。若要模拟具体传感器，还需要把该 contact surface/pressure field 重采样到真实 taxel 布局。进一步使用 FEM、MPM 或粒子法求解 elastomer deformation，则能获得更真实的压力与剪切分布，但计算成本明显更高，并逐渐接近 physics-based elastomer simulation。

### 3.4 代表性仿真器与实现

| 方案 | 底层平台 | Taxel/pressure 实现 | 典型输出 | 关键限制 |
|---|---|---|---|---|
| 自定义 Isaac Sim raw-contact map | Isaac Sim / PhysX | contact positions 投影、分箱或核扩散 | $[N,H,W,C]$ | contact points 稀疏；映射需自行实现 |
| Isaac Lab 多区域 ContactSensor | Isaac Lab / PhysX | 多个可区分 bodies/regions 后张量化 | $[N,R,3]$ | body-level 接口不会自动给出 link 内 map |
| Isaac Lab Contrib TacSL Sensor | Isaac Lab + SDF | mesh tactile points 查询 SDF；penalty normal/shear model | normal $[N,V]$，shear $[N,V,2]$ | 需预计算 SDF 和指定接触对象 |
| MuJoCo 多 `<touch>` sites | MuJoCo | 每个 site 聚合法向接触力 | $[R]$ | 无切向力；site 覆盖需人工设计 |
| MuJoCo `touch_grid` plugin | MuJoCo | 将 contact forces/torques 按球面角度分箱成 taxels | $[C,H,W]$，$C\in[1,6]$ | 受求解器 contact-point 稀疏性限制 |
| MuJoCo `touch_stress` plugin | MuJoCo + SDF | SDF 几何与滑动速度生成高分辨率 stress image | $[C,H,W]$ | stress 绝对值与动力学 contact force 不等价 |
| MuJoCo `<tactile>` | MuJoCo + SDF geom | mesh vertices 上采样 penetration pressure 与切向滑速 | $[V,3]$ | 仅 SDF contacts；不是三维力场 |
| Drake Hydroelastic | Drake | hydroelastic contact surface 上的 pressure field | contact mesh + $p(\mathbf x)$ | 需再次采样到具体 taxel layout |
| TactileSim | 专用可微触觉模拟 | 任意布局上的解析 normal/shear tactile force field | dense force field + gradients | 传感器模型和材料参数需要标定 |
| TacSL | Isaac Gym 系 | GPU force/deformation field，并可继续渲染 tactile RGB | dense force field / RGB | 同时跨 Taxel 与 Vision-based 两类 |
| Gazebo custom pressure plugin | Gazebo | contact messages 分箱或多个 collision regions | 用户定义 | 没有统一的原生 taxel tensor 标准 |

这里需要特别区分“原生接口”和“构建在仿真器上的传感器模型”：Isaac Sim 与 Gazebo 的标准 Contact Sensor 主要提供接触数据来源，本身不会自动生成规则 pressure map；Isaac Lab Contrib TacSL、MuJoCo plugins、TactileSim 等则显式加入了 surface sampling、SDF 或 spatial aggregation。

### 3.5 重点方案细节

#### MuJoCo `touch_grid`

`touch_grid` 与传统 `<touch>` 不同。它把 site parent body 与其他 body 之间的接触力和力矩，按照 site frame 中的球面视场划分到 $H\times W$ angular bins。可选择 1 到 6 个通道，顺序为：

```text
[normal force, tangent-x force, tangent-y force,
 torsional torque, rolling-x torque, rolling-y torque]
```

因此：

$$
\mathbf T\in\mathbb R^{C\times H\times W},\qquad 1\le C\le6
$$

它能直接保留法向与切向接触力，但 map 的空间坐标是相对 site 的角度 bin，不一定等于平面电子皮肤的笛卡尔 taxel layout。更重要的是，一个 bin 中必须先有求解器生成的 contact point 才有信号，因此提高 $H,W$ 不会自动提高底层接触分辨率。

#### MuJoCo `touch_stress` 与原生 `<tactile>`

`touch_stress` 使用 SDF 克服 `touch_grid` 的稀疏性，可以生成更密集的 normal/tangential stress image。但是官方明确说明，其 stress 绝对值基于 SDF 几何和滑动速度，与主求解器的 contact forces 没有直接数值对应关系。

原生 `<tactile>` 同样基于 SDF contact，不过输出落在任意 mesh vertices 上：

$$
\mathbf T_v=[p_v,v_{t1,v},v_{t2,v}]
$$

它适合描述 pressure proxy 与 slip field，而不是直接测量三个方向的力。如果 mesh 非规则，输出是 per-vertex field；只有规则拓扑或经过 UV 重采样后才是二维 map。

#### Isaac Lab Contrib TacSL Sensor

该接口同时支持 camera-based tactile 与 force-field tactile。力场部分从 elastomer mesh 生成固定 tactile points，查询指定接触对象的 SDF，得到 penetration depth，再使用 penalty model 计算法向与剪切力：

$$
\text{depth}\in\mathbb R^{N\times V}
$$

$$
\mathbf F_n\in\mathbb R^{N\times V},\qquad
\mathbf F_t\in\mathbb R^{N\times V\times2}
$$

相较于标准 Isaac Lab ContactSensor 的 $[N,B,3]$，它保留了每个 tactile point 的空间信息；代价是需要 SDF 预计算、传感表面 mesh 和额外的逐点查询。

#### Drake Hydroelastic pressure field

Drake 不把接触简化成少数点，而是近似构造具有面积的接触面和 pressure field：

$$
p:\mathcal S_c\rightarrow\mathbb R_{\ge0}
$$

其中 $\mathcal S_c$ 是离散的 hydroelastic contact surface。这个 pressure field 本身不是某款电子皮肤的 observation，但它为压力型 tactile sensor 提供了更自然的物理数据源：将 $p(\mathbf x)$ 在每个 taxel footprint 上积分，就能得到具体布局的读数。

### 3.6 关键限制

刚体物理引擎生成 contact points 的目的，是求解动力学约束，而不是采样连续压力场。因此：

```text
taxel grid 分辨率很高
            ≠
底层 contact data 分辨率很高
```

如果底层只有少数 contact points，高分辨率 taxel map 仍然会很稀疏。空间核扩散或插值可以生成 pressure-like map，但它引入了额外的 sensor model。SDF sampling 可以得到稠密场，却可能与动力学求解器真正施加的接触力不一致；hydroelastic/soft-body 方法更接近连续压力分布，但更依赖材料参数、网格分辨率和计算预算。

还需要考虑：

- taxel 面积不同，直接比较力值会混入面积因素，必要时应使用 pressure；
- 曲面上的 local normal 和 tangent frame 随 taxel 位置变化；
- 重叠 bins 会重复计数，存在间隙则会漏检；
- 接触力需要统一到 sensor local frame 才便于跨姿态学习；
- 仿真时间步、接触刚度和阻尼都会改变峰值力；
- 真实电子皮肤还存在迟滞、串扰、饱和、漂移、噪声与坏点，理想 map 不会自动包含这些效应。

### 3.7 与 Contact / Force Tactile 的关系

Taxel 方法可以看作 contact data 的空间展开：

```text
Contact / Force Tactile
多个点 → 一个 link/region 读数

Taxel / Pressure-map Tactile
多个点、表面采样或压力场 → 多个空间单元读数
```

两者可以使用同一个 physics backend，区别主要在 observation construction。一个 taxel map 还可以重新池化为 region-level signal：

$$
s_r=\sum_{(h,w)\in A_r}T_{hw}
$$

但反过来不能从一个 region scalar 唯一恢复其内部 taxel distribution。

### 3.8 优势与局限

优势：

- 保留接触在表面上的位置；
- 能表达多点接触和压力分布；
- 输出仍具有明确的物理量语义；
- 可以使用 CNN 等空间模型处理。

局限：

- 需要定义表面坐标和 taxel topology；
- 曲面指腹的二维参数化较复杂；
- 底层 contact points 可能过于稀疏；
- 插值结果依赖人为设定的扩散模型；
- 不包含真实 soft-gel 和光学成像过程。

## 4. Vision-based Tactile

### 4.1 核心思想

Vision-based tactile 模拟内部装有 LED 和 RGB camera 的柔软触觉传感器。外部物体按压 gel，内部相机观察 gel 内表面的形变和反射变化：

```text
external object
       ↓ contact
soft gel surface
       ↓ deformation
surface geometry and markers
       ↓ illumination
internal RGB camera
       ↓
tactile image
```

相机拍摄的不是机器人外部环境，而是触觉传感器内部的 gel 表面。

主要输出为：

$$
I\in\mathbb R^{H\times W\times3}
$$

其中三个通道表示 RGB 光强。部分方案还可以提供：

- gel deformation/depth；
- surface-normal map；
- marker displacement；
- tactile optical flow；
- contact-force distribution。

### 4.2 仿真链路

完整的 vision-based tactile simulation 可能包含：

```text
rigid/soft contact
       ↓
gel deformation model
       ↓
surface normal/depth field
       ↓
marker motion model
       ↓
lighting and reflection model
       ↓
virtual internal camera
       ↓
tactile RGB image
```

不同方案的差异主要在于：gel deformation 是否具有物理真实性、光学模型如何标定，以及能否进行 GPU 并行。

### 4.3 代表方案

| 方案 | 主要技术路线 | 定位 |
|---|---|---|
| TACTO | 刚体接触 + 快速光学渲染 | 经典、简单的视觉触觉模拟 |
| Taxim | 标定驱动的光学 lookup table | 快速模拟 GelSight 光学响应与 marker |
| FOTS | MLP optical mapping + marker approximation | 可插拔的快速光学模拟 |
| Tacchi | Taichi particle elastomer | 低成本 physics-based gel deformation |
| TacSL | GPU visuotactile simulation | 面向大规模策略学习 |
| TacEx | GIPC soft body + Taxim/FOTS | Isaac Sim/Lab 中的高保真 gel 与光学模拟 |
| TactSim-IsaacLab | compliant contact + internal camera/lights | 较轻量的 Isaac Lab 工程方案 |
| DiffTactile | FEM + differentiable contact | 可微分接触、形变和参数优化 |
| Taccel | IPC/ABD GPU physics | 大规模 robot-object-gel 联合仿真 |
| Gazebo OpticalTactilePlugin | contact sensor + depth camera | Gazebo 原生 optical tactile 路线 |

### 4.4 优势与局限

优势：

- 空间分辨率高；
- 能表达接触形状和局部几何；
- marker motion 可以提供剪切与形变线索；
- 可以复用 CNN、ViT 等视觉模型；
- 与 GelSight/DIGIT 类传感器输出形式一致。

局限：

- 仿真链路更长；
- 计算和显存成本更高；
- gel 材料参数和光学参数需要标定；
- soft contact 与 rendering 都可能产生 sim-to-real gap；
- 不同方案对形变真实性的支持差异很大。

## 5. 三类方法的本质区别

三类方法不是三个完全不同的物理世界，而是对同一接触过程进行不同深度的观测编码：

```text
底层接触数据
│
├── 直接区域聚合
│   └── Contact / Force Tactile
│
├── 按表面位置离散
│   └── Taxel / Pressure-map Tactile
│
└── 继续模拟软体形变和光学成像
    └── Vision-based Tactile
```

| 比较维度 | Contact / Force | Taxel / Pressure map | Vision-based |
|---|---|---|---|
| 输出语义 | 接触状态、力、冲量 | 空间力/压力分布 | gel 的光学图像 |
| 空间分辨率 | 低 | 中到高 | 高 |
| 方向信息 | 可选 | 通常可保留 | 需要从图像或 marker 推断 |
| 是否模拟 soft gel | 否 | 通常否 | 通常是 |
| 是否需要 lighting/camera | 否 | 否 | 是 |
| 计算成本 | 低 | 中 | 中到高 |
| observation 网络 | MLP | CNN/空间编码器 | CNN/ViT |
| 典型用途 | 低维接触反馈 | 空间接触表示 | 高分辨率形变表示 |

## 6. Raw Contact Data 是三类方法之间的桥梁

许多物理引擎都可以产生类似的 raw contact record：

```text
body pair
position
normal
impulse or force
penetration/separation
```

假设当前有 $M$ 个 contacts：

$$
X\in\mathbb R^{M\times D}
$$

$M$ 随时间变化，因此不能直接输入普通固定维度 MLP。可以选择：

```text
raw contacts
│
├── 按 region 求和
│   └── [B,R] 或 [B,R,3]
│
├── 投影到表面 grid
│   └── [B,C,H,W]
│
├── Top-K + padding + mask
│   └── [B,K,D]
│
└── set/point-cloud encoder
    └── fixed latent feature
```

所以从 high-level 看：

> Contact Sensor 提供接触数据来源；region、taxel 或更复杂的 tactile representation 是后续的 observation design。

## 7. 选择技术路线时应问什么？

选择方案时，应该先回答策略需要什么信息。

### 只需要接触状态和区域强度

选择 Contact / Force Tactile：

```text
ManiSkill Allegro Touch
Isaac Lab ContactSensor
MuJoCo Touch Sensor
```

### 需要接触在表面上的空间分布

选择 Taxel / Pressure-map：

```text
raw contact projection
surface discretization
normal/shear aggregation
```

### 需要接触形状、gel deformation 或 tactile RGB

选择 Vision-based Tactile：

```text
TacSL / TacEx / Taccel
DiffTactile / Tacchi
TACTO / Taxim / FOTS
```

选择不应只看“哪个方案更真实”，而应同时考虑：

- 策略真正需要的信息；
- 并行环境数量；
- physics timestep 与 observation frequency；
- 是否需要 sim-to-real；
- 是否有真实传感器标定数据；
- 开发成本与计算资源。

## 8. 当前阶段的研究定位

当前可以先建立以下研究顺序：

```text
Stage 1
理解 Contact / Force Tactile
ManiSkill Allegro Touch + Isaac contact APIs
        ↓
Stage 2
研究 raw contacts 如何构造 region/taxel observation
        ↓
Stage 3
比较 Vision-based simulators 的 gel、optics 与 GPU 路线
```

当前阶段的核心结论是：

1. ManiSkill3 Allegro Touch 属于 Contact / Force Tactile，而不是视觉触觉仿真。
2. Isaac Sim raw contact data 可以由用户编码为 region signals 或 taxel maps。
3. Taxel map 与 vision-based tactile 即使 shape 相似，通道语义仍完全不同。
4. Vision-based tactile 需要在 contact physics 之上继续模拟 gel deformation、lighting 和 internal camera。
5. 三类方法的选择取决于策略所需的信息层级，而不是单纯追求最高维度的 observation。
6. Isaac Lab 并不是简单调用 Isaac Sim Experimental Contact Sensor 的高层输出；两者共享底层 contact reporting，但分别面向单传感器读取和批量 RL tensor。
7. Isaac Lab 的 body-level `net_forces_w` 无法区分同一 link 的前部、后部等接触位置；这种空间信息在按 body 聚合时已经丢失。
8. MuJoCo 传统 `<touch>` 是 site-level 法向力标量，不包含切向摩擦力；摩擦信息需要通过新版 `<contact>` 或底层 `mj_contactForce` 获取。
9. MuJoCo 新版 `<tactile>` 输出的是 per-vertex penetration-pressure proxy 和切向滑动速度，不是严格的三维力分布，也不是 RGB tactile image。
10. 面向 RL，固定低维 region signal 通常是更直接的 baseline；只有任务确实依赖接触位置、形状或滑动分布时，才需要提高到 contact slots、taxel field 或 vision-based tactile。
11. Taxel map 是一种空间数据组织方式，不保证通道一定是真实力；contact-force binning、SDF stress、penetration pressure 和 hydroelastic pressure 的数值语义不同。
12. 高分辨率网格不等于高分辨率物理接触：raw-contact binning 受 contact-point 数量限制，SDF sampling 更稠密但属于额外传感器模型。
13. MuJoCo `touch_grid` 可以输出分箱后的法向、切向力及力矩；`touch_stress` 和原生 `<tactile>` 则用 SDF 获得稠密场，但数值不应直接当作求解器逐 taxel 接触力。
14. Drake Hydroelastic 的 contact-surface pressure field 是构建压力型 tactile sensor 的良好数据源，但仍需按具体硬件 taxel footprint 做重采样或面积积分。

## 9. 参考资料

- [ManiSkill Allegro Hand Right with Touch Sensing](https://maniskill.readthedocs.io/en/latest/robots/allegro_hand_right_touch/index.html)
- [ManiSkill Allegro Touch source](https://github.com/mani-skill/ManiSkill/blob/main/mani_skill/agents/robots/allegro_hand/allegro_touch.py)
- [Isaac Sim Experimental Contact Sensor](https://docs.isaacsim.omniverse.nvidia.com/6.0.1/sensors/isaacsim_sensors_physics_contact.html)
- [Isaac Lab ContactSensor](https://isaac-sim.github.io/IsaacLab/main/source/overview/core-concepts/sensors/contact_sensor.html)
- [MuJoCo Traditional Touch Sensor](https://mujoco.readthedocs.io/en/3.6.0/XMLreference.html#sensor-touch)
- [MuJoCo Fixed-size Contact Sensor](https://mujoco.readthedocs.io/en/3.6.0/XMLreference.html#sensor-contact)
- [MuJoCo Mesh-based Tactile Sensor](https://mujoco.readthedocs.io/en/3.6.0/XMLreference.html#sensor-tactile)
- [MuJoCo Touch Grid and Touch Stress Plugins](https://github.com/google-deepmind/mujoco/tree/main/plugin/sensor)
- [Isaac Lab Contrib TacSL Sensor Data](https://isaac-sim.github.io/IsaacLab/main/_modules/isaaclab_contrib/sensors/tacsl_sensor/visuotactile_sensor_data.html)
- [Drake Hydroelastic Contact User Guide](https://drake.mit.edu/doxygen_cxx/group__hydroelastic__user__guide.html)
- [TactileSim: Efficient Tactile Simulation with Differentiability](https://tactilesim.csail.mit.edu/)
- [Gazebo Contact Message](https://gazebosim.org/api/msgs/9/classgz_1_1msgs_1_1Contact.html)
- [TACTO](https://github.com/facebookresearch/tacto)
- [Taxim](https://arxiv.org/abs/2109.04027)
- [FOTS](https://github.com/Rancho-zhao/FOTS)
- [Tacchi](https://arxiv.org/abs/2301.08343)
- [TacSL](https://iakinola23.github.io/tacsl/)
- [TacEx](https://arxiv.org/abs/2411.04776)
- [DiffTactile](https://difftactile.github.io/)
- [Taccel](https://taccel-simulator.github.io/)
