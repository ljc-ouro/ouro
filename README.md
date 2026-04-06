<div align="center">

![logo](./images/logo.png)

</div>

<div align="center">

![visitors](https://visitor-badge.laobi.icu/badge?page_id=ljc-ouro/ouro)
[![GitHub Repo stars](https://img.shields.io/github/stars/ljc-ouro/ouro?style=social)](https://github.com/ljc-ouro/ljc-ouro/stargazers)
[![GitHub Code License](https://img.shields.io/github/license/ljc-ouro/ouro)](LICENSE)
[![GitHub last commit](https://img.shields.io/github/last-commit/ljc-ouro/ouro)](https://github.com/ljc-ouro/ljc-ouro/commits/master)
[![GitHub pull request](https://img.shields.io/badge/PRs-welcome-blue)](https://github.com/ljc-ouro/ouro/pulls)
[![Collection](https://img.shields.io/badge/🤖-Gridman%20%20Collection-blue)](https://hf.co/collections/ljc-ouro/gridman)

</div>

<div align="center">
  <h3>"我来从无聊的世界中拯救你了"</h3>
</div>

<div align="center">

中文 | [English](./README_en.md)

</div>

* 此开源项目旨在完全从 0 开始, 构建第一代带状态 AI 架构 `Ouro`, 并以全新字节级语言模型 `Gridman` 作为体验开端.

* 仅用 3 块钱成本与 2 小时训练时间，即可训练出规模约为 52M 全新架构的超小语言模型 `Gridman-Mini`.
* `Gridman` 系列从极轻量模型到 B 级别模型全线覆盖，主线版本体积基本和 GPT-2 系列规模相当, Mini 版力求让普通个人 GPU 也能快速完成训练与复现.
* 项目同时开源了完整训练链路，覆盖预训练 (Pretrain), 监督微调 (SFT) 等全过程代码.
* 项目所有核心算法代码均从 0 使用 PyTorch 原生实现, 不依赖第三方库提供的高层抽象接口.
* 这不仅是一个全新架构的大语言模型全阶段开源项目，也是一套面向 `Ouro` 入门与实践的教程.
* 希望此项目能为更多人提供一个可复现, 可理解, 可扩展 `Ouro` 的起点, 一起感受状态 AI 模型的魅力, 并推动更广泛 AI 社区的进步, 为未来世界的变革做好准备.
* 项目交流 QQ 群: 198302483. 答案: State.

> 注：本项目基于 Apache 2.0 协议开源; "2小时" 基于 NVIDIA 3090 硬件设备 (单卡) 预估."3块钱" 指 GPU 服务器租用成本, 具体规格详情见下文.

---

# 📌 架构优势

<table style="width: 100%; border-collapse: separate; border-spacing: 12px; background: transparent; border: none; table-layout: fixed;">
  <tr>
    <td style="background: linear-gradient(145deg,#1a1a1a,#111); border-radius: 16px; padding: 22px; border: 1px solid rgba(255,255,255,0.1); vertical-align: top;">
      <div style="color:rgba(255,255,255,0.7); font-size:13px; margin-bottom:10px;">⚙️ Theory</div>
      <div style="color:#fff; font-size:19px; font-weight:600; margin-bottom:6px;">理论完备</div>
      <div style="color:rgba(255,255,255,0.5); font-size:12px;">Ouro complete</div>
    </td>
    <td style="background: linear-gradient(145deg,#1a1a1a,#111); border-radius: 16px; padding: 22px; border: 1px solid rgba(255,255,255,0.1); vertical-align: top;">
      <div style="color:rgba(255,255,255,0.7); font-size:13px; margin-bottom:10px;">⚡ Speed</div>
      <div style="color:#fff; font-size:19px; font-weight:600; margin-bottom:6px;">恒定速度</div>
      <div style="color:rgba(255,255,255,0.5); font-size:12px;">Constant generation speed</div>
    </td>
    <td style="background: linear-gradient(145deg,#1a1a1a,#111); border-radius: 16px; padding: 22px; border: 1px solid rgba(255,255,255,0.1); vertical-align: top;">
      <div style="color:rgba(255,255,255,0.7); font-size:13px; margin-bottom:10px;">💾 VRAM</div>
      <div style="color:#fff; font-size:19px; font-weight:600; margin-bottom:6px;">恒定显存</div>
      <div style="color:rgba(255,255,255,0.5); font-size:12px;">Constant VRAM</div>
    </td>
  </tr>
  <tr>
    <td style="background: linear-gradient(145deg,#1a1a1a,#111); border-radius: 16px; padding: 22px; border: 1px solid rgba(255,255,255,0.1); vertical-align: top;">
      <div style="color:rgba(255,255,255,0.7); font-size:13px; margin-bottom:10px;">📦 No Cache</div>
      <div style="color:#fff; font-size:19px; font-weight:600; margin-bottom:6px;">无需 KV Cache</div>
      <div style="color:rgba(255,255,255,0.5); font-size:12px;">Without KV Cache</div>
    </td>
    <td style="background: linear-gradient(145deg,#1a1a1a,#111); border-radius: 16px; padding: 22px; border: 1px solid rgba(255,255,255,0.1); vertical-align: top;">
      <div style="color:rgba(255,255,255,0.7); font-size:13px; margin-bottom:10px;">✨ Learning</div>
      <div style="color:#fff; font-size:19px; font-weight:600; margin-bottom:6px;">持续学习</div>
      <div style="color:rgba(255,255,255,0.5); font-size:12px;">Continual learning</div>
    </td>
    <td style="background: linear-gradient(145deg,#1a1a1a,#111); border-radius: 16px; padding: 22px; border: 1px solid rgba(255,255,255,0.1); vertical-align: top;">
      <div style="color:rgba(255,255,255,0.7); font-size:13px; margin-bottom:10px;">∞ Context</div>
      <div style="color:#fff; font-size:19px; font-weight:600; margin-bottom:6px;">无限上下文</div>
      <div style="color:rgba(255,255,255,0.5); font-size:12px;">Infinite ctxlen</div>
    </td>
  </tr>
</table>

---

# 📌 项目介绍

注意力机制以及 `Transformer` 架构的出现, 拉开了了大语言模型和全民 AI 时代的序幕. 从 2022 年 GPT-3.5 第一次震惊世界开始, 时间连带着模型尺寸飞速增长, 整个 AI 世界在朝前狂奔. 但站在真正起作用的底层架构的视角上回顾, 我们似乎一直在原地踏步. 

那么问题是什么? 不存在更优的架构了吗? 并非如此, 我们犯了范式层级的错误: AI 模型正在逐步退化为一种事实上的极大规模函数. 输入被映射为输出, 人类只需要为这个静态怪物不断增加参数.

该项目尝试对这一默认前提进行一次彻底的反转. 不再将 AI 视为一个输入驱动的函数近似器, 而是将其构建为一个围绕内部 State 持续运行的系统. 在这一视角下: 

- State 不是缓存
- 不是附属变量
- 也不是 prompt 或上下文的延伸或压缩

相反, State 是模型的核心主体. 

这正是 `Ouro` 构建的核心哲学: **State is all you need**.

😊 一起感受状态模型的乐趣吧！

---

#### 🎉 本项目包含以下内容

- 提供完整的理论框架, 给出数学上 AGI 必备的完备性理论.
- 提供完整的 `Ouro` 结构代码，开启全新架构生态.
- 提供完整的 `Gridman` 语言模型训练代码, 预训练/微调权重同时开源.
- 提供 `GridmanByteTokenizer` 无需任何先验分词器, 支持自定义模板标记扩展.
- 覆盖 Pretrain, SFT 完整训练流程.
- 提供全阶段开源数据，覆盖收集, 蒸馏, 清洗与去重后的高质量数据集.
- 提供原生 `GridmanDataLoader` 数据加载器, 保证数据流贴合架构特性. 
- 关键训练算法与核心模块均从 0 实现, 不依赖第三方框架封装.

#### 🎉 已 (预) 发布架构/模型列表

<details> 
<summary> <b>🔥 Ouro-Naxi</b> </summary>

- 

`Ouro` 架构 `v1` 版本命名为 `Naxi`, 源自中国地名纳溪, 取纳溪成川之意. 后统一用 `-Naxi` 指代 `v1` 架构版本及对应的 `Gridman` 模型版本.

使用 `Ouro-Naxi` 架构训练的原生字节级语言模型 `Gridman` 模型列表:

| 模型 | 参数量 | 嵌入维度 | Blocks | Layers | Release |
|------|--------|--------|------|------|---------|
| Gridman-Naxi-Mini | 52.31 (50.74 + 1.57) M | 512 | 2 | 6 | 2026.04.01 |
| Gridman-Naxi-Small | 150.47 (145.75 + 4.72) M | 768 | 2 | 8 | 2026.04.01 |
| Gridman-Naxi-Medium | 396.31 (383.73 + 12.58) M | 1024 | 3 | 8 | 2026.04.01 |
| Gridman-Naxi-Large | 866.51 (840.30 + 26.21) M | 1280 | 4 | 9 | 2026.04.01 |
| Gridman-Naxi-XL (即将发布) | 1712.10 (1660.90 + 51.20) M | 1600  | 4 | 12 | 2026.04.01 |

</details>

> 注：模型参数组成为 `可训练参数大小` + `状态参数大小`, 其中可训练参数被定义为训练时通过反向传播更新的参数; 模型名称后无显式标注 "即将发布" 的均已发布.

---

# 📌 架构理论

#### 💡 无状态模型

为什么我们需要一个状态? 在传统的 RNN 模型中, 状态转移方程通常被如下描述:

$$s_{t+1}, y = f(s_t, x)$$

$x$ 是输入, $y$ 是目标值, 这已经是强约束. 但是对于状态 $s_t$, 这里不存在的一个显式的约束. 这既为架构的设计提供了自由度, 也带来一种冗余性的暗示: $s_t$ 可能是完全多余的. 如果你遵照这种谕示将 $s_t$ 替换为历史输入的一种展平, 那么恭喜你, 你发明了 `Transformer`.

显然, 按照这么理解, `Transformer` 是一种标准的无状态模型.

#### 💡 基于概率的状态转移模型

哦, 等等, `Transformer` 真的抛弃了 $s_t$ 吗? 非也. 如果将上下文看作状态, 那 `Transformer` 不就描述了一个标准的状态转移

$$s_{t+1} = T(s_t)$$

吗?

这里其实存在着微妙的区别. 即使我们将上下文看作这里的状态, $T(s_t)$ 实际上也是给出了下一个状态的分布而非具体的状态. 这实际上是一种基于概率的状态转移模型.

是的, 状态在这里依然存在, 只是从模型内部打包到了外部. 此时状态转移的约束完全来自外部约束.

#### 💡 约束状态的隐变量

`Transformer` 是一种对于状态转移过度简化的模型, 让我们回顾标准的状态转移方程

$$s_{t+1}, y = f(s_t, x)$$

除了 $(x, y)$ 带来的外部约束, 模型自身对 $s_t$ 的约束到底是什么? 这强烈暗示我们这里存在一个隐变量 $\theta$, 仔细一想, 这正是权重的含义.

我们重写标准的状态转移方程

$$s_{t+1}, y = F(s_t, x, G(\theta, s_t))$$

称之为 Ouro 型状态转移方程, $F$ 由 $\theta, s_t$ 确定的权重 $G(\theta, s_t)$ 唯一决定. 现在距离我们得出最终的状态约束只有一步之遥了.

#### 💡 等效原理

为了得到我们想要的约束, 这里必须做出一个深刻的假设: 一个足够好的系统, 其推理 (前向传播) 和学习 (反向传播) 在局部不可区分. 这个假设称之为等效原理.

- 推理: $s_t$ 的改变

- 学习: $G(\theta, s_t)$ 的改变

基于等效原理, 在这里做一些简单的推导.

我们通过反向传播来更新权重, 即更新 $G(\theta, s_t)$. 那么在一次反向传播后权重变为 $G(\theta + \mathrm{d}\theta, s_t + \mathrm{d}s)$. 当模型收敛时展开这个式子得到

$$G(\theta + \mathrm{d}\theta, s_t + \mathrm{d}s)=G(\theta', s_t)+\frac{\partial G}{\partial s}(\theta', s_t)\mathrm{d}s$$

由于等效原理和递推方程我们自然的要求 $s_{t+1} = s_t + \mathrm{d}s$, 带入得到

$$G(\theta', s_{t+1}) + s_t\frac{\partial G}{\partial s}(\theta', s_t)=G(\theta', s_t)+ s_{t+1}\frac{\partial G}{\partial s}(\theta', s_t)$$

令 $J_{t}=\frac{\partial G}{\partial s}(\theta', s_t)$, 重写为

$$G(\theta', s_{t+1})-G(\theta', s_{t})=J_t (s_{t+1} - s_{t})$$

实际上这就是我们需要的约束!

也可以直接写作连续形式

$$\mathrm{d}G=J\mathrm{d}s$$

这告诉我们学习-推理的局部不可区分性本质上来自于链式法则.

#### 💡 Ouro 完备

设数据域：
$$
\mathcal{D} \subseteq \mathcal{X} \times \mathcal{Y}
$$

Ouro 型状态转移方程定义为：
$$
(s_{t+1}, y_t) = F\big(s_t, x_t, G(\theta, s_t)\big), 
\quad (x_t, y_t') \sim \mathcal{D}
$$

定义总损失：

$$L(\theta) = L_1(\theta) + \lambda L_2(\theta)$$

$L_1$ 任务损失

$$
L_1(\theta)
= \mathbb{E}_{(x,y') \sim \mathcal{D}}
\big[ \ell(y_t, y') \big]
$$

$L_2$ 状态约束损失

$$
L_2(\theta)
= \mathbb{E}_{(x_t,y_t') \sim \mathcal{D}}
\left[
\left\|
G(\theta, s_{t+1}) - G(\theta, s_t)
- J_t (s_{t+1} - s_t)
\right\|^2
\right]
$$

其中：

- $s_t \in \mathcal{S}$ 为状态
- $\theta \in \Theta$ 为参数
- $G : \Theta \times \mathcal{S} \to \mathcal{W}$
- $J_t = \frac{\partial G}{\partial s}(\theta, s_t)$

并满足：

$$
\left\{
\begin{aligned}
&\lim_{t \to \infty} |\nabla_\theta L(\theta_t)| = 0 \\
&\lim_{t \to \infty} |L_2| = 0 \\
&\lim_{t \to \infty} \theta_t = \theta' \\
&\sup_t |s_t| < \infty \\
&G \in C^1(\Theta \times \mathcal{S})
\end{aligned}
\right.
$$

则称 $F$ 在 $\mathcal{D}$ 上是 **Ouro 完备的**.

#### 💡 AGI

若 $F$ 同时满足 Ouro 完备与图灵完备, 则称 $F$ 是 AGI (Artificial General Intelligence).

---

# 📌 模型

---