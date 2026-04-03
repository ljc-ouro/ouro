<div align="center">

![logo](./images/logo.png)

</div>

<div align="center">

![visitors](https://visitor-badge.laobi.icu/badge?page_id=ljc-ouro/ouro)
[![GitHub Repo stars](https://img.shields.io/github/stars/ljc-ouro/ouro?style=social)](https://github.com/ljc-ouro/ljc-ouro/stargazers)
[![GitHub Code License](https://img.shields.io/github/license/ljc-ouro/ouro)](LICENSE)
[![GitHub last commit](https://img.shields.io/github/last-commit/ljc-ouro/ouro)](https://github.com/ljc-ouro/ljc-ouro/commits/master)
[![GitHub pull request](https://img.shields.io/badge/PRs-welcome-blue)](https://github.com/ljc-ouro/ouro/pulls)
[![Collection](https://img.shields.io/badge/🤖-Gridman%20%20Collection-blue)](https://huggingface.co/collections/ljc-ouro/gridman)

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

> 注：本项目基于 Apache 2.0 协议开源; "2小时" 基于 NVIDIA 3090 硬件设备 (单卡) 预估."3块钱" 指 GPU 服务器租用成本, 具体规格详情见下文.

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

#### 📝 更新日志

<details> 
<summary> <b>🔥 2026-04-01</b> </summary>

- 项目正式开源
- 发布 `Ouro-Naxi` / `Gridman-Naxi`：结构、Tokenizer、训练链路、推理接口与默认配置全面更新
- 结构图资源更新，README 大面积更新

</details>

---

