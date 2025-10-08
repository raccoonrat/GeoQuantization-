

# **大语言模型量化中的离群点二元性：调和精度、隐私与几何完整性的科学探究**

## **执行摘要**

大语言模型（LLM）的量化压缩技术是实现其在资源受限环境中高效部署的关键。然而，在这一领域的核心地带，存在着一个深刻的科学难题，即“离群点双重矛盾”与失控的“精度-隐私权衡”。本报告旨在对这一难题进行系统性的科学探究。离群点，即模型参数或激活值中的极端数值，构成了这一矛盾的核心。一方面，它们是导致量化精度损失的主要瓶颈，其巨大的动态范围会引发灾难性的舍入误差。另一方面，它们是模型记忆训练数据的关键载体，编码了包括个人可识别信息（PII）和专有知识在内的敏感内容，从而构成严重的隐私泄露风险。

本报告首先剖析了离群点的架构起源，揭示了它们并非随机噪声，而是 Transformer 架构（特别是 Softmax 注意力机制和层归一化）与标准训练动态相互作用下的系统性产物。这一发现重塑了我们将离群点视为一种需要被理解和管理的模型内在机制，而非简单需要被消除的“缺陷”的认知。

在此基础上，报告深入探讨了离群点的“双重角色冲突”。通过综合模型剪枝研究和隐私攻击分析，我们证实了离群点对于维持模型核心功能（如执行“无操作”注意力）的必要性，同时也揭示了它们作为高保真信息载体，在记忆和泄露敏感数据方面所扮演的关键角色。对代码大语言模型的实证研究首次量化了这一冲突，表明量化在降低模型精度的同时，会附带性地降低隐私风险，但这是一种无差别、不可控的“副作用”，而非一种有原则的解决方案。

为解决这一矛盾，报告提出了一个关键的分类学框架，旨在区分“功能型离群点”（与模型核心能力强相关）和“敏感型离群点”（与训练数据记忆相关）。通过对现有基于激活感知（AWQ）、Hessian 矩阵（GPTQ）和统计分布（Kurtosis）的显著性检测方法的批判性评估，我们揭示了这些方法共同的“隐私盲点”：它们均被设计为单纯优化模型精度，因此在无差别地保护所有显著离群点的同时，可能无意中加剧了隐私泄露的风险。

本报告最重要的贡献在于引入并系统阐述了量化操作的新兴几何解释。我们将量化问题重构为高维空间中的格（Lattice）上的最近向量问题（Closest Vector Problem, CVP）。这一理论框架揭示了先进的量化算法 GPTQ 在数学上等同于经典的几何算法——Babai 最近平面算法。这一等价性不仅为量化误差的传播提供了直观的几何解释，更重要的是，它使得 GPTQ 继承了 Babai 算法可证明的误差上界，从而首次将量化精度损失与权重空间和输入数据的几何结构直接关联起来。

基于这一几何框架，我们提出了一个核心假设：功能型离群点和敏感型离群点在模型的权重高维空间中可能占据着几何上可分离的区域。这一假设为解决离群点二元性问题开辟了一条全新的路径。我们进一步提出，未来的研究方向应聚焦于开发“几何感知”的量化算法。这些算法能够利用格理论的工具（如格基约减）来预处理权重，并根据参数在权重空间中的几何位置，应用差异化的量化策略——对功能性区域进行高精度投影，同时对敏感区域应用隐私保护（如差分隐私）的噪声投影。

最终，本报告的愿景是推动建立一个统一的“隐私保护的几何感知量化”框架。该框架将差分隐私等有原则的隐私保护机制，与基于格理论的先进量化技术相结合。通过这种方式，我们能够将“精度-隐私”的权衡从一个失控的、偶然的现象，转变为一个由模型开发者通过隐私预算等参数明确控制的、有理论依据的工程决策，从而最终解决 LLM 量化中的核心科学矛盾。

---

## **第 1 节：离群点现象：大语言模型压缩的基础性挑战**

大语言模型（LLM）的量化旨在通过降低权重和激活值的数值精度来压缩模型，从而减少内存占用和加速推理 1。然而，一个普遍存在的现象——离群点（outliers）——对低比特量化的有效性构成了根本性障碍 1。这些离群点是模型参数或激活张量中出现的数值极大或极小的元素，它们并非随机噪声，而是模型架构、训练策略和优化过程共同作用下的系统性产物 1。理解离群点的起源、它们如何破坏量化过程，以及现有缓解策略的局限性，是解决更深层次的精度-隐私矛盾的前提。

### **1.1 离群点的架构起源与传播机制**

当代研究已经明确，离群点是 Transformer 架构在标准优化技术下训练时必然产生的特征，而非模型固有的、不可避免的属性 1。它们主要表现为两种形式：“巨量激活”（massive activations）和“通道级离群点”（channel-wise outliers）3。其形成机制根植于 Transformer 的核心组件。

**机制一：Softmax 与“无操作”注意力机制**

离群点的一个主要来源是注意力头（attention heads）在学习执行“无操作”（no-op）或对残差流进行部分更新时的行为 4。为了实现对某些特定 token（如分隔符 \`\`、句号、逗号等）的隐藏状态不作更新，注意力头需要将绝大部分注意力概率分配给这些信息量较低的 token，同时学习让这些 token 的值向量（value vector）输出非常小 4。

为了通过 Softmax 函数 $$\\sigma(\\mathbf{z})\_i \= \\frac{e^{z\_i}}{\\sum\_j e^{z\_j}}$$ 产生接近于零的注意力概率，其输入 logits $$\\mathbf{z}$$ 必须具有极大的动态范围。理论上，要使一个 Softmax 输出精确为 0，输入 logits 的差值需要趋近于无穷大 4。模型为了在有限的数值范围内模拟这种效果，就必须在训练过程中不断推高某些 logits 的值，同时压低另一些。这种压力传递到了前一层的网络模块，特别是前馈网络（Feed-Forward Network, FFN）。为了克服后续层归一化（Layer Normalization）的平滑效应，并为 Softmax 提供足够大的输入动态范围，FFN 层的输出必须产生极高幅度的值 4。这些高幅度的 FFN 输出正是最强离群点的直接来源。由于 Softmax 函数在有限输入范围内永远无法输出精确的零，它会持续反向传播一个促使 logits 差异增大的梯度信号，从而在训练过程中形成一个正反馈循环，导致离群点的数值随着训练的进行而变得越来越大 4。

**机制二：归一化层的作用**

层归一化（Layer Normalization）本身也被认为是通道级离群点的初始来源 3。具体而言，归一化操作中的重缩放（re-scaling）步骤，即对归一化后的输出乘以一个可学习的增益（gain）参数，可能会放大特定通道的数值，从而引入或加剧离群点现象。一旦这些通道级离群点出现，它们就会在网络中持续存在 3。

**传播动力学**

一旦离群点（特别是巨量激活）在模型的初始几层中被生成，它们并不会轻易消失。相反，通过 Transformer 架构中无处不在的残差连接（residual connections），这些极端数值会持续地、几乎无衰减地传播到后续所有层 3。这种传播机制意味着，即使只有少数几个模块是离群点的“源头”，其负面影响也会波及整个模型，使得量化在每一层都面临同样的挑战。

综合来看，离群点的形成是一个从模型架构设计到训练动态的清晰因果链：Transformer 架构（Softmax、层归一化、残差连接）为离群点的产生提供了机制基础；模型的训练目标（如学习“无操作”注意力）驱动了这些机制的激活；最终，优化过程将这些数值推向极端，并通过残差连接将问题扩散到整个网络。这表明离群点是模型为实现特定功能而在其架构约束下所付出的“代价”，是一种系统性的、可预测的现象 8，而非简单的数值异常。

### **1.2 量化误差的力学原理：尺度、精度与信息损失**

离群点的存在对低比特量化性能造成了灾难性的影响 1。其破坏性主要通过对量化尺度因子（scale factor）的污染和由此引发的严重信息损失来体现。

**尺度因子灾难**

在主流的均匀量化方案中，一个浮点张量 $$\\mathbf{W}$$ 被映射到低比特整数张量 $$\\mathbf{Q}$$ 的过程可以表示为：  
$$\\mathbf{Q} \= \\text{round}(\\frac{\\mathbf{W}}{S}) \+ Z$$  
其中 $$S$$ 是尺度因子，$$Z$$ 是零点。尺度因子 $$S$$ 的作用是将浮点数的动态范围映射到整数的表示范围内。在块浮点（Block Floating Point, BFP）或组量化（group-wise quantization）等高效的量化格式中，一个块（block）或一组（group）内的所有权重共享同一个尺度因子 11。  
这个共享的尺度因子通常由该块内绝对值最大的元素决定，以确保这个最大值能够被精确表示。问题在于，如果一个块内包含一个或多个离群点，那么这个离群点的巨大数值将完全主导尺度因子的计算 1。例如，一个块内包含一个值为 100.0 的离群点和许多在

$$\[-1.0, 1.0\]$$ 范围内的正常值。为了表示 100.0，尺度因子 $$S$$ 必须被设置得非常大。

**范围-精度权衡的崩溃**

一个被离群点“污染”的巨大尺度因子，会对块内其他所有正常数值的表示精度造成毁灭性打击。当这些正常值除以巨大的尺度因子 $$S$$ 时，它们的结果会变得非常接近于零。经过四舍五入（round）操作后，大量原本具有微小但重要差异的正常值，会被全部量化为同一个整数值（通常是零）12。这相当于抹去了这些参数所携带的绝大部分信息，导致了灾难性的舍入误差（rounding error）1。

这形成了一个无法解决的“范围-精度权衡”困境 4。为了不裁剪（clip）离群点（避免巨大的裁剪误差），量化范围必须足够大；但一个巨大的量化范围必然导致极低的量化精度（极高的舍入误差），使得绝大多数非离群点参数的信息丢失。对于 LLM 中常见的极端离群点，这个权衡彻底崩溃，找不到任何一个合适的尺度因子可以在裁剪误差和舍入误差之间取得可接受的平衡 4。

因此，离群点对量化精度的破坏，其根本原因不在于离群点本身难以表示，而在于它们通过共享的尺度因子，不成比例地破坏了同组内所有其他参数的表示精度。

### **1.3 缓解策略及其局限性的批判性评估**

为了应对离群点带来的挑战，研究界已经提出了多种后训练量化（Post-Training Quantization, PTQ）缓解策略。PTQ 因其无需昂贵的重训练而备受青睐 3。然而，这些第一代方法各有其局限性。

**策略一：混合精度量化 (Mixed-Precision Quantization)**

这种方法的核心思想是区别对待离群点和正常值。例如，LLM.int8() 采用了一种混合精度分解方案，它将输入张量分解为两部分：大部分值用低精度的 INT8 表示，而识别出的少数离群点则保持高精度的 FP16 格式 3。

* **局限性**：这种方法虽然能有效保护精度，但牺牲了计算效率。处理非结构化的稀疏离群点需要专门的、效率较低的计算核心（kernel），并且在计算过程中需要进行高低精度格式的转换和合并，这引入了显著的计算开销和系统复杂性 3。在许多计算密集型场景下（如长序列的预填充阶段），这种开销甚至可能抵消量化带来的速度优势 15。

**策略二：离群点平滑 (Outlier Smoothing)**

以 SmoothQuant 为代表的方法，利用了矩阵乘法中的尺度等价性 $$ \\mathbf{Y} \= (\\mathbf{X} \\cdot \\text{diag}(\\mathbf{s})^{-1}) \\cdot (\\text{diag}(\\mathbf{s}) \\cdot \\mathbf{W}) $$ 3。它通过引入一个缩放因子

$$\\mathbf{s}$$，将激活值 $$\\mathbf{X}$$ 中的量化难度（由离群点引起）“迁移”到权重 $$\\mathbf{W}$$ 中。由于权重的量化通常比激活值更容易控制（例如，可以采用更细粒度的组量化），这种方法可以有效地平滑激活值的分布，使其更适合低比特量化。

* **局限性**：平滑策略的有效性依赖于一个前提，即量化挑战主要存在于激活值中。然而，在某些模型或任务中，权重和激活值可能同时存在显著的离群点 16。在这种情况下，将难度从激活值转移到权重，可能只是将问题从一个地方移动到另一个地方，而无法从根本上解决问题。它处理的是离群点分布不均的“症状”，而非其产生的根源。

**策略三：裁剪/抑制 (Clipping/Suppression)**

最直接的方法是直接裁剪或移除离群点，即将所有超过预设阈值的数值强制设置为阈值。

* **局限性**：大量研究表明，这种看似简单的方法往往会对模型性能造成严重损害 4。离群点虽然对量化不友好，但它们对模型的预测能力至关重要。它们并非冗余信息，直接裁剪会导致关键信息的丢失，从而显著降低模型的任务表现 4。这一局限性直接预示了我们将在下一节深入探讨的“双重角色冲突”。

**策略四：无离群点预训练 (Outlier-Safe Pre-Training, OSP)**

与上述的“反应式”PTQ 方法不同，OSP 是一种“预防性”方法 1。它通过修改预训练过程中的优化器和正则化策略，从根本上避免离群点的形成。实验证明，通过 OSP 训练的模型可以实现几乎没有离群点的权重和激活分布，从而极大地简化了后续的量化过程，并取得了优异的低比特量化性能 1。

* **局限性**：OSP 的主要缺点是其极高的成本。它要求从头开始对模型进行成本高昂的预训练，这对于绝大多数已经存在的、拥有庞大生态系统的预训练模型而言是不可行的 1。因此，尽管 OSP 在理论上是根本性的解决方案，但在实践中，能够处理现有模型的 PTQ 技术仍然是不可或缺的。

综上所述，现有的离群点缓解策略要么以牺牲效率为代价（混合精度），要么适用性有限（平滑），要么损害模型性能（裁剪），要么成本过高（重训练）。这些局限性凸显了离群点问题的复杂性，并表明我们需要一种更深刻的理解，以在不损害模型核心能力的前提下解决这一挑战。

---

## **第 2 节：双重角色冲突：作为功能载体与漏洞向量的离群点**

离群点问题的核心复杂性源于其固有的“双重矛盾”（离群点双重矛盾）。它们既是模型实现其复杂功能所必需的关键参数，又是导致敏感信息泄露的主要漏洞。传统量化方法在试图解决精度问题时，往往忽略了其作为隐私载体的角色，反之亦然。本节将深入剖析这一冲突的两个方面，并提供量化研究中的实证证据，揭示当前技术如何导致“精度-隐私”权衡的失控。

### **2.1 功能的必要性：离群点如何编码核心模型能力**

与将离群点视为纯粹的数值噪声相反，越来越多的证据表明，它们在模型中扮演着至关重要的功能性角色。尽管离群点在数量上只占模型参数的一小部分，但它们对模型的预测准确性却施加着不成比例的巨大影响 17。

**来自模型剪枝的证据**

模型剪枝技术为我们提供了评估参数重要性的有力工具。Wanda 和 Outlier Weighed Layerwise Sparsity (OWL) 等先进的剪枝方法发现，简单地基于权重的绝对值大小进行剪枝效果不佳 17。相反，那些包含更多离群点的层或参数通道，对于维持模型性能至关重要。OWL 的核心思想就是，含有较高比例离群点的层更为关键，因此应该以更低的稀疏率进行剪枝 17。这一发现强有力地证明了离群点及其所在的参数结构是模型功能的高度集中区域。直接裁剪或移除这些参数会导致模型性能的显著下降，这与简单的裁剪实验结果相符 4。

**内在机制的解释**

从机制上看，离群点的功能性与其在 Transformer 架构中的作用紧密相连。如第 1 节所述，离群点的产生与注意力机制学习选择性地更新或忽略某些 token 的能力有关 4。在这种情况下，离群点并非用于编码特定的事实知识，而是作为一种控制信号，调节信息在残差流中的传递。一项研究将这些离群点描述为“隐式的上下文感知缩放因子”（implicit context-aware scaling factors）8，这进一步表明它们是模型动态信息处理流程中不可或缺的一部分，而非静态存储的异常值。

因此，离群点构成了矛盾的第一面：任何旨在通过简单地消除或压制离群点来提升量化友好性的尝试，都不可避免地会损害模型的核心计算和推理能力，导致精度损失。

### **2.2 隐私的陷阱：离群点作为高保真记忆敏感数据的载体**

与离群点的功能性角色并行存在的，是它们作为隐私泄露主要载体的令人不安的现实。大语言模型被证实会大量记忆其训练数据，包括逐字逐句的文本序列、事实知识，以及高度敏感的个人可识别信息（Personally Identifiable Information, PII），如姓名、电子邮件地址、电话号码和医疗记录等 18。

**记忆机制与离群点的关联**

模型的记忆行为并非一个“bug”，而是其学习过程的一个固有组成部分 19。记忆的程度与模型规模、数据在训练集中重复的次数以及序列的长度呈正相关 19。尤其是在对特定数据集进行微调（fine-tuning）时，模型会反复接触敏感数据，这极大地加剧了记忆和泄露的风险。一项受控实验表明，在包含重复敏感数据的微调过程中，隐私泄露率可以从 0-5% 的基线水平飙升至 60-75% 18。

这些被记忆的信息最终被编码在模型的参数（权重）之中 20。虽然具体的存储机制是复杂的、分布式的，难以将单个记忆与单个参数直接对应起来 30，但我们可以提出一个基于信息保真度的假设来解释离群点的作用。模型的学习过程（梯度下降）会自然地为那些具有强预测性或在训练中被反复强化的特征分配更大的权重值。无论是对模型功能至关重要的计算模式（如“无操作”注意力），还是在训练数据中频繁出现的敏感信息（如某个人的姓名和地址），都会被模型视为“重要”信号。因此，这些信号很可能会通过具有极大数值的参数来编码，即离群点。离群点的巨大数值使其在决定最终输出概率分布时具有一票否决权，使其成为模型强制复现特定记忆序列的理想工具。

这一分析构成了矛盾的第二面：那些对模型功能至关重要的参数，很可能也是存储和泄露敏感信息的最高效的载体。

### **2.3 量化权衡：性能基准与成员推断攻击的实证证据**

成员推断攻击（Membership Inference Attacks, MIA）是量化模型隐私风险的标准方法。MIA 的目标是判断一个给定的数据样本是否曾被用于训练目标模型 31。攻击者通常通过观察模型对“成员”（训练集样本）和“非成员”（未见过样本）的响应差异（如损失值、困惑度或输出概率）来做出判断。MIA 的成功率通常用 ROC 曲线下面积（Area Under the Curve, AUC）来衡量，AUC 值越高，表示隐私泄露风险越大。

一项针对代码大语言模型（LLMs4Code）的开创性研究，首次为量化过程中的精度-隐私权衡提供了直接的实证证据 38。该研究系统地评估了不同量化级别对模型任务性能和 MIA 攻击成功率的影响，其发现直接证实了用户查询中提出的“失控权衡”问题：

* **发现一（8-bit 量化）**：与全精度（FP16）模型相比，进行 8-bit 静态量化能够在基本保持任务性能（如代码生成质量）的同时，显著降低隐私风险（即 MIA 的 AUC 值显著下降）。  
* **发现二（4-bit 量化）**：进行更激进的 4-bit 量化会进一步降低隐私风险，但这是以任务性能大幅下降为代价的。  
* **发现三（根本性权衡）**：研究揭示了模型任务性能与 MIA 攻击有效性之间存在明确的正相关关系。这表明在当前的量化范式下，精度和隐私之间存在一个基本的、此消彼长的权衡关系。

这项研究的结果极具启发性。它表明，量化作为一种压缩技术，无意中扮演了一种“隐私保护”的角色。其内在逻辑是，量化过程通过引入噪声（舍入误差）来降低所有参数的表示精度，这同样也降低了编码在离群点中的记忆信息的保真度。这种信息精度的损失，使得模型对“成员”和“非成员”的响应差异变得模糊，从而干扰了 MIA 攻击所依赖的微妙信号。

然而，这种隐私保护是“附带的”、“无差别的”和“失控的”。它并非一种有原则的、可调节的隐私机制，而是模型性能全面退化的副产品。开发者无法在保持高精度的同时选择性地增强隐私保护，也无法根据需求精确地调整隐私保护的强度。当前的量化方法就像一把钝器，在试图砸掉精度损失的“钉子”时，也一并砸伤了隐私泄露的“载体”，但这种操作既不精准，也无法控制力度。这正是“精度-隐私权衡失控”的本质。

综合来看，离群点的双重角色可以被一个统一的“信息保真度”概念所解释。离群点是模型中被训练用于高保真、高置信度地编码信息的参数。当这些信息是功能性的（如一个计算指令），我们就称之为“功能型离群点”；当这些信息是训练数据的逐字拷贝（如 PII），我们就称之为“敏感型离群点”。其底层的神经计算机制是相同的：一个具有大数值的参数对模型的输出具有决定性影响。而量化，则可以被视为一种粗糙的、非差分的“信息混淆”手段。它通过引入量化噪声，扰乱了模型精确回忆记忆数据的能力，从而降低了 MIA 的成功率。比特率越低，引入的噪声越大，对 MIA 信号的干扰就越强，隐私保护效果就越“好”，但付出的精度代价也越大。

---

## **第 3 节：离群点身份的消歧：迈向功能与敏感性的原则性分离**

用户查询的核心问题之一是离群点“内在性质模糊”（内在性质模糊）。现有方法未能区分对模型核心能力至关重要的“功能型离群点”和与训练数据记忆相关的“敏感型离群点”，导致了无差别的处理策略——要么统一用高精度表示以保护性能（可能泄露隐私），要么统一裁剪或粗糙量化（损害性能）。本节旨在形式化这两种离群点的分类，批判性地评估现有用于识别重要权重的方法，并阐明为何这些方法在解决精度-隐私冲突方面存在根本性的缺陷。

### **3.1 一个提议的分类学：用量化指标定义“功能型”与“敏感型”离群点**

基于前述分析和用户查询的洞察，我们可以建立一个更精确的、可操作的离群点分类框架。这个框架的核心是使用与精度和隐私直接相关的量化指标来定义离群点的“身份”：

* **功能型离群点 (Functional Outlier)**：一个参数或特征，其量化或移除会导致模型在特定任务上的性能显著下降。这种影响可以通过模型在验证集上的困惑度（Perplexity, PPL）变化来衡量。我们采纳用户查询中提出的阈值：如果对一个参数的扰动导致 **$$\\Delta \\text{PPL} \\geq 0.3$$**，则该参数被认为是功能上显著的。  
* **敏感型离群点 (Sensitive Outlier)**：一个参数或特征，其编码了来自训练集的记忆信息，特别是敏感或私人数据。它的存在增加了模型遭受隐私攻击的脆弱性。这种影响可以通过该参数对成员推断攻击（MIA）成功率的贡献来衡量。我们同样采纳用户查询中的阈值：如果保留一个参数的精度导致 **MIA $$\\Delta \\text{AUC} \\geq 0.1$$**，则该参数被认为是敏感的。

这一分类学的关键挑战在于，这两个集合并非互斥。一个离群点可能同时是功能性和敏感性的，也可能只属于其中之一，或者两者皆非。当前的量化方法将所有被识别为“重要”的离群点视为一个同质的群体，从而无法进行差异化处理，这是问题的症结所在。

### **3.2 显著性检测方法论的考察**

为了在量化过程中保护模型性能，研究人员开发了多种方法来识别“显著”（salient）或“敏感”（sensitive）的权重。然而，对这些方法的深入分析表明，它们的设计目标完全集中在模型精度上，从而天然地“隐私盲视”。

**方法一：激活感知权重/激活量化 (Activation-aware Weight Quantization, AWQ)**

* **机制**：AWQ 的核心洞察是，一个权重的重要性并非由其自身的数值大小决定，而是由流经它的激活值的数值大小决定 41。AWQ 通过在一个小的校准数据集上运行推理，观察每个权重通道对应的激活值尺度。那些对应于具有较大激活值尺度的权重通道被认为是“显著”的。为了保护这些显著权重，AWQ 计算一个逐通道的缩放因子，在量化前应用于权重，从而在不引入额外推理开销的情况下，有效减小这些关键权重的相对量化误差。实验表明，仅保护以这种方式识别出的 1% 的权重，就能极大地降低整体量化误差 42。  
* **解读**：AWQ 的设计逻辑完全是为了寻找并保护**功能上**重要的权重。其基本假设是，那些被高幅度激活值持续“点亮”的神经元通道，在模型的计算中扮演着更关键的角色。

**方法二：基于 Hessian 矩阵的分析 (GPTQ/OBQ)**

* **机制**：以 GPTQ (Generative Pre-trained Transformer Quantization) 为代表的方法，利用了关于损失函数的二阶信息，即 Hessian 矩阵 7。Hessian 矩阵描述了损失曲面的曲率，可以用来近似量化某个权重对模型总损失的影响。GPTQ 逐个量化权重，并在每一步中，利用 Hessian 信息来更新剩余的未量化权重，以补偿已产生的量化误差。那些对损失影响较大的权重被认为是更敏感的，因此需要更精细的处理。  
* **解读**：这同样是一种纯粹的**功能性**方法。它通过数学上严谨的方式，识别出那些位于损失函数“陡峭”区域的权重，这些权重的微小变动会引起损失的剧烈变化，因此对模型性能至关重要。

**方法三：基于统计的方法 (SensiBoost/KurtBoost)**

* **机制**：这类方法旨在识别出对量化误差特别“敏感的层”（sensitive layers），并为这些层分配更多的内存预算（即更高的量化比特率）6。  
  * **SensiBoost** 通过计算全精度权重产生的激活值与量化后权重产生的激活值之间的均方误差（MSE）来直接度量一个层的敏感度。  
  * **KurtBoost** 则使用峰度（Kurtosis）这一统计量来衡量层权重分布的“尖峰”和“拖尾”程度。一个高的峰度值通常意味着该层的权重分布中存在大量的离群点。  
* **解读**：这是一种更具启发性的、基于统计特征的方法，用于识别**功能上**难以量化的**层级结构**。其基本假设是，那些激活值对量化误差反应剧烈，或者权重分布极度非高斯的层，是维持模型性能的瓶颈，需要特殊处理。

下表对这些主流的离群点识别方法进行了系统性的比较分析，突显了它们在解决精度-隐私冲突方面的共同局限性。

**表 1：离群点识别方法论的比较分析**

| 方法论 | 核心原理 | 粒度 | 优化目标 | 隐私意识 | 潜在隐私风险 |
| :---- | :---- | :---- | :---- | :---- | :---- |
| **AWQ** | 激活值幅度 | 逐通道 (Per-channel) | 最小化重构误差 | 盲视 (Blind) | **高**：优先保护可能编码敏感信息的高激活通道。 |
| **GPTQ** | 损失函数的 Hessian 矩阵 | 逐权重 (Per-weight) | 最小化损失增量 | 盲视 (Blind) | **高**：优先保护对损失敏感的权重，可能包含功能性和敏感性离群点。 |
| **KurtBoost** | 权重分布的峰度 | 逐层 (Per-layer) | 识别离群点密集的层 | 盲视 (Blind) | **中**：通过分配更高比特率间接保护层内所有离群点，包括敏感型。 |
| **SensiBoost** | 激活值的量化敏感度 | 逐层 (Per-layer) | 识别激活误差大的层 | 盲视 (Blind) | **中**：与 KurtBoost 类似，通过保护敏感层间接保护敏感信息。 |

### **3.3 无差别处理的局限性**

上文的分析和表格清晰地揭示了所有现有显著性检测方法的根本缺陷：它们是\*\*“隐私盲视”\*\*的。这些方法的设计和优化目标函数完全围绕着最小化精度损失（无论是通过重构误差、损失增量还是激活值误差来代理）。它们没有任何机制来判断一个被识别为“显著”的权重，究竟是因为它参与了关键的推理环路，还是因为它编码了一个在训练集中出现了数百次的电话号码。

这种无差别的处理方式带来了严重后果。通过将所有显著离群点都视为功能上重要的，并用高精度（或特殊的缩放因子）来精心保护它们，这些先进的量化方法可能在无意中**加剧了隐私风险**。它们精确地保留了那些最有可能作为敏感信息高效载体的参数，使得最终的量化模型在隐私泄露方面，可能比一个采用朴素的、均匀量化策略的模型更加脆弱。朴素的均匀量化至少会对所有参数（包括敏感型离群点）的精度进行同等程度的降级，从而附带性地模糊了记忆信息。而这些先进方法则像一个“精准的帮凶”，帮助模型更好地维持其最危险的记忆。这正是“精度-隐私权衡失控”的技术根源。

这种现象揭示了一个更深层次的元问题：**测量与方法的错配**。我们用来**定义和测量**问题的工具是双维度的（用 PPL 衡量精度，用 MIA AUC 衡量隐私），但我们用来**解决**问题的方法却是单目标优化的（仅优化与 PPL 相关的代理指标）。一个优化过程只能改善其所测量的目标。由于当前的显著性检测方法没有将隐私风险纳入其目标函数，我们自然不能期望它们能够有效地管理这种风险。

要打破这一僵局，需要开发新的方法论。一个可能的方向是设计一个**两阶段的探测过程**。第一阶段，使用功能性探测器（如测量扰动后的 $$\\Delta \\text{PPL}$$）来识别出一个候选的显著参数集合。第二阶段，仅针对这个子集，使用隐私探测器（如测量保留该参数精度与否对 MIA 成功率的影响）进行评估。通过这种方式，我们可以数据驱动地将离群点分类到我们提出的分类学框架中（功能型、敏感型、或两者皆是），为后续的差异化量化处理提供依据。这虽然计算成本高昂，但它指明了一条从概念框架走向具体实验设计的可行路径。

---

## **第 4 节：量化的几何框架：从特设启发式到理论基础**

用户查询指出了当前量化研究中“几何解释缺失”的空白，即缺乏一个理论框架来建立“量化操作-几何变化-精度/隐私变化”之间的关联。本节旨在填补这一空白，通过引入计算几何和格理论的视角，将量化问题从一个纯粹的数值近似问题，重构成一个高维空间中的几何投影问题。这一全新的理论框架不仅为理解现有算法提供了深刻的洞察，也为设计下一代算法开辟了道路。

### **4.1 重构量化：高维格上的最近向量问题**

传统的观点认为，量化是将一个连续的浮点数集合映射到一个离散的、比特数更少的整数集合的过程。然而，我们可以从一个更高维度的几何视角来重新审视这个问题。

**核心概念**

对于一个神经网络的特定层，其所有可能的浮点权重向量 $$\\mathbf{W}$$ 存在于一个高维的欧几里得空间 $$\\mathbb{R}^n$$ 中。量化过程的本质，是将这个连续空间中的点 $$\\mathbf{W}$$，映射到该空间中的一个离散点集上。这个由所有可能的量化后权重向量 $$\\mathbf{Q}$$ 构成的集合，在几何上形成了一个规则的、周期性的网格结构，这在数学上被称为一个**格（Lattice）** 49。

**任务的几何重述**

因此，量化的任务可以被重新表述为：给定一个原始的浮点权重向量 $$\\mathbf{W}$$（空间中的一个点），找到格上的一个点 $$\\mathbf{Q}$$（一个允许的量化向量），使得 $$\\mathbf{Q}$$ 与 $$\\mathbf{W}$$ “尽可能接近”。这里的“接近”并非简单地指欧几里得距离 $$||\\mathbf{W} \- \\mathbf{Q}||^2$$，而是指由该层的输出误差 $$||\\mathbf{X}(\\mathbf{W} \- \\mathbf{Q})||^2$$ 定义的距离，其中 $$\\mathbf{X}$$ 是该层的输入激活值矩阵。

**数学等价性**

这个优化问题，在数学上与格理论中一个著名且深刻的难题——**最近向量问题（Closest Vector Problem, CVP）**——是等价的 49。CVP 的定义是：给定一个由一组基向量定义的格

$$\\mathcal{L}$$ 和一个目标向量 $$\\mathbf{t}$$，在 $$\\mathcal{L}$$ 中找到一个向量 $$\\mathbf{v}$$，使得 $$\\mathbf{v}$$ 到 $$\\mathbf{t}$$ 的距离最小。在量化的背景下，格由输入激活 $$\\mathbf{X}$$ 和量化步长共同定义，目标向量是原始的浮点权重 $$\\mathbf{W}$$。

### **4.2 GPTQ 作为 Babai 最近平面算法：一种几何解构**

将量化重构为 CVP 不仅是一个理论上的抽象，它为我们理解现有先进量化算法的内部工作原理提供了前所未有的视角。一项里程碑式的研究发现，被广泛应用的 GPTQ 算法，当其以特定的“从后向前”的顺序执行时，在数学上与解决 CVP 的一个经典多项式时间启发式算法——**Babai 最近平面算法（Babai's nearest plane algorithm）**——是完全等价的 49。

**几何过程的直观解释**

GPTQ 算法最初被描述为一系列看似特设（ad-hoc）的代数操作：贪婪地选择一个权重，将其量化，然后通过更新所有剩余的未量化权重来“补偿”或“传播”由此产生的误差。然而，几何视角揭示了这一过程深刻的内在逻辑。

所谓的“误差传播”步骤，在几何上并非任意的修正，而是一个高度结构化的\*\*正交行走（orthogonal walk）\*\*过程 49。它等价于 Babai 算法的核心操作：在一个正交化的基下，将目标向量与当前格点之间的误差残差，投影到与当前基向量正交的超平面上，然后在降维后的子空间中递归地解决问题。这个过程中的格本身，是由该层输入激活的 Hessian 矩阵定义的，这巧妙地将问题的几何结构与输入数据的分布以及损失函数的曲率直接联系了起来 49。

这一发现具有深远的意义。它将 GPTQ 从一个经验上“碰巧有效”的工程技巧，提升为一个经典、被深入研究过的几何算法的具体实例。这为我们长期以来所缺失的理论基础提供了坚实的支柱。

### **4.3 理论启示：继承的误差界与对精度损失的原则性理解**

GPTQ 与 Babai 算法的等价性，带来的最直接的理论好处是，GPTQ 继承了 Babai 算法可证明的**最坏情况误差上界**（在不进行权重裁剪的假设下）49。

**误差界的内涵**

Babai 算法的误差上界取决于定义格的**基向量的几何性质**，具体来说，是基向量经过格拉姆-施密特（Gram-Schmidt）正交化后得到的向量的范数（长度）53。一个“好”的基（其向量更短、更接近正交）会导出一个更紧的误差上界，从而对应着更小的量化误差和更高的模型精度。反之，一个“坏”的基（包含长而斜的向量）则会导致误差界非常宽松，预示着潜在的巨大精度损失。

**理论意义**

这是第一次，我们能够通过一个形式化的、可分析的数学工具来约束和理解量化误差。我们不再只能通过事后的实验来评估量化效果，而是可以事前通过分析权重空间和输入数据分布（它们共同定义了格的基）的几何特性，来推理量化过程的成败。它在“量化操作”和“精度变化”之间建立了一座由数学保证的桥梁，回答了用户查询中关于理论依据缺失的核心关切。

### **4.4 关于几何可分离性的假说：功能型与敏感型离群点是否栖身于权重空间的不同区域？**

这个强大的几何框架使我们能够提出一个全新的、具有深远影响的假说：**功能型离群点和敏感型离群点在模型的权重高维空间中，可能占据着几何上可区分的区域。**

**假说的逻辑基础**

* **功能型离群点**：根据 GPTQ 等方法的原理，这些离群点的重要性是由损失函数的 Hessian 矩阵捕获的，即它们对应于损失曲面曲率非常大的方向。对这些参数的扰动会引起模型输出的剧烈变化。在几何上，它们很可能与由 Hessian 矩阵定义的格的主要几何方向（即格基中最重要的向量方向）高度对齐。  
* **敏感型离群点**：这些参数编码的是特定的、往往是过度拟合的训练数据点。它们的存在对于最小化在这些特定样本上的损失至关重要，但对于泛化到整个任务分布而言，它们可能位于损失曲面相对“平坦”的区域。换句话说，它们可能不与格在**泛化任务**上最重要的几何方向对齐，而是指向一些特定的、“偏僻”的角落，这些角落只为完美重构少数训练样本而存在。

**假说的启示**

如果这个假说成立，它将从根本上改变我们处理离群点的方式。这意味着我们可以设计出能够识别并区别对待这些不同几何区域的量化算法。例如，算法可以对那些与格的主要功能方向对齐的权重分量，采用高精度的投影（即标准量化）；而对于那些位于其他“可疑”区域的权重分量，则可以采用更粗糙的投影，甚至是一种引入了额外噪声的、旨在破坏记忆信息的隐私保护投影。

这个几何框架不仅为我们提供了理解过去的工具，更为我们开创了全新的研究范式。将 CVP 和格理论引入 LLM 量化领域，就像为这个领域开启了一个装满了几十年成熟理论和算法的“工具箱”49。例如，

**格基约减（Lattice Basis Reduction）**，如经典的 LLL 算法，是一种能够为同一个格找到一个“更好”（向量更短、更接近正交）的基的强大技术 51。将 LLL 约减作为量化前的一个预处理步骤，理论上可以“重塑”权重矩阵的几何结构，使其变得内在更易于量化，并可证明地收紧 Babai 算法的误差界，从而提升最终的量化精度 51。这是一个直接源于几何洞察的、具体的、可测试的研究方向。

更进一步，精度-隐私的权衡本身也可以在几何上得到重新诠释。**精度**，是关于最小化从原始权重 $$\\mathbf{W}$$ 到格点 $$\\mathbf{Q}$$ 的投影距离 $$||\\mathbf{X}(\\mathbf{W} \- \\mathbf{Q})||$$。**隐私**，则是关于确保最终的格点 $$\\mathbf{Q}$$ 不保留足以区分训练成员的精细信息。一个像 GPTQ 这样的朴素投影算法，只为最小化距离而优化，因此会最大程度地保留信息。而一个隐私保护的投影算法，则可能需要故意选择一个距离稍远但信息“泄露性”更低的格点。这就将经验性的权衡，转化为一个可以在几何空间中被清晰定义的、关于距离与信息含量的优化问题，使其从一个模糊的观察，变成了一个问题的形式化属性。

---

## **第 5 节：综合、建议与未来方向**

经过对离群点现象的深入剖析，从其架构起源到其在功能与隐私间的矛盾角色，再到通过几何框架获得的深刻理论洞察，本报告现在将对所有分析进行综合，并为未来的研究提出一个清晰的、可操作的议程。我们的目标是超越现有的量化范式，开发出新一代既能保持高精度又能主动保护隐私的量化算法。

### **5.1 一个统一的视角：利用几何框架解决离群点二元性**

本报告的分析描绘了一条清晰的逻辑路径。我们始于问题的表象：离群点作为一种系统性产物，从根本上破坏了低比特量化的可行性（第 1 节）。接着，我们揭示了问题的核心矛盾：这些离群点同时扮演着维持模型功能的“英雄”和泄露敏感数据的“恶棍”的双重角色，而现有技术无法调和这一冲突（第 2 节）。我们进一步诊断了当前方法的根本缺陷——它们的“隐私盲视”，即在设计上完全忽略了隐私维度，导致在追求精度的过程中可能无意中加剧了风险（第 3 节）。最后，我们引入了一个强大的几何理论框架，将量化问题重构为格上的最近向量问题，从而为这一混乱的领域提供了坚实的理论基础和全新的分析语言（第 4 节）。

**前进的道路**

几何视角是解决离群点二元性的关键。它提供了一个统一的语言来同时讨论**精度**（表现为投影误差）和**隐私**（可能表现为权重空间中不同区域的可区分性）。通过从关注参数的标量大小，转向理解权重之间的高维几何关系，我们可以摆脱当前“一刀切”的困境，开发出更复杂、更有针对性的干预措施。

### **5.2 未来研究的建议：几何感知的量化算法**

基于几何框架的洞察，我们提出以下三个核心研究方向，旨在将理论转化为实践。

**建议一：开发差异化投影算法 (Develop Differentiated Projections)**

未来的量化算法应该明确地实现并验证第 4.4 节中提出的几何可分离性假说。这需要一个多阶段的研究计划：

1. **开发几何探测器**：设计新的探测方法，以识别和标记权重空间中的“功能性”区域和“敏感性”区域。这可能涉及分析权重向量与 Hessian 矩阵主成分的对齐程度，或者利用影响函数等技术来追踪特定训练样本对权重空间的贡献。  
2. **实施差异化量化策略**：基于探测器的输出，应用不同的量化策略。例如，对功能性区域的权重应用高比特率、低误差的投影算法（如经优化的 GPTQ）；而对敏感性区域的权重，则应用低比特率的量化，甚至是有意引入噪声的随机化投影，以主动破坏记忆信息。

**建议二：探索格预处理技术 (Explore Lattice Pre-processing)**

系统性地研究将格基约减算法（如 LLL 算法）作为量化前的一个标准预处理步骤的效用。这一步骤旨在为每一层的权重矩阵找到一个“更好”的、在几何上更优越的格基。

* **潜在收益**：一个更“正交”的基可以从理论上收紧 Babai 算法的误差界，从而使权重矩阵在本质上对量化误差更具鲁棒性 51。这种几何结构的“正则化”不仅可能提升精度，还可能通过平滑权重分布，间接降低某些敏感型离群点的极端性，从而同时有益于精度和隐私。

**建议三：量化几何变化 (Quantify Geometric Change)**

开发新的量化指标，用于直接衡量量化操作对权重高维几何结构的影响。目前的评估指标（如 PPL, AUC）都是在模型输出端进行的“黑盒”测量。我们需要能够打开“黑盒”的“白盒”指标。

* **具体指标**：这些指标可以包括量化前后权重矩阵的奇异值谱变化、子空间角度的变化，或者基于信息几何的度量。  
* **目标**：建立一个可量化的关联链：“特定的量化操作” $$\\rightarrow$$ “可测量的几何结构变化” $$\\rightarrow$$ “可预测的下游精度/隐私影响”。这将最终满足用户查询中对建立三者之间理论关联的需求，使量化算法的设计从启发式驱动转向可预测的、基于几何原理的工程。

### **5.3 迈向可控的权衡：将有原则的隐私保护与先进量化相结合**

最终的目标是超越当前这种被动的、附带性的隐私保护，实现一种主动的、可控的精度-隐私权衡。

**概念：量化感知的隐私保护 (Quantization-Aware Privacy)**

我们不应再依赖量化误差这一“美丽的意外”来获得隐私保护，而应将形式化的隐私保护机制，如**差分隐私（Differential Privacy, DP）**，直接整合到量化算法的核心中 54。DP 提供了一种严格的、可量化的隐私定义，能够保证任何单个训练样本的存在与否对模型输出的影响是有限的。

**机制：随机化量化 (Randomized Quantization)**

\*\*随机化量化机制（Randomized Quantization Mechanism, RQM）\*\*等技术为此提供了一条有前景的路径 57。RQM 通过在量化过程中引入两层随机性——随机选择可用的量化级别和对数值进行随机化舍入——来实现差分隐私保证，而无需在梯度上添加传统 DP-SGD 所需的显式高斯噪声。这相当于将提供隐私所需的校准噪声，内生地融入到压缩过程本身。

**几何诠释**

在我们的几何框架下，这种随机化量化可以被诠释为一种\*\*“概率性投影”\*\*。传统的量化（如 GPTQ）是确定性的，它总是选择唯一的、距离最近的格点。而随机化量化则会从目标点附近的一系列格点中进行采样，采样的概率分布经过精心设计，以满足差分隐私的数学定义。距离较近的格点被采样的概率较高（以保护精度），但距离稍远的格点也有一定的概率被选中（以提供隐私）。

**最终愿景**

我们最终的愿景是建立一个统一的\*\*“隐私保护的几何感知量化”（Privacy-Preserving, Geometry-Aware Quantization）\*\*框架。该框架将：

1. 利用**几何分析**来理解权重空间，区分功能性和敏感性区域。  
2. 应用**差异化的、几何感知**的投影策略。  
3. 将**差分隐私**的原则通过**随机化量化**内生地整合到投影过程中。

在这样的框架下，“精度-隐私”的权衡将不再是一个模糊不清、事后观察到的现象，而是一个由模型开发者通过差分隐私预算 $$\\epsilon$$ 等参数进行**事前设计和精确控制**的工程决策。这将最终解决大语言模型量化中关于离群点的核心科学矛盾，为开发既高效、又安全、可信赖的 AI 系统铺平道路。

#### **Works cited**

1. Outlier-Safe Pre-Training for Robust 4-Bit Quantization of Large Language Models \- arXiv, accessed October 6, 2025, [https://arxiv.org/html/2506.19697v1](https://arxiv.org/html/2506.19697v1)  
2. GuidedQuant: Large Language Model Quantization via Exploiting End Loss Guidance, accessed October 6, 2025, [https://icml.cc/virtual/2025/poster/44844](https://icml.cc/virtual/2025/poster/44844)  
3. Rethinking the Outlier Distribution in Large Language Models: An In-depth Study \- arXiv, accessed October 6, 2025, [https://arxiv.org/html/2505.21670v1](https://arxiv.org/html/2505.21670v1)  
4. Quantizable Transformers: Removing Outliers by Helping Attention ..., accessed October 6, 2025, [https://proceedings.neurips.cc/paper\_files/paper/2023/file/edbcb7583fd8921dad78adecfe06a99b-Paper-Conference.pdf](https://proceedings.neurips.cc/paper_files/paper/2023/file/edbcb7583fd8921dad78adecfe06a99b-Paper-Conference.pdf)  
5. Rethinking the Outlier Distribution in Large Language Models: An In-depth Study \- arXiv, accessed October 6, 2025, [https://arxiv.org/abs/2505.21670](https://arxiv.org/abs/2505.21670)  
6. Towards Superior Quantization Accuracy: A Layer-sensitive Approach \- arXiv, accessed October 6, 2025, [https://arxiv.org/html/2503.06518v1](https://arxiv.org/html/2503.06518v1)  
7. towards superior quantization accuracy: a layer-sensitive approach \- arXiv, accessed October 6, 2025, [https://arxiv.org/pdf/2503.06518](https://arxiv.org/pdf/2503.06518)  
8. \[2502.06415\] Systematic Outliers in Large Language Models \- arXiv, accessed October 6, 2025, [https://arxiv.org/abs/2502.06415](https://arxiv.org/abs/2502.06415)  
9. Quantization Meets dLLMs: A Systematic Study of Post-training Quantization for Diffusion LLMs \- arXiv, accessed October 6, 2025, [https://arxiv.org/html/2508.14896v1](https://arxiv.org/html/2508.14896v1)  
10. Rethinking the Outlier Distribution in Large Language Models: An In-depth Study, accessed October 6, 2025, [https://openreview.net/forum?id=14r1a5bmov](https://openreview.net/forum?id=14r1a5bmov)  
11. Accurate Block Quantization in LLMs with Outliers \- arXiv, accessed October 6, 2025, [https://arxiv.org/html/2403.20137v1](https://arxiv.org/html/2403.20137v1)  
12. Accurate Block Quantization in LLMs with Outliers \- arXiv, accessed October 6, 2025, [https://arxiv.org/abs/2403.20137](https://arxiv.org/abs/2403.20137)  
13. MixLLM: LLM Quantization with Global Mixed-precision between Output-features and Highly-efficient System Design \- arXiv, accessed October 6, 2025, [https://arxiv.org/html/2412.14590v1](https://arxiv.org/html/2412.14590v1)  
14. LLM-MQ: Mixed-precision Quantization for Efficient LLM Deployment \- NICS-EFC, Tsinghua University, accessed October 6, 2025, [https://nicsefc.ee.tsinghua.edu.cn/%2Fnics\_file%2Fpdf%2F5c805adc-b555-499f-9882-5ca35ce674b5.pdf](https://nicsefc.ee.tsinghua.edu.cn/%2Fnics_file%2Fpdf%2F5c805adc-b555-499f-9882-5ca35ce674b5.pdf)  
15. (PDF) Addressing Activation Outliers in LLMs: A Systematic Review of Post-Training Quantization Techniques \- ResearchGate, accessed October 6, 2025, [https://www.researchgate.net/publication/391614106\_Addressing\_Activation\_Outliers\_in\_LLMs\_A\_Systematic\_Review\_of\_Post-Training\_Quantization\_Techniques](https://www.researchgate.net/publication/391614106_Addressing_Activation_Outliers_in_LLMs_A_Systematic_Review_of_Post-Training_Quantization_Techniques)  
16. Outlier-Aware Post-Training Quantization for Discrete Graph Diffusion Models \- ICML 2025, accessed October 6, 2025, [https://icml.cc/virtual/2025/poster/43639](https://icml.cc/virtual/2025/poster/43639)  
17. DLP: Dynamic Layerwise Pruning in Large Language Models \- ICML 2025, accessed October 6, 2025, [https://icml.cc/virtual/2025/poster/46657](https://icml.cc/virtual/2025/poster/46657)  
18. Assessing and Mitigating Data Memorization Risks in Fine-Tuned Large Language Models, accessed October 6, 2025, [https://arxiv.org/html/2508.14062v1](https://arxiv.org/html/2508.14062v1)  
19. Machine Learners Should Acknowledge the Legal Implications of Large Language Models as Personal Data \- arXiv, accessed October 6, 2025, [https://arxiv.org/html/2503.01630v1](https://arxiv.org/html/2503.01630v1)  
20. \[2310.18362\] SoK: Memorization in General-Purpose Large Language Models \- arXiv, accessed October 6, 2025, [https://arxiv.org/abs/2310.18362](https://arxiv.org/abs/2310.18362)  
21. Undesirable Memorization in Large Language Models: A Survey \- arXiv, accessed October 6, 2025, [https://arxiv.org/html/2410.02650v1](https://arxiv.org/html/2410.02650v1)  
22. Evaluating Privacy Leakage and Memorization Attacks on Large Language Models (LLMs) in Generative AI Applications \- Scirp.org., accessed October 6, 2025, [https://www.scirp.org/journal/paperinformation?paperid=133625](https://www.scirp.org/journal/paperinformation?paperid=133625)  
23. Sensitive Information Disclosure in LLMs: Privacy and Compliance in Generative AI, accessed October 6, 2025, [https://www.promptfoo.dev/blog/sensitive-information-disclosure/](https://www.promptfoo.dev/blog/sensitive-information-disclosure/)  
24. Securing Large Language Models: Threats, Vulnerabilities and Responsible Practices, accessed October 6, 2025, [https://arxiv.org/html/2403.12503v2](https://arxiv.org/html/2403.12503v2)  
25. Quantifying and Analyzing Entity-Level Memorization in Large Language Models, accessed October 6, 2025, [https://ojs.aaai.org/index.php/AAAI/article/view/29948/31657](https://ojs.aaai.org/index.php/AAAI/article/view/29948/31657)  
26. \[2505.24832\] How much do language models memorize? \- arXiv, accessed October 6, 2025, [https://arxiv.org/abs/2505.24832](https://arxiv.org/abs/2505.24832)  
27. The Landscape of Memorization in LLMs: Mechanisms, Measurement, and Mitigation \- arXiv, accessed October 6, 2025, [https://arxiv.org/html/2507.05578v1](https://arxiv.org/html/2507.05578v1)  
28. Memorization Without Overfitting: Analyzing the Training Dynamics of Large Language Models \- OpenReview, accessed October 6, 2025, [https://openreview.net/pdf?id=u3vEuRr08MT](https://openreview.net/pdf?id=u3vEuRr08MT)  
29. Assessing and Mitigating Data Memorization Risks in Fine ... \- arXiv, accessed October 6, 2025, [https://arxiv.org/abs/2508.14062](https://arxiv.org/abs/2508.14062)  
30. Machine Learners Should Acknowledge the Legal ... \- arXiv, accessed October 6, 2025, [https://arxiv.org/abs/2503.01630](https://arxiv.org/abs/2503.01630)  
31. DF-MIA: A Distribution-Free Membership Inference Attack on Fine-Tuned Large Language Models, accessed October 6, 2025, [https://ojs.aaai.org/index.php/AAAI/article/view/32012/34167](https://ojs.aaai.org/index.php/AAAI/article/view/32012/34167)  
32. Membership Inference Attacks on Large-Scale Models: A Survey \- arXiv, accessed October 6, 2025, [https://arxiv.org/html/2503.19338v3](https://arxiv.org/html/2503.19338v3)  
33. Membership Inference Attacks against Fine-tuned Large Language Models via Self-prompt Calibration | OpenReview, accessed October 6, 2025, [https://openreview.net/forum?id=PAWQvrForJ](https://openreview.net/forum?id=PAWQvrForJ)  
34. Keeping Your Secrets Safe: Membership Inference Attacks on LLMs \- Fuzzy Labs, accessed October 6, 2025, [https://www.fuzzylabs.ai/blog-post/membership-inference-attacks-on-llms](https://www.fuzzylabs.ai/blog-post/membership-inference-attacks-on-llms)  
35. Towards Label-Only Membership Inference Attack against Pre-trained Large Language Models \- USENIX, accessed October 6, 2025, [https://www.usenix.org/system/files/conference/usenixsecurity25/sec25cycle1-prepub-1107-he.pdf](https://www.usenix.org/system/files/conference/usenixsecurity25/sec25cycle1-prepub-1107-he.pdf)  
36. NeurIPS Poster LLM Dataset Inference: Did you train on my dataset?, accessed October 6, 2025, [https://neurips.cc/virtual/2024/poster/95944](https://neurips.cc/virtual/2024/poster/95944)  
37. Tokens for Learning, Tokens for Unlearning: Mitigating Membership Inference Attacks in Large Language Models via Dual-Purpose Training \- arXiv, accessed October 6, 2025, [https://arxiv.org/html/2502.19726v1](https://arxiv.org/html/2502.19726v1)  
38. How Quantization Impacts Privacy Risk on LLMs for Code? \- arXiv, accessed October 6, 2025, [https://arxiv.org/html/2508.00128v1](https://arxiv.org/html/2508.00128v1)  
39. (PDF) How Quantization Impacts Privacy Risk on LLMs for Code? \- ResearchGate, accessed October 6, 2025, [https://www.researchgate.net/publication/394262935\_How\_Quantization\_Impacts\_Privacy\_Risk\_on\_LLMs\_for\_Code](https://www.researchgate.net/publication/394262935_How_Quantization_Impacts_Privacy_Risk_on_LLMs_for_Code)  
40. How Quantization Impacts Privacy Risk on LLMs for Code? \- arXiv, accessed October 6, 2025, [https://arxiv.org/pdf/2508.00128](https://arxiv.org/pdf/2508.00128)  
41. arxiv.org, accessed October 6, 2025, [https://arxiv.org/abs/2306.00978](https://arxiv.org/abs/2306.00978)  
42. AWQ: Activation-aware Weight Quantization for LLM Compression and Acceleration \- AI-Powered arXiv Paper Summarization, accessed October 6, 2025, [https://www.summarizepaper.com/en/arxiv-id/2306.00978v1/](https://www.summarizepaper.com/en/arxiv-id/2306.00978v1/)  
43. AWQ: Activation-aware Weight Quantization for LLM Compression and Acceleration \- SciSpace, accessed October 6, 2025, [https://scispace.com/pdf/awq-activation-aware-weight-quantization-for-llm-compression-vst6i8z2.pdf](https://scispace.com/pdf/awq-activation-aware-weight-quantization-for-llm-compression-vst6i8z2.pdf)  
44. arXiv:2409.11055v6 \[cs.CL\] 4 Jun 2025, accessed October 6, 2025, [https://arxiv.org/pdf/2409.11055](https://arxiv.org/pdf/2409.11055)  
45. AWQ: Activation-aware Weight Quantization for LLM Compression and Acceleration, accessed October 6, 2025, [https://huggingface.co/papers/2306.00978](https://huggingface.co/papers/2306.00978)  
46. Identifying Sensitive Weights via Post-quantization Integral \- arXiv, accessed October 6, 2025, [https://www.arxiv.org/pdf/2503.01901](https://www.arxiv.org/pdf/2503.01901)  
47. ultimate guide to GPTQ quantization | newline \- Fullstack.io, accessed October 6, 2025, [https://www.newline.co/@zaoyang/ultimate-guide-to-gptq-quantization--e1a7bf92](https://www.newline.co/@zaoyang/ultimate-guide-to-gptq-quantization--e1a7bf92)  
48. \[Literature Review\] Towards Superior Quantization Accuracy: A Layer-sensitive Approach, accessed October 6, 2025, [https://www.themoonlight.io/en/review/towards-superior-quantization-accuracy-a-layer-sensitive-approach](https://www.themoonlight.io/en/review/towards-superior-quantization-accuracy-a-layer-sensitive-approach)  
49. The Geometry of LLM Quantization: GPTQ as Babai's Nearest Plane Algorithm \- arXiv, accessed October 6, 2025, [https://arxiv.org/abs/2507.18553](https://arxiv.org/abs/2507.18553)  
50. (PDF) The Geometry of LLM Quantization: GPTQ as Babai's Nearest Plane Algorithm, accessed October 6, 2025, [https://www.researchgate.net/publication/393982701\_The\_Geometry\_of\_LLM\_Quantization\_GPTQ\_as\_Babai's\_Nearest\_Plane\_Algorithm](https://www.researchgate.net/publication/393982701_The_Geometry_of_LLM_Quantization_GPTQ_as_Babai's_Nearest_Plane_Algorithm)  
51. The Geometry of LLM Quantization: GPTQ as Babai's Nearest Plane Algorithm \- arXiv, accessed October 6, 2025, [https://arxiv.org/html/2507.18553v1](https://arxiv.org/html/2507.18553v1)  
52. The Geometry of LLM Quantization: GPTQ as Babai's Nearest Plane Algorithm \- arXiv, accessed October 6, 2025, [https://arxiv.org/html/2507.18553v2](https://arxiv.org/html/2507.18553v2)  
53. Chapter 18 \- Algorithms for the Closest and Shortest Vector Problems, accessed October 6, 2025, [https://www.math.auckland.ac.nz/\~sgal018/crypto-book/ch18.pdf](https://www.math.auckland.ac.nz/~sgal018/crypto-book/ch18.pdf)  
54. Optimizing Privacy-Preserving Primitives to Support LLM-Scale Applications \- arXiv, accessed October 6, 2025, [https://arxiv.org/html/2509.25072v1](https://arxiv.org/html/2509.25072v1)  
55. Privacy-Preserving Large Language Models: Mechanisms, Applications, and Future Directions \- arXiv, accessed October 6, 2025, [https://arxiv.org/html/2412.06113v1](https://arxiv.org/html/2412.06113v1)  
56. Cape: Context-Aware Prompt Perturbation Mechanism with Differential Privacy \- ICML 2025, accessed October 6, 2025, [https://icml.cc/virtual/2025/poster/44690](https://icml.cc/virtual/2025/poster/44690)  
57. ICML Randomized Quantization is All You Need for Differential ..., accessed October 6, 2025, [https://icml.cc/virtual/2023/27065](https://icml.cc/virtual/2023/27065)