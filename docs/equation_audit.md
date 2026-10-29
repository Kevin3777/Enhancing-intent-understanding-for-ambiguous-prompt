# VCA PDF公式审计

唯一方法依据：paper/main.pdf，19页，物理页码与印刷页码一致。论文SHA256见equations.json。
原文文本已完整读取，Eq1-33所处p4-8已目视核对图像。每项先忠实转写，再审查，未静默修正。
当前执行原则与术语见execution_principles.md。README仅留档，不是方法来源。
代码映射不等于全部公式已精确复现；运行证据见experiments.md和milestones.json。

## Eq.(1) / p4

原式（paper-faithful 转写）：

$$
y_{t+1}=M(p_{refine} \parallel x \parallel y_0 \parallel fb_0 \parallel ... \parallel y_t \parallel fb_t)
$$

含义与输入输出：输入原始请求、历次输出/反馈与系统指令，输出下一轮改写。

审查及必要implementation-assumption：M/p_refine/x 的完整提示模板、上下文裁剪和停止条件未给出。Table 1(p9)仅规定 current_prompt,user_input -> new_prompt,type。

实现/边界：`sesd/dialogue.py::Dialogue.refine`；测试：`test_sesd.py::test_multiturn`；原审计标签：PAPER_AMBIGUOUS。

## Eq.(2) / p5

原式（paper-faithful 转写）：

$$
y_{t+1}=M(x \parallel y_t \parallel fb_t)
$$

含义与输入输出：只显式携带最近输出和反馈的简化递推。

审查及必要implementation-assumption：与Eq1的全历史输入不同；实现保留全历史，构造最近状态，标为工程选择，不能断言二者等价。

实现/边界：`sesd/dialogue.py::Dialogue.refine`；测试：`test_sesd.py::test_multiturn`；原审计标签：PAPER_AMBIGUOUS。

## Eq.(3) / p5

原式（paper-faithful 转写）：

$$
I(X;Y)=\int_x\int_y p(x,y)\log\frac{p(x,y)}{p(x)p(y)}\,dy\,dx
$$

含义与输入输出：连续随机变量互信息定义。

审查及必要implementation-assumption：需联合密度、可积性和绝对连续性。单图-文本cosine不估计这些密度；本式理论仅列示，不声称计算真实MI。

实现/边界：`scoring/mi.py (theory only)`；测试：`test_mi.py::test_not_clip_mi`；原审计标签：PAPER_EXACT。

## Eq.(4) / p5

原式（paper-faithful 转写）：

$$
CLIP\ Score=\mathbf f_I^\top\mathbf f_T=\cos(\theta)
$$

含义与输入输出：单位范数图像和文本嵌入的原始余弦。

审查及必要implementation-assumption：须同维、范数非零；不是logit_scale放大分数，也不是0-100 CLIPScore。论文未固定CLIP模型版本。

实现/边界：`scoring/clip_reward.py::CLIPReward`；测试：`test_reward.py::test_cosine`；原审计标签：PAPER_EXACT。

## Eq.(5) / p5

原式（paper-faithful 转写）：

$$
\Sigma_{xy}\approx\rho\mathbf u\mathbf u^\top
$$

含义与输入输出：交叉协方差秩一假设。

审查及必要implementation-assumption：一般X/Y维数可不同，需u_X u_Y^T；同一个u要求同维且同方向。需中心化；单位CLIP嵌入并不自动满足高斯/各向同性。

实现/边界：`scoring/mi.py (theory only)`；测试：`test_mi.py::test_gaussian_domain`；原审计标签：PAPER_AMBIGUOUS。

## Eq.(6) / p6

原式（paper-faithful 转写）：

$$
\rho=\frac1N\sum_{n=1}^{N}(\mathbf f_I^{(n)})^\top\mathbf f_T^{(n)}
$$

含义与输入输出：配对CLIP余弦的样本均值。

审查及必要implementation-assumption：未中心化的点积均值不是一般意义的交叉协方差特征值或相关系数；不能由此证明Eq5。

实现/边界：`scoring/clip_reward.py::cosine`；测试：`test_reward.py::test_cosine`；原审计标签：PAPER_AMBIGUOUS。

## Eq.(7) / p6

原式（paper-faithful 转写）：

$$
I(X;Y)=\frac12\log\left(\frac{\det(\Sigma_X)\det(\Sigma_Y)}{\det(\Sigma_Z)}\right)
$$

含义与输入输出：联合非退化高斯的互信息。

审查及必要implementation-assumption：协方差必须正定；归一化嵌入位于球面，一般不严格联合高斯。实现可用slogdet作数学验证，非经验CLIP到MI转换。

实现/边界：`scoring/mi.py::gaussian_mi`；测试：`test_mi.py::test_gaussian_domain`；原审计标签：PAPER_EXACT。

## Eq.(8) / p6

原式（paper-faithful 转写）：

$$
\det(\Sigma_Z)=\det(\Sigma_X)\det(\Sigma_Y-\Sigma_{YX}\Sigma_X^{-1}\Sigma_{XY})
$$

含义与输入输出：Schur补行列式。

审查及必要implementation-assumption：要求Sigma_X可逆，联合协方差块尺寸相容。工程计算用solve而非显式求逆。

实现/边界：`scoring/mi.py::gaussian_mi`；测试：`test_mi.py::test_gaussian_domain`；原审计标签：PAPER_EXACT。

## Eq.(9) / p6

原式（paper-faithful 转写）：

$$
\Sigma_X=\sigma_x^2\mathbf I,\quad\Sigma_Y=\sigma_y^2\mathbf I
$$

含义与输入输出：各向同性假设。

审查及必要implementation-assumption：不是CLIP预训练保证；本地不将该假设写成数据事实。

实现/边界：`scoring/mi.py (theory only)`；测试：`test_mi.py::test_gaussian_domain`；原审计标签：PAPER_AMBIGUOUS。

## Eq.(10) / p6

原式（paper-faithful 转写）：

$$
\Sigma_{XY}\approx\rho\mathbf u\mathbf u^\top
$$

含义与输入输出：重复Eq5秩一近似。

审查及必要implementation-assumption：u范数未指定，C与sigma/u尺度有关；维度和中心化问题同Eq5。

实现/边界：`scoring/mi.py (theory only)`；测试：`test_mi.py::test_gaussian_domain`；原审计标签：PAPER_AMBIGUOUS。

## Eq.(11) / p6

原式（paper-faithful 转写）：

$$
\det(\Sigma_Z)\approx\sigma_x^{2D}\sigma_y^{2D}(1-C\rho^2)
$$

含义与输入输出：在同维D的各向同性、秩一条件下简化行列式。

审查及必要implementation-assumption：需C=||u||^4/(sigma_x^2 sigma_y^2)，且C*rho^2<1；原文仅说C>0，未交代尺度/正定域。

实现/边界：`scoring/mi.py::rank_one_proxy`；测试：`test_mi.py::test_gaussian_domain`；原审计标签：PAPER_AMBIGUOUS。

## Eq.(12) / p7

原式（paper-faithful 转写）：

$$
I(X;Y)\approx\frac12\log\left(\frac1{1-C\rho^2}\right)
$$

含义与输入输出：条件假设下的高斯MI近似。

审查及必要implementation-assumption：仅在C*rho^2<1有效；关于|rho|递增，关于负rho并非递增。单调性不能无条件推广为CLIP=MI。

实现/边界：`scoring/mi.py::rank_one_proxy`；测试：`test_mi.py::test_gaussian_domain`；原审计标签：PAPER_AMBIGUOUS。

## Eq.(13) / p7

原式（paper-faithful 转写）：

$$
\theta_{new}=\theta_{old}-\eta\nabla I(X;Y\mid f)
$$

含义与输入输出：字面为条件MI的梯度下降。

审查及必要implementation-assumption：PDF p7 Eq13减号与最大化叙述冲突。paper_literal保留减号；mi_ascent为独立显式implementation-assumption变体。未定义条件MI估计器，不能假造对生成器的真实MI梯度。

实现/边界：`scoring/mi.py::update`；测试：`test_mi.py::test_direction`；原审计标签：PAPER_AMBIGUOUS。

## Eq.(14) / p7

原式（paper-faithful 转写）：

$$
\mathcal R(\Theta)=CLIP(I_{gen},P_{prev})+\lambda CLIP(I_{gen},P_{new}),\quad\lambda=0.2
$$

含义与输入输出：同一生成图对历史/新提示的双重奖励。

审查及必要implementation-assumption：历史权重1，新权重0.2；与强调新意图的直觉不同但不得静默交换。new_priority显式用0.2*prev+new。人类评分不在此式，单列日志，默认不混入。

实现/边界：`scoring/clip_reward.py::dual_reward`；测试：`test_reward.py::test_weights`；原审计标签：PAPER_EXACT。

## Eq.(15) / p7

原式（paper-faithful 转写）：

$$
\Theta^{(k+1)}=\Theta^{(k)}+\alpha\nabla_\Theta\mathcal R(\Theta^{(k)})
$$

含义与输入输出：控制器参数梯度上升。

审查及必要implementation-assumption：未解释离散动作、PIL图像和采样步骤的可微路径；本实现使用PPO score-function更新策略，不能称本式逐像素直接梯度精确复现。

实现/边界：`rl/ppo.py::PPO.update`；测试：`test_ppo.py::test_updates_actor_critic`；原审计标签：PAPER_AMBIGUOUS。

## Eq.(16) / p7

原式（paper-faithful 转写）：

$$
I_{gen}^{(k+1)}=Upd(I_{gen}^{(k)},\Theta^{(k)})
$$

含义与输入输出：图像迭代更新抽象算子。

审查及必要implementation-assumption：Upd未定义，未给真实图像反演方法。实现从相同seed/latents重采样双分支，不声称对任意原图反演。

实现/边界：`editing/pipeline_adapter.py::PairDiffusion.generate`；测试：`scripts/test_attention_effect.py`；原审计标签：PAPER_AMBIGUOUS。

## Eq.(17) / p7

原式（paper-faithful 转写）：

$$
J(\theta)=\mathbb E_{\pi_\theta}\left[\sum_{t=0}^{T}\gamma^t r_t\right]
$$

含义与输入输出：最大化折扣累积奖励。

审查及必要implementation-assumption：此段说r_i=CLIP(I_i,P_i)，与Eq14双奖励不完全相同。默认Eq14，配置/日志记录。状态编码器和T未给，采用明确工程MDP。

实现/边界：`rl/state_action.py; runtime/pipeline.py`；测试：`test_ppo.py; test_runtime.py`；原审计标签：PAPER_AMBIGUOUS。

## Eq.(18) / p7

原式（paper-faithful 转写）：

$$
\delta_t=r_t+\gamma V(s_{t+1},\theta)-V(s_t,\theta)
$$

含义与输入输出：一步TD误差。

审查及必要implementation-assumption：原式省略终止mask；实现r+gamma*(1-done)*Vnext-V，critic target detach；不默认GAE。时间上限作为截断与任务终止分别记录。

实现/边界：`rl/rollout.py::td_targets`；测试：`test_ppo.py::test_done_mask`；原审计标签：PAPER_EXACT。

## Eq.(19) / p7

原式（paper-faithful 转写）：

$$
p_i\propto e^{-\lambda t_i}\cdot|\delta_i|+\epsilon
$$

含义与输入输出：按经验年龄衰减和绝对TD误差的优先采样。

审查及必要implementation-assumption：训练实际使用q_i=exp(-lambda*age_i)*abs(TD_i)+eps，p_i=q_i/sum(q)，w_i=(N*p_i)^(-beta)。每个epoch刷新TD优先级，保存behavior old_logprob与版本。PDF未说明陈旧轨迹与PPO兼容；仅同版本rollout参与PPO重采样，历史pool只供诊断，标implementation-assumption。

实现/边界：`rl/replay.py::DiagnosticReplay; rl/ppo.py::PPO.update`；测试：`test_replay.py`；原审计标签：PAPER_AMBIGUOUS。

## Eq.(20) / p7

原式（paper-faithful 转写）：

$$
\mathcal L^{PPO}(\theta)=\mathbb E_t[\min(\rho_t(\theta)\hat A_t,\operatorname{clip}(\rho_t(\theta),1-\epsilon,1+\epsilon)\hat A_t)]
$$

含义与输入输出：clipped PPO surrogate；rho=pi_new/pi_old；A为一步TD。

审查及必要implementation-assumption：以old_logprob.detach、exp(new-old)、clipped surrogate、critic bootstrap、done mask、KL/entropy监测和optimizer.step实施。按PDF p8保留2.2/0.3数值，分别映射到value MSE与行为策略KL是implementation-assumption。没有额外entropy优化目标；默认一步TD，不用GAE。

实现/边界：`rl/ppo.py::PPO.update`；测试：`test_ppo.py`；原审计标签：PAPER_EXACT。

## Eq.(21) / p8

原式（paper-faithful 转写）：

$$
Edit(M_t,M_t^*,t):=\begin{cases}M_t^*,&t<\tau\\M_t,&otherwise\end{cases}
$$

含义与输入输出：阈值内替换注意力。

审查及必要implementation-assumption：PDF p8称M*为更新/梯度细化图，同时要求保留整体结构；未唯一规定来源。保留literal目标图解释与base_to_edited_assumption来源分支注入路径。后者仅是对PDF保结构意图的实现假设，外部代码不作为方法定义。采样进度索引和原始扩散t均记录。

实现/边界：`editing/controllers.py::AttentionController`；测试：`test_attention.py; scripts/test_attention_effect.py`；原审计标签：PAPER_AMBIGUOUS。

## Eq.(22) / p8

原式（paper-faithful 转写）：

$$
M_{new}=(M+\eta\cdot\Delta M)\cdot M_{cross\_attention}
$$

含义与输入输出：更新映射矩阵后乘cross-attention。

审查及必要implementation-assumption：M/mapper二义，DeltaM来源、矩阵维度和归一化缺失。实现显式[Ks,Kt]映射及einsum，映射固定、非论文所称梯度训练；可训练mapper NOT_REPRODUCED。

实现/边界：`editing/token_alignment.py::align`；测试：`test_alignment.py::test_mapping_shape`；原审计标签：PAPER_AMBIGUOUS。

## Eq.(23) / p8

原式（paper-faithful 转写）：

$$
(Edit(M_t,M_t^*,t))_{i,j}:=\begin{cases}(M_t^*)_{i,j},&A(j)=None\\(M_t)_{i,A(j)},&otherwise\end{cases}
$$

含义与输入输出：新插入token用目标图，匹配token用来源图。

审查及必要implementation-assumption：A是离散token对应，如何梯度上升未定义。必须有base/edited两分支；subword对齐、重复词、删除/插入和特殊token需处理。安全路径对总token维重归一化；标记为工程选择。

实现/边界：`editing/controllers.py; editing/token_alignment.py`；测试：`test_alignment.py; test_attention.py`；原审计标签：PAPER_AMBIGUOUS。

## Eq.(24) / p8

原式（paper-faithful 转写）：

$$
\theta_m^\prime=\theta_m+\eta\nabla_{\theta_m}\mathbb E[R]
$$

含义与输入输出：mapper参数奖励梯度上升。

审查及必要implementation-assumption：参数化、动作采样、梯度估计器、训练目标未给；离散token mapper固定对齐不等于本式训练。NOT_REPRODUCED，PPO只训练明确的actor/critic和控制参数分布。

实现/边界：`docs/reproduction_scope.md (not reproduced)`；测试：`test_alignment.py`；原审计标签：PAPER_AMBIGUOUS。

## Eq.(25) / p8

原式（paper-faithful 转写）：

$$
\theta_\alpha^\prime=\theta_\alpha+\eta\nabla_{\theta_\alpha}\mathbb E[R]
$$

含义与输入输出：混合系数参数更新。

审查及必要implementation-assumption：未说明与Eq27 beta、Eq31 alpha的关系。安全实现策略分布选择离散strength/schedule，属于score-function工程近似，不冒充mapper/alpha直接微分。

实现/边界：`rl/state_action.py; rl/policy.py`；测试：`test_policy.py`；原审计标签：PAPER_AMBIGUOUS。

## Eq.(26) / p8

原式（paper-faithful 转写）：

$$
attn\_base\_replace_{ijk}=attn\_base_{ijk}\cdot mapper_{kj}\quad\Longrightarrow\quad(attn\_base\_replace)_{permute(2,0,1,3)}
$$

含义与输入输出：原文以三指标乘法后permute四个维度。

审查及必要implementation-assumption：三维/四维矛盾、k同时为空间或token不明、无显式求和。保留此原式。工程明确S[b,h,q,k], map[k,j], mapped[b,h,q,j]=sum_k S*map；einsum(bhqk,kj->bhqj)。不执行不合法permute。

实现/边界：`editing/token_alignment.py::TokenAlignment.project`；测试：`test_alignment.py::test_einsum`；原审计标签：PAPER_AMBIGUOUS。

## Eq.(27) / p8

原式（paper-faithful 转写）：

$$
M_{update}^{(t)}=\beta_t M_{orig}^{(t)}+(1-\beta_t)M_{new}^{(t)}
$$

含义与输入输出：映射后来源/目标attention逐层逐步混合。

审查及必要implementation-assumption：orig先对齐到目标token轴；beta约束[0,1]未给，工程加界；新token保留目标。同维且各自为概率分布时凸组合才保行和。

实现/边界：`editing/controllers.py`；测试：`test_attention.py::test_identity`；原审计标签：PAPER_AMBIGUOUS。

## Eq.(28) / p8

原式（paper-faithful 转写）：

$$
(Edit(M_t,M_{t+1},t))_{i,j}:=\begin{cases}c M_t(i,j),&j=j^*\\M_t(i,j),&otherwise\end{cases},\quad c\in[-2,2]
$$

含义与输入输出：指定token后softmax权重缩放。

审查及必要implementation-assumption：c<0破坏非负性，c!=1一般破坏行和。paper_literal保留算术并显式验证失败；safe_attention用log-prob/logit bias + softmax支持[-2,2]且标RESEARCH_VARIANT，绝不用clamp掩盖差异。

实现/边界：`editing/controllers.py::literal_reweight; safe_reweight`；测试：`test_attention.py::test_negative_literal`；原审计标签：PAPER_AMBIGUOUS。

## Eq.(29) / p8

原式（paper-faithful 转写）：

$$
M_{refine}^{(t)}=\beta_t M_{orig}^{(t)}+(1-\beta_t)M_{new}^{(t)},\quad\beta_t=\beta_{t-1}+\gamma\nabla_{\beta_t}\mathcal R(\theta)
$$

含义与输入输出：先混合，再用奖励梯度更新beta。

审查及必要implementation-assumption：右侧对beta_t求导但左侧也beta_t，隐式/更新时序不清；gamma也用于discount。混合可实现，直接CLIP图像梯度未复现；PPO action beta为工程替代。

实现/边界：`editing/controllers.py; rl/policy.py`；测试：`test_attention.py; test_policy.py`；原审计标签：PAPER_AMBIGUOUS。

## Eq.(30) / p8

原式（paper-faithful 转写）：

$$
M_{reweight}^{(t)}=\sum_i\gamma_{t,i}M_{refine}^{(t,i)},\quad\gamma_{t,i}=\gamma_{t-1,i}+\kappa\nabla_{\gamma_{t,i}}\mathcal R(\theta)
$$

含义与输入输出：按feature权重组合attention。

审查及必要implementation-assumption：i是token/head/feature未定义，约束、归一化及梯度路径缺失。工程仅对token轴施加显式bias并归一化，不宣称逐项实现原和式/梯度。

实现/边界：`editing/controllers.py; docs/reproduction_scope.md`；测试：`test_attention.py`；原审计标签：PAPER_AMBIGUOUS。

## Eq.(31) / p8

原式（paper-faithful 转写）：

$$
\alpha_{t+1}=\alpha_t+\eta\nabla_{\alpha_t}\mathcal R(\theta)
$$

含义与输入输出：自注意力注入比例更新。

审查及必要implementation-assumption：注入层、时段、比例边界未给。工程在小空间self-attention层按[0,1]比例注入base分支，限制Q<=256避免显存溢出；参数来自策略离散分布。

实现/边界：`editing/controllers.py; rl/state_action.py`；测试：`test_attention.py::test_self_injection`；原审计标签：PAPER_AMBIGUOUS。

## Eq.(32) / p8

原式（paper-faithful 转写）：

$$
\zeta_{t+1}=\zeta_t+\gamma\nabla_{\zeta_t}\mathcal R(\theta)
$$

含义与输入输出：cross-attention替换比例更新。

审查及必要implementation-assumption：以0..1进度门控和混合强度工程映射；策略分布间接优化，非直接对PIL反传；与Eq25/27同名不同义风险记入配置。

实现/边界：`editing/controllers.py; rl/state_action.py`；测试：`test_attention.py`；原审计标签：PAPER_AMBIGUOUS。

## Eq.(33) / p8

原式（paper-faithful 转写）：

$$
\delta_{t+1}=\delta_t+\kappa\nabla_{\delta_t}\mathcal R(\theta)
$$

含义与输入输出：采样步数更新。

审查及必要implementation-assumption：步数为整数，不能普通微分；delta又与TD误差同名。工程将steps作为Categorical动作选取固定集合，保存joint logprob；这是ENGINEERING_CHOICE，非字面梯度。

实现/边界：`rl/state_action.py; rl/policy.py`；测试：`test_policy.py`；原审计标签：PAPER_AMBIGUOUS。
