# 未决问题与最小假设

|问题/原文位置|最小假设及实现边界|置信度/标签|
|---|---|---|
|SESD的ChatGPT-4具体版本、微调权重/提示(p4-6,p9)|保留LLMClient.generate接口；默认无密钥确定性规则mock，provider返回JSON后校验；不称复现原LLM|低 / ENGINEERING_CHOICE|
|Actor/Critic网络与state输入(p7)|frozen CLIP图像/文本投影或明确toy状态；actor/critic各自MLP，不更新diffusion|低 / ENGINEERING_CHOICE|
|用户决定与policy决定(p6-7)|用户/SESD给语义编辑提案，policy可选算子/strength/schedule/self injection/整数steps；记录二者来源|中 / ENGINEERING_CHOICE|
|连续控制与log_prob(p8)|有限Categorical联合动作，条件mask对用户锁定动作；rollout保存mask，重算同mask概率|低 / ENGINEERING_CHOICE|
|Eq13条件MI/方向|不估计真实MI；数学函数提供literal和mi_ascent，CLIP仅proxy|高(冲突存在) / PAPER_AMBIGUOUS|
|Eq14与Algorithm1权重次序|默认Eq14 prev+0.2new；Algorithm1 lambda1/2未给值；new_priority显式变体|高 / PAPER_EXACT + RESEARCH_VARIANT|
|Eq19 replay与on-policy|同版本优先回放实际参与PPO训练，旧版本被拒绝；存logprob、policy_version、action mask、seed与latents|高(需保护) / ENGINEERING_CHOICE|
|critic/value/KL reward coefficients 2.2/0.3(p8)|记录原参数但不擅自断定完整loss；当前value_coef=2.2、entropy=0、KL=0.3；数值按PDF，映射语义为implementation-assumption，配置独立可改|低 / ENGINEERING_CHOICE|
|Diffusion backbone与Efficient U-Net(p8)|选择公开SD1.5镜像固定revision，DDIM冻结；这不是Efficient U-Net训练复现|低 / NOT_REPRODUCED|
|CLIP版本/标定(p5)|openai/clip-vit-base-patch32固定revision，单位向量raw cosine，非logits|低 / ENGINEERING_CHOICE|
|Eq21 M/M*来源|literal目标图与base_to_edited_assumption来源图路径分开；后者为PDF保结构意图的实现假设|低 / PAPER_AMBIGUOUS|
|Eq22/24 mapper训练|token动态对齐矩阵固定，无论文未公开的mapper训练器；不将此项标完成|低 / NOT_REPRODUCED|
|Eq26形状|S[B,H,Q,Ks]，map[Ks,Kt]，einsum->T[B,H,Q,Kt]；padding不丢弃原attention质量|高(形状契约) / ENGINEERING_CHOICE|
|Eq28负权重|literal显式抛验证错误；safe logit bias[-2,2]，softmax重归一化|高(概率冲突) / RESEARCH_VARIANT|
|Eq31-33机制|离散self注入/跨注意力比例/采样步集合，PPO score-function；自注意力只对Q<=256注入|低 / ENGINEERING_CHOICE|
|Algorithm1 H生命周期|每轮append图/提示/reward/action/evidence；更新current；按raw_new或显式utility选best；保留会话回退和恢复|高(必要但原文遗漏) / ENGINEERING_CHOICE|
|stop(p6-7)|max rounds、用户满意、raw_new>=threshold；threshold无原值，配置记录；空history不得SelectBest|中 / ENGINEERING_CHOICE|
|知识动态更新(p6)|仅存本次session带source的记录，不把生成结果直接当外部事实；可重建索引|中 / ENGINEERING_CHOICE|
|原数据/split/人类评审(p8-9)|三个链接独立审计；无可核对1673 JSON则标BLOCKED，不从README示例捏造调查|高 / NOT_REPRODUCED|
|统计单元与绝对指标(p10-17)|固定样本/seed/模型；bootstrap区间仅对实际测量；用户满意度仅真实调查|高 / ENGINEERING_CHOICE|


## PDF内部进一步需披露的口径

- p5 Eq4定义单位嵌入余弦，数学范围[-1,1]；p10指标段却描述为[0,1]。本实现记录raw cosine，不私自重标定。
- p16 Table5说明文字将final CLIP/Aesthetic与主观0-5量表联系，但表中CLIP写0.92/1.0；不能据此把本地B32 raw cosine与其直接比数。
- p8-9公开数据段、p16未来释放/available on request表述并存。是否有1673 JSON必须独立审计文件，不能从文字推定已取得。
- p9的30位标注员、p16的100位PPO比较用户与81位SSIM任务用户是不同评估协议，不能混为本地样本数。
- Algorithm3命名Layout；附录A7-A11实际列Polysemy、Style、Enhancers、Viewpoint、Atmosphere，没有独立Layout表。本地Layout项是implementation-assumption补充。
