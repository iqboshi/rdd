# RDD 持久上下文（固定模板）

## 0. 文档元信息
- 最后更新时间: 2026-03-19（A8-LTO 提速优化已落地，epoch 时间已显著下降）
- 维护规则:
  - 每次代码变更后必须更新本文件中的 `4/7/8` 章节。
  - 每次实验（训练/评测/盲评）后必须追加到 `documents/RDD_EXPERIMENT_LOG.md`。
  - 文档风格固定为本模板，不随会话切换而改变。
- Obsidian库路径(用户指定): `C:\Users\23581\Documents\some_scientific_ideas\rdd`

## 1. 项目目标与任务定义
- 任务: 复杂冠层叶片实例分割。
- 数据特点: 叶片细长、遮挡严重、截断常见、相邻叶片紧贴。
- 关键难点:
  - 过分割: 同一叶片被拆成多个实例。
  - 欠分割: 多片相邻叶片被并成同一实例。

## 2. 用户硬约束（长期有效）
- 创新必须有依据且效果明显，拒绝“为了创新而创新”。
- 评估不只看指标，还要看可视化；但只有出现明显突破才请用户打分。
- 训练默认使用 CUDA，并保留进度条可见。
- 训练轮数通常不少于 50 epochs。
- 使用环境: `D:\py_project\rdd-3-6`（缺包先询问用户安装）。

## 3. 固定基线与版本锚点
- 固定对比基线: `outputs/review_pack_aux_a7_topology_q128_s2_lowthr_split_20260318`
- 固定基线checkpoint: `outputs/ablation_aux_a7_topology_q128/exp_20260318_120451/checkpoints/best.pth`
- 旧基线（保留对照）:
  - `outputs/review_pack_aux_a4_q96_r3_20260318`
  - `outputs/ablation_aux_a4_q96/exp_20260317_121106/checkpoints/best.pth`
- A5.2 全程50轮真 best:
  - checkpoint: `outputs/ablation_aux_a5_2_q128/exp_20260317_213902/checkpoints/best.pth`
  - 别名: `outputs/ablation_aux_a5_2_q128/best_global_50ep_epoch49.pth`
  - 元信息: `outputs/ablation_aux_a5_2_q128/global_best_50ep_meta.json`

## 4. 当前创新路线（A6/A7/A8）
- 目标: 同时降低
  - `split`（同叶片被拆分）
  - `merge`（多叶片被合并）
- 方案骨架:
  - A6-P1: 后处理双向拓扑约束
    - 基于中心峰+offset投票的“过合并实例拆分”（防欠分割）
    - 保留A4/A5已有中心拼接策略（防过分割）
  - A6-P2: 评测门控升级
    - 在原有 MAE/split 基础上新增 merge 指标门控
  - A6-P3: 训练稳定性
    - 使用可恢复分段训练脚本，避免长训练中断
  - A6-P4: 训练阶段创新（架构+损失）
    - 新增 `separation head`（边界分离预测）
    - 新增 `separation loss`（边界监督）
    - 新增 `repulsion loss`（相邻实例中心投票排斥约束）
- 当前落地状态:
  - `scripts/build_review_pack.py` 已加入 A6 split + merge 指标。
  - `predict-v1.py` 已同步 A6 split 逻辑，保证评测/推理一致。
  - `scripts/gate_review_candidate.py` 已支持 merge 相关门控阈值。
  - `scripts/run_resumable_train.py` 已新增（分段续训、自动resume）。
  - `model.py` 与 `train-v5.py` 已加入 A6 训练阶段改动并通过 smoke 检查。
  - A6 50轮训练完成:
    - 实验目录: `outputs/ablation_aux_a6_archloss_q128`
    - 全程最优: `outputs/ablation_aux_a6_archloss_q128/exp_20260318_090429/checkpoints/best.pth`
    - best_val: `1.701828`（epoch 45）
  - A7（训练侧防过分裂）已完成首轮正式训练与评测:
    - 新增 `Boundary Conflict Head` + conflict loss（边界冲突监督）
    - 新增 `Leaf Affinity Embedding Head` + discriminative affinity loss（同叶片跨遮挡一致性）
    - 新增 `Mutual-Exclusion Overlap Loss`（抑制实例掩码重叠污染/混色）
    - 代码位置: `model.py`, `train-v5.py`
    - smoke验证: `outputs/a7_smoke/exp_20260318_103308`（CUDA，新增损失均正常回传）
    - 正式50轮目录: `outputs/ablation_aux_a7_topology_q128`
    - 全程最优: `outputs/ablation_aux_a7_topology_q128/exp_20260318_120451/checkpoints/best.pth`
    - best_val: `1.593240447666334`（epoch 44）
    - 首轮后处理评测（plain/s1/s2）:
      - `outputs/review_pack_aux_a7_topology_q128_plain_20260318`
      - `outputs/review_pack_aux_a7_topology_q128_s1_split_default_20260318`
      - `outputs/review_pack_aux_a7_topology_q128_s2_lowthr_split_20260318`
    - 门控结论: 三版均未通过（最佳 s2 仅 `rel_improve=2.03%`, `dense_abs_improve=+1.8333`，且 merge 指标负向）
    - 盲评复核（A7-s2 vs A4-r3）:
      - 盲评包: `outputs/review_pack_blind_ab_a7s2_vs_a4r3_20260318`
      - 解码结果: A7-s2 `14/24`（58.33%）vs A4-r3 `10/24`（41.67%）
      - 桶级差异: boundary/dense 明显偏向 A7-s2；highlight 与 scale_anchor_1024 偏向 A4-r3
  - 训练工程化（2026-03-18）:
    - 已将 `train-v5.py` 中 loss/匹配器/aux目标构建逻辑拆分到 `losses_v5.py`
    - `train-v5.py` 改为通过 `from losses_v5 import ...` 统一调用
    - loss 相关 CLI 参数注册集中到 `add_loss_args(parser)`，便于统一维护默认值并减少每次训练命令输入
    - `losses_v5.py` 默认 loss 参数已对齐 A7 基线常用配置（含 `enable_aux_heads=True`）
  - A8（论文创新主线，2026-03-19 启动）:
    - 核心思想: `Leaf Topology Ordering (LTO)`，显式学习并利用叶片上下拓扑顺序
    - 训练侧改动:
      - `model.py`: 新增 query 级顺序头 `pred_order`
      - `losses_v5.py`: 新增 `compute_order_consistency_loss(...)`
      - `train-v5.py`: 新增 `loss_order` 集成与日志/CSV字段
      - 新增参数: `w_order`, `order_min_dy`, `order_pair_max_dist`, `order_max_pairs`
    - 解码侧改动:
      - `scripts/build_review_pack.py`: 新增 `--enable_lto`, `--lto_order_weight`, `--lto_overlap_min_area`
      - 若 checkpoint 不含 `order head` 权重，自动回退到 score-only 解码并告警
    - 训练恢复兼容修复:
      - 场景: 从 A7 checkpoint 恢复到 A8 时，optimizer param-group 不匹配
      - 策略: 仅加载模型权重；`optimizer/scheduler/scaler` 不兼容则自动 fresh，不中断
      - 启动日志关键字段: `optimizer_loaded`, `scheduler_loaded`, `scaler_loaded`
    - 当前训练状态:
      - 目录: `outputs/ablation_aux_a8_lto_v1_q128`
      - 启动实验: `exp_20260319_102747`
      - 推荐总轮数: `--epochs 95`（从 `start_epoch=45` 继续，约再跑 50 轮）
      - 说明: 恢复阶段出现的两条 `lr_scheduler` warning 为一次性提示，不影响训练有效性
    - 训练提速优化（2026-03-19）:
      - 瓶颈定位结论（实时采样）:
        - GPU 长时间 `97%~100%`，显存约 `7776/8188MiB`，`P0`
        - CPU 总占用约 `11%~17%`（24线程），磁盘队列低（`<0.04`）
        - 结论: 主要是计算/实现开销瓶颈，不是 dataloader 喂不饱（`num_workers=4,prefetch=2` 不是主因）
      - 等价代码级优化（不改训练目标）:
        - `losses_v5.py`
          - 向量化 `build_order_targets_for_batch`（去逐实例 `torch.where`）
          - 向量化 `compute_affinity_embedding_loss`（`flatten + unique + index_add_`）
          - 移除 `order/overlap` 中不必要 `.item()` 同步点
          - 重写 `HungarianMatcher` mask/dice cost 计算，去除 `QxNxHxW` 大张量展开
        - `train-v5.py`
          - 逐 step loss 统计改为 GPU 端累计，epoch 末再一次性取值
          - 新增 `--tqdm_postfix_interval`（默认 20）降低进度条同步开销
          - 验证阶段改 `torch.inference_mode()`
          - 新增 runtime 加速开关: `--cudnn_benchmark`（默认开）、`--allow_tf32`（默认开）
          - 优化器自动尝试 `AdamW(fused=True)`，不支持则回退
      - 验证结果:
        - `python -m py_compile losses_v5.py train-v5.py` 通过
        - `dl40` 环境下 matcher/loss smoke 通过
      - 用户实测: epoch 约 `20分钟 -> 5分钟`
      - 说明: 训练逻辑与损失定义不变；`TF32/fused` 会带来轻微数值差异（非算法变更）

## 5. 关键命令模板
- 训练（cuda + 进度条）:
  - `D:\anaconda\envs\dl40\python.exe train-v5.py --device cuda ...`
- A8-LTO 训练（从 A7 best 续跑，推荐 95 轮）:
  - `D:\anaconda\envs\dl40\python.exe train-v5.py --save_dir outputs\ablation_aux_a8_lto_v1_q128 --epochs 95 --resume outputs\ablation_aux_a7_topology_q128\exp_20260318_120451\checkpoints\best.pth --pretrained_backbone_path pretrain_riceseg\outputs\exp_20260307_124458\checkpoints\best_backbone_for_instance.pth --amp <其余A7/A8参数>`
- 评测包构建:
  - `D:\anaconda\envs\dl40\python.exe scripts/build_review_pack.py --checkpoint <ckpt> --device cuda --seed 20260316 --save_dir <out>`
- A8-LTO 解码评测（训练完成后）:
  - `D:\anaconda\envs\dl40\python.exe scripts/build_review_pack.py --checkpoint <a8_best_ckpt> --device cuda --enable_lto --lto_order_weight <w> --lto_overlap_min_area <n> --save_dir <out>`
- 门控:
  - `D:\anaconda\envs\dl40\python.exe scripts/gate_review_candidate.py --baseline_summary <a4_csv> --candidate_summary <cand_csv> ...`
- 可恢复分段训练（推荐长跑）:
  - `D:\anaconda\envs\dl40\python.exe scripts/run_resumable_train.py --save_dir outputs\ablation_aux_a6_train --total_epochs 50 --chunk_size 10 <其余train-v5参数>`

## 6. 风险与预案
- 风险: 长训练因进程/显存/分页异常中断。
- 预案:
  - 分段续训（chunk）+ 自动 resume latest checkpoint。
  - 每段结束校验 `latest.pth` 与 `best.pth`。
  - 评测统一使用“全程真 best checkpoint”。

## 7. 最新结论（截至 2026-03-19）
- A5.2 在“全程真 best”下未达明显突破门槛。
- A6（架构+损失）训练后，在同协议评测中出现两类结果:
  - 保守后处理（plain/s1/s2）: 整体退化或提升有限。
  - 激进拆分（s3/s4/s5）: 计数 MAE 大幅下降（最优约 `10.25 -> 5.29`），dense 改善显著。
- 冲突验证（已完成盲评解码）:
  - 自动指标显示 s4 大幅提升，但盲评结果显示主观质量明显劣于基线。
  - 盲评统计: A4-r3 胜 `18/24`（75%），A6-s4 胜 `6/24`（25%）。
  - 其中 dense 桶为 `A4-r3 6/6` 全胜，说明 s4 的“计数提升”伴随不可接受视觉副作用。
- 阶段结论（A6阶段）: 不采纳 A6-s4；当时保持 A4-r3 为固定基线。
- A7 现状:
  - 已完成正式 50epoch 训练，全程最优为 epoch44（`best_val=1.593240447666334`）。
  - 已完成 plain/s1/s2 三版后处理门控评测，均未通过。
  - 当前最佳为 s2（`MAE=10.0417`, `rel_improve=2.03%`, `dense_abs_improve=+1.8333`），但 merge 相关指标仍负向。
  - 已完成 A7-s2 vs A4-r3 盲评解码，A7-s2 胜出（`14/24`）。
  - 已正式切换固定基线到 A7-s2（保留 A4-r3 作为历史对照锚点）。
  - 训练脚本已完成 loss 模块化，后续 loss 调整优先在 `losses_v5.py` 进行。
- A8-LTO 现状:
  - 创新代码已落地（模型顺序头 + 顺序损失 + LTO解码开关）。
  - 已修复 A7->A8 resume 的 optimizer group 不匹配崩溃问题（改为兼容降级，不中断训练）。
  - 启动时出现 `Resume optimizer state incompatible...` 属预期，不是失败。
  - 恢复阶段两条 `lr_scheduler` warning 属一次性提示，可继续训练。
  - 已完成一轮等价代码级加速（matcher/loss向量化 + 同步点削减 + CUDA runtime优化）。
  - 用户反馈训练速度显著提升（约 `20分钟/epoch -> 5分钟/epoch`）。
  - 当前处于正式训练进行中阶段；收敛后需先过门控，再决定是否触发盲评。
- 相关文件:
  - 本地: `outputs/review_pack_blind_ab_a6s4_vs_a4r3_20260318`
  - Obsidian: `C:\Users\23581\Documents\some_scientific_ideas\rdd\review_pack_blind_ab_a6s4_vs_a4r3_20260318`
  - 解码汇总: `blind_decode_summary.json`
  - 解码明细: `blind_decode_detailed.csv`
  - A7-s2盲评包: `outputs/review_pack_blind_ab_a7s2_vs_a4r3_20260318`
  - A7-s2解码汇总: `outputs/review_pack_blind_ab_a7s2_vs_a4r3_20260318/blind_decode_summary.json`
  - A7-s2解码明细: `outputs/review_pack_blind_ab_a7s2_vs_a4r3_20260318/blind_decode_detailed.csv`

## 8. 下一步 TODO（执行优先级）
1. 完成 A8-LTO 正式训练（目标 `--epochs 95`），记录全程真 best checkpoint 与 best_val。
2. 在当前提速配置下做一次 A/B 稳定性复核（固定seed，建议 100~300 step），确认无异常指标漂移。
3. 基于 A8 best 生成两套评测包并对比:
   - score-only（不启用LTO）
   - LTO decode（`--enable_lto` + order weight）
4. 使用 `scripts/gate_review_candidate.py` 对 A8 候选与 A7-s2 固定基线做门控。
5. 若出现“指标显著提升 + 门控通过”，生成盲评包并同步到 Obsidian。
6. 若 A8 对 highlight/scale_anchor_1024 仍回退，进入 LTO 权重与 overlap 规则小范围扫参。
