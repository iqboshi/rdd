# RDD 持久上下文（固定模板）

## 0. 文档元信息
- 最后更新时间: 2026-03-18（已切换固定基线为 A7-s2）
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

## 4. 当前创新路线（A6/A7）
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

## 5. 关键命令模板
- 训练（cuda + 进度条）:
  - `D:\anaconda\envs\dl40\python.exe train-v5.py --device cuda ...`
- 评测包构建:
  - `D:\anaconda\envs\dl40\python.exe scripts/build_review_pack.py --checkpoint <ckpt> --device cuda --seed 20260316 --save_dir <out>`
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

## 7. 最新结论（截至 2026-03-18）
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
- 相关文件:
  - 本地: `outputs/review_pack_blind_ab_a6s4_vs_a4r3_20260318`
  - Obsidian: `C:\Users\23581\Documents\some_scientific_ideas\rdd\review_pack_blind_ab_a6s4_vs_a4r3_20260318`
  - 解码汇总: `blind_decode_summary.json`
  - 解码明细: `blind_decode_detailed.csv`
  - A7-s2盲评包: `outputs/review_pack_blind_ab_a7s2_vs_a4r3_20260318`
  - A7-s2解码汇总: `outputs/review_pack_blind_ab_a7s2_vs_a4r3_20260318/blind_decode_summary.json`
  - A7-s2解码明细: `outputs/review_pack_blind_ab_a7s2_vs_a4r3_20260318/blind_decode_detailed.csv`

## 8. 下一步 TODO（执行优先级）
1. 冻结新基线：`A7-s2` 评测协议与 checkpoint（作为后续所有候选统一对照）。
2. 在 `losses_v5.py` 统一维护 loss 默认参数；训练命令优先只传必要增量参数。
3. 在新基线上做定向优化：重点补齐 highlight/scale_anchor_1024 的回退问题。
4. 维持门控 + 盲评双重机制，避免仅靠单一指标做错误采纳。
5. 仅当出现新一轮“指标显著提升 + 门控通过”时，触发下一次人工盲评。
