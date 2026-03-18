# RDD 实验日志（固定模板）

## 记录规范
- 每次实验新增一条，禁止覆盖历史记录。
- 统一字段:
  - 日期
  - 实验ID
  - 目标
  - 关键配置
  - 结果摘要
  - 结论（是否进入用户打分）

---

## 2026-03-17 | A4-fixed-baseline
- 实验ID: `ablation_aux_a4_q96`
- 目标: 固定强基线
- 关键配置: q=96 + aux heads + patch scale weighting
- 结果摘要: 作为后续对比固定基线
- 结论: 是（已完成盲评）

## 2026-03-17 | A5-vote-consistency
- 实验ID: `ablation_aux_a5_vote`
- 目标: 通过中心投票一致性降低过分割
- 关键配置: `w_vote_consistency=0.02`
- 结果摘要: 有小幅改进，但不明显
- 结论: 否（未达到“明显突破”门槛）

## 2026-03-17/18 | A5.2-q128
- 实验ID: `ablation_aux_a5_2_q128`
- 目标: 增强容量并稳住投票损失
- 关键配置: q=128, `w_vote_consistency=0.005`
- 结果摘要:
  - 50轮跨三段完成；全程最佳在 epoch49
  - 全程真best: `outputs/ablation_aux_a5_2_q128/exp_20260317_213902/checkpoints/best.pth`
- 结论: 否（门控未通过，不进入人工打分）

## 2026-03-18 | A6-kickoff
- 实验ID: `A6_dual_topology`
- 目标: 同时解决“过分割 + 欠分割”
- 关键配置: 待本次代码落地后填写
- 结果摘要: 进行中
- 结论: 待定

## 2026-03-18 | A6-postproc-r1（快速验证）
- 实验ID: `review_pack_aux_a6_split_plain_r1_20260318`
- 目标: 验证 center-split 对欠分割的修复力度
- 关键配置: split启用（峰值+offset投票拆分），不启用stitch
- 结果摘要:
  - A4_r3 MAE: 10.25
  - A6_plain MAE: 9.21（显著下降）
  - 但 split/merge 比例指标存在副作用，未通过完整门控
- 结论: 不进入人工打分，继续优化

## 2026-03-18 | A6-postproc-sweep（s1/s2/balanced/guarded）
- 实验ID:
  - `review_pack_aux_a6_split_s1_r1_20260318`
  - `review_pack_aux_a6_split_s2_mid_r1_20260318`
  - `review_pack_aux_a6_split_balanced_r1_20260318`
  - `review_pack_aux_a6_split_guarded_r2_20260318`
- 目标: 平衡 MAE 提升与 split/merge 副作用
- 关键配置: split参数与stitch参数小范围扫
- 结果摘要:
  - s1/s2: MAE 仍优于A4，但 split_ratio/merge_ratio 未达门控
  - guarded: 副作用降低但 MAE 反弹
- 结论: 继续进入 A6 训练阶段，暂不盲评

## 2026-03-18 | A6-postproc-sweep-v2（仅后处理，不训练）
- 实验ID:
  - `review_pack_aux_a6_sweep_c1_plain_default_20260318`
  - `review_pack_aux_a6_sweep_c2_plain_strict_20260318`
  - `review_pack_aux_a6_sweep_c3_split_lightstitch_20260318`
  - `review_pack_aux_a6_sweep_c4_strict_lightstitch_20260318`
  - `review_pack_aux_a6_sweep_e1_elong_plain_20260318`
  - `review_pack_aux_a6_sweep_e2_elong_lightstitch_20260318`
- 目标: 在不训练的前提下，仅靠后处理同时改善 MAE / split / merge
- 关键配置:
  - 新增 split 防误拆约束: `split_second_support_ratio`, `split_min_peak_separation`, `split_max_elongation`
  - 并扫了轻量 stitching 组合
- 结果摘要:
  - 最好 MAE 约 `10.00`（相对 A4_r3 的 `10.25` 提升有限）
  - 一旦 MAE 明显下降，通常伴随 split/merge 比例副作用
  - 严格约束虽能减少误拆，但会把 MAE 拉回甚至变差
- 结论: “仅后处理”已接近瓶颈，暂不触发人工打分
- 汇总文件: `outputs/a6_postproc_sweep_20260318_summary.csv`

## 2026-03-18 | A6-archloss-50ep（训练阶段创新）
- 实验ID: `ablation_aux_a6_archloss_q128`
- 目标: 在训练阶段同时约束“欠分割+过分割”
- 关键配置:
  - q=128, patch scale weighting
  - `w_vote_consistency=0.005`
  - `w_separation=0.08`（新增分离头监督）
  - `w_repulsion=0.015`（新增相邻实例排斥约束）
  - 使用 `scripts/run_resumable_train.py` 分段续训到 50 轮（CUDA）
- 结果摘要:
  - 50轮完成，最优在 epoch45
  - best checkpoint: `outputs/ablation_aux_a6_archloss_q128/exp_20260318_090429/checkpoints/best.pth`
  - best val loss: `1.701828`
- 结论: 进入训练后 checkpoint 的后处理评测阶段

## 2026-03-18 | A6-archloss-postproc-sweep（同一ckpt）
- 实验ID:
  - `review_pack_aux_a6_archloss_q128_plain_20260318`
  - `review_pack_aux_a6_archloss_q128_s1_split_default_20260318`
  - `review_pack_aux_a6_archloss_q128_s2_lowthr_split_20260318`
  - `review_pack_aux_a6_archloss_q128_s3_aggrsplit_20260318`
  - `review_pack_aux_a6_archloss_q128_s4_demerge_20260318`
  - `review_pack_aux_a6_archloss_q128_s5_demerge_stitch_20260318`
- 目标: 判断训练改进后，是否能出现“明显突破”且无严重副作用
- 关键配置:
  - 基线对比: `outputs/review_pack_aux_a4_q96_r3_20260318/review_summary.csv`
  - 门控脚本: `scripts/gate_review_candidate.py`（含 split/merge 条件）
- 结果摘要:
  - plain/s1/s2: 退化或提升有限（未通过门控）
  - s3/s4/s5: 计数MAE显著提升（最佳 s4: `10.25 -> 5.29`，dense 提升 `+9.0`）
  - 但 split/merge 代理指标仍负向（提示潜在过分裂/实例归属副作用）
- 结论: 达到“可视化显著变化”级别，触发一次人工盲评确认真实性能

## 2026-03-18 | Blind-AB（A6-s4 vs A4-r3）
- 实验ID: `review_pack_blind_ab_a6s4_vs_a4r3_20260318`
- 目标: 在指标冲突时由人工可视化判定真实收益
- 关键配置:
  - A: `outputs/review_pack_aux_a4_q96_r3_20260318`
  - B: `outputs/review_pack_aux_a6_archloss_q128_s4_demerge_20260318`
  - 盲评包输出: `outputs/review_pack_blind_ab_a6s4_vs_a4r3_20260318`
- 结果摘要:
  - 已生成 `BLIND_REVIEW_INDEX.md` 与 `blind_scoring_template.csv`
  - 已同步到 Obsidian:
    `C:\Users\23581\Documents\some_scientific_ideas\rdd\review_pack_blind_ab_a6s4_vs_a4r3_20260318`
- 结论: 等待用户填写盲评打分

## 2026-03-18 | Blind-AB-Decode（A6-s4 vs A4-r3）
- 实验ID: `review_pack_blind_ab_a6s4_vs_a4r3_20260318`
- 目标: 解码盲评，判断是否采纳 A6-s4
- 关键配置:
  - 评分表: `blind_scoring_template.csv`（用户已填写）
  - 映射表: `blind_mapping_private.csv`
  - 解码输出:
    - `blind_decode_summary.json`
    - `blind_decode_detailed.csv`
- 结果摘要:
  - 有效样本: 24/24
  - 胜出统计: `a4_r3_baseline=18` vs `a6_s4_candidate=6`
  - 胜率: A4-r3 `75%`，A6-s4 `25%`
  - 按桶:
    - dense: A4-r3 `6/6` 全胜
    - boundary: A4-r3 `4` vs A6-s4 `2`
    - highlight: A4-r3 `3` vs A6-s4 `3`
    - scale_anchor_1024: A4-r3 `2` vs A6-s4 `0`
    - scale_anchor_512: A4-r3 `2` vs A6-s4 `0`
    - scale_anchor_768: A4-r3 `1` vs A6-s4 `1`
- 结论:
  - 不采纳 A6-s4，维持 A4-r3 为固定基线
  - 说明: A6-s4 虽有显著计数指标提升，但主观可视化质量下降，不符合“效果明显且真实可用”的约束

## 2026-03-18 | A7-Module-Impl-Smoke
- 实验ID: `a7_module_impl_smoke`
- 目标: 在训练侧落地“防过分裂+防混色”模块并验证可训练性
- 关键配置:
  - 模型改动: `model.py`
    - 新增 `pred_conflict`（Boundary Conflict Head）
    - 新增 `pred_affinity`（Leaf Affinity Embedding Head）
  - 训练改动: `train-v5.py`
    - 新增 `loss_conflict`
    - 新增 `loss_affinity`
    - 新增 `loss_overlap_excl`（Mutual-Exclusion Overlap Loss）
    - 新增对应CLI参数、日志列与进度条显示
  - smoke命令: CUDA, `epochs=1`, `max_train_steps=2`, `max_val_steps=1`
- 结果摘要:
  - smoke目录: `outputs/a7_smoke/exp_20260318_103308`
  - 训练与验证均跑通，新增损失项 `cfl/aff/mex` 均有非零有效值
  - 无崩溃、无shape错误、checkpoint正常保存
- 结论: 通过，可进入A7正式50轮训练

## 2026-03-18 | A7-topology-50ep + postproc-gate-r1
- 实验ID:
  - `ablation_aux_a7_topology_q128`
  - `review_pack_aux_a7_topology_q128_plain_20260318`
  - `review_pack_aux_a7_topology_q128_s1_split_default_20260318`
  - `review_pack_aux_a7_topology_q128_s2_lowthr_split_20260318`
- 目标: 验证 A7 正式50轮后，是否在中等强度后处理下达到门控并触发盲评
- 关键配置:
  - 训练输出目录: `outputs/ablation_aux_a7_topology_q128`
  - 全程best: `outputs/ablation_aux_a7_topology_q128/exp_20260318_120451/checkpoints/best.pth`
  - 最优轮次: epoch44, `val_total=1.593240447666334`
  - 门控基线: `outputs/review_pack_aux_a4_q96_r3_20260318/review_summary.csv`
- 结果摘要:
  - plain: MAE `10.2083`, 相对提升 `0.41%`, dense提升 `+1.8333`，门控未通过
  - s1(default split): MAE `10.2083`, 相对提升 `0.41%`, dense提升 `+1.8333`，门控未通过
  - s2(lowthr split): MAE `10.0417`, 相对提升 `2.03%`, dense提升 `+1.8333`，门控未通过
  - 三版候选在 merge 相关指标上均为负向改进（`merge_mean_improve < 0`, `gt_merged_ratio_improve < 0`）
- 结论: 否（不触发人工盲评，进入 A7 定向优化阶段）

## 2026-03-18 | Blind-AB-Decode（A7-s2 vs A4-r3）
- 实验ID: `review_pack_blind_ab_a7s2_vs_a4r3_20260318`
- 目标: 在你主观偏好 A7-s2 的前提下，用盲评确认是否采纳 A7-s2
- 关键配置:
  - 评分表: `blind_scoring_template.csv`（用户已填写）
  - 映射表: `blind_mapping_private.csv`
  - 解码输出:
    - `blind_decode_summary.json`
    - `blind_decode_detailed.csv`
- 结果摘要:
  - 有效样本: 24/24
  - 胜出统计: `a7_s2_candidate=14` vs `a4_r3_baseline=10`
  - 胜率: A7-s2 `58.33%`，A4-r3 `41.67%`
  - 平均分: A7-s2 `2.9167`，A4-r3 `2.8750`
  - 按桶:
    - boundary: A7-s2 `5` vs A4-r3 `1`
    - dense: A7-s2 `5` vs A4-r3 `1`
    - highlight: A7-s2 `2` vs A4-r3 `4`
    - scale_anchor_1024: A7-s2 `0` vs A4-r3 `2`
    - scale_anchor_512: A7-s2 `1` vs A4-r3 `1`
    - scale_anchor_768: A7-s2 `1` vs A4-r3 `1`
- 结论: 采纳 A7-s2，固定基线从 A4-r3 切换为 A7-s2
