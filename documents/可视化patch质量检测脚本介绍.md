- visualize_patch_quality_checks.py

  它会随机抽样 patch，并对每个样本输出 2x3 可视化面板，分别对应你要的 6 件事：

  1. image 与 semantic 对齐检查
  2. instance 合理性（背景是否含 0、ID 连续性、数量）
  3. center 峰值是否落在实例区域
  4. offset 方向是否大体指向对应实例中心（含余弦一致性指标）
  5. density 是否保留整图定义（按实例贡献统计，而非 patch 内重归一）
  6. patch 边界截断实例检查（边界触碰实例高亮）

  我已实测可运行，示例命令：

  python scripts/visualize_patch_quality_checks.py --patch_root data/patches --output_dir data/
  patches_vis_checks --num_samples 20

  输出：

  - 每个样本一张检查图：{stem}_check.jpg
  - 汇总指标：summary.csv