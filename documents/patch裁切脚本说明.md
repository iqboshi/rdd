# patch 裁切脚本说明

更新时间：2026-03-06

脚本路径：`scripts/crop_processed_to_patches.py`

## 1. 作用

将 `data/processed/` 下已经对齐好的整图数据，按统一滑窗坐标裁成 patch，用于后续训练。  
支持同步裁切以下目标：

- `image`
- `semantic`
- `instance`
- `center`
- `offset`
- `density`

所有 target 与 image 使用同一组裁切坐标，保证空间严格对齐。

## 2. 输入目录约定

默认输入根目录：`data/processed`

- `images/`（必须）
- `semantic/`
- `instance/`
- `center/`
- `offset/` 或 `offset_x/ + offset_y/`
- `density/`

说明：

- 脚本遍历 `images/`，按同名文件在其它 target 目录中查找对应样本。
- 若选择了某个 target，但该 target 对应文件缺失，则该样本跳过并打印 warning。

## 3. 输出目录约定

默认输出根目录：`data/patches`

- `images/`
- `semantic/`
- `instance/`
- `center/`
- `offset/`
- `density/`

命名规则：

- patch 文件名：`原图名_x{left}_y{top}`
- 示例：`xxx_x512_y768.png`、`xxx_x512_y768.npy`

## 4. 滑窗与边界策略

参数：

- `patch_size`（默认 512）
- `stride`（默认 256）

坐标生成规则：

1. 先按 `range(0, L - patch_size + 1, stride)` 生成常规起点。
2. 若最后一个起点不等于 `L - patch_size`，额外补一个 `L - patch_size`。
3. 分别对宽度和高度执行该逻辑，组合得到 `(x, y)` 网格坐标。
4. 所有 patch 均完全落在图内，不做 padding。

小图策略：

- 当 `H < patch_size` 或 `W < patch_size`：
  - 默认 `--skip_small_images`：跳过并 warning；
  - 使用 `--no-skip_small_images`：抛出错误终止。

## 5. 各 target 处理规则

### 5.1 image / semantic

- 直接裁切并保持原图像格式保存。

### 5.2 instance

- 先裁切，再可选实例 ID 重映射（默认开启 `--remap_instance`）。
- 重映射规则：
  - 背景保持 `0`；
  - patch 内出现的非零实例 ID，按出现集合映射到 `1..N`。
- 输出固定保存为 `.npy`，避免实例 ID 精度丢失。

### 5.3 center

- 直接裁切整图 center。
- 不重新生成中心点。
- patch 内即使没有中心峰也允许。

### 5.4 offset

直接裁切，不重算 offset，保持整图定义。支持两种输入：

1. 合并格式：`offset/{stem}.npy`
   - 兼容 `H x W x 2` 或 `2 x H x W`。
2. 分离格式：
   - 同目录双文件（如 `{stem}_x` / `{stem}_y` 等命名模式）；
   - 或双目录：`offset_x/{stem}.*` + `offset_y/{stem}.*`。

输出统一为 `.npy`（`H x W x 2`）。

### 5.5 density

- 直接裁切整图 density。
- 不在 patch 内重新归一化。
- 输出固定保存为 `.npy`。

## 6. 命令行参数

- `--input_root`：输入根目录，默认 `data/processed`
- `--output_root`：输出根目录，默认 `data/patches`
- `--patch_size`：patch 尺寸，默认 `512`
- `--stride`：滑窗步长，默认 `256`
- `--targets`：要裁切的目标，默认 `all`
- `--remap_instance / --no-remap_instance`：是否进行实例重映射（默认开）
- `--skip_small_images / --no-skip_small_images`：是否跳过小图（默认开）

`--targets` 可选值：

- `all`
- `image semantic instance center offset density`（可组合）

## 7. 使用示例

全量裁切（默认参数）：

```bash
python scripts/crop_processed_to_patches.py
```

指定 patch 参数：

```bash
python scripts/crop_processed_to_patches.py --patch_size 640 --stride 320
```

只裁切 image+semantic+instance：

```bash
python scripts/crop_processed_to_patches.py --targets image semantic instance
```

关闭实例重映射：

```bash
python scripts/crop_processed_to_patches.py --no-remap_instance
```

## 8. 运行统计输出

脚本结束后会打印：

- 原图总数（`Input images total`）
- 成功处理原图数（`Input images processed`）
- 生成 patch 总数（`Total patches generated`）
- 各 target 保存数量（`Saved patch counts by target`）
- warning 数量（`Warnings`）

