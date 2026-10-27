# 数据集说明

## 输入文件

完整数据构建默认读取以下本地文件：

| 文件 | 用途 |
| --- | --- |
| `data/sbxx.csv` | 传感器时序读数 |
| `data/dyxx.csv` | 事件、工位和时间元数据 |
| `data/ms_cvtlimit.csv` | 故障阈值规则 |

原始文件不进入 Git。

## 特征与标签

使用 9 个特征：

1. `OilMistTemperature`
2. `Motor1_Current`
3. `Motor1_Temperature`
4. `Motor1_Torque`
5. `Motor1_Speed`
6. `Motor2_Current`
7. `Motor2_Temperature`
8. `Motor2_Torque`
9. `Motor2_Speed`

分类目标为 10 类：正常状态，以及两台电机的电流、温度、扭矩、转速异常和
油雾温度异常。

## 处理流程

1. 按事件 ID 关联传感器与事件元数据。
2. 从长表过滤目标特征并透视为宽表。
3. 按工位和事件时间排序。
4. 将无效低温值设为缺失，执行插值并处理剩余缺失。
5. 按工位时间顺序划分 train/val/test。
6. 使用窗口长度 1000 构建滑窗样本索引。

具体参数以各脚本的 `--help` 和输出元数据为准。默认输出位于
`dataset_final/`，包括数据表、滑窗索引、清洗统计和预览图。
