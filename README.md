# DGDETS：图扩散进化与禁忌局部搜索

## 四个基准入口

在工程根目录运行：

```powershell
python NAS-Bench-101\main.py --profile balanced
python NAS-Bench-201\main.py --dataset cifar10 --proxy-backend auto --profile balanced
python NAS-Bench-301\main.py --profile balanced
python TransNASBench-101\main.py --task class_object --search-space micro --profile balanced
```

四个入口默认读取各自 `config/config.py` 中固定记录的随机种子列表。`--exp-type fixed` 使用原固定种子；`--runs 2` 只运行列表前两个种子；`--exp-type single --seed 0` 用于单次诊断。

## NAS-Bench-201 的正确后端路由

`--proxy-backend auto` 是推荐设置，并严格复现 CDENAG 的数据集分工：

| 数据集 | 搜索评价后端 | 搜索输出含义 | 最终输出含义 |
|---|---|---|---|
| CIFAR-10、CIFAR-100、ImageNet16-120 | NAS-Bench-201 benchmark | 基准 Test Accuracy | 基准 Test Accuracy |
| Aircraft、Pets | MetaD2A | 校准代理排序分数，不是 Top-1 | 默认训练 top-k 候选后得到的真实任务 Test Top-1 |

标准任务可直接使用随包提供的 15,625 个架构紧凑测试标签；如提供完整 API，则用 `--api-path` 指向官方 PTH：

```powershell
python NAS-Bench-201\main.py --dataset cifar10 --proxy-backend auto
python NAS-Bench-201\main.py --dataset cifar100 --proxy-backend auto
python NAS-Bench-201\main.py --dataset ImageNet16-120 --proxy-backend auto
```

迁移任务默认采用 CDENAG 的“代理搜索 + top-k 完整训练”协议：

```powershell
python NAS-Bench-201\main.py --dataset aircraft --proxy-backend auto --meta-final-eval train
python NAS-Bench-201\main.py --dataset pets --proxy-backend auto --meta-final-eval train
```

## TransNASBench-101 全任务协议

下面命令遍历 macro/micro 两个空间和 API 中全部 7 个任务：

```powershell
python TransNASBench-101\all.py --profile balanced
```

可用 `--spaces micro`、`--tasks class_object jigsaw` 或 `--runs 1` 缩小范围。


