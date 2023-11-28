### 运行参数选项

```
-M           模型 espcn_ex 或 espcn_um，默认 espcn_ex
-D           数据集 test 或 val，默认 val
--device     TPU编号，默认0
--n_workers  线程数，默认4
```

### 运行命令行示例

我们提交了两个模型 **espcn_ex** 和 **espcn_um**，取决于评分计算公式中的 **i_time** 到底应该包含哪些时间统计，您应该运行不同的模型

⚪ i_time = TPU 推理时间 (默认，也是提交的最高分配置)

```shell
# A榜数据集
python run.py -M espcn_ex -D test
# B榜数据集
python run.py -M espcn_ex -D val
```

⚪ i_time = TPU 推理时间 + CPU 前/后处理时间

```shell
# A榜数据集
python run.py -M espcn_um -D test
# B榜数据集
python run.py -M espcn_um -D val
```

在精简并打包代码之后我们跑了前 **100** 张图，以下为参考数据：

| niqe/time/score | test (A榜) | val (B榜) |
| :-: | :-: | :-: |
| espcn_ex | 4.0762/0.1323/2583.9500 | 4.2076/0.1013/3297.7700 |
| espcn_um | 4.0129/0.1556/2220.8975 | 4.1256/0.1181/2870.7939 |
