# 物理采样审计状态

当前 `physical-fit` 实现位于 `env_l2f.py`，其主惯量比范围为
`Jz/Jxy = 1.45–1.95`。基于当前源码、4 个固定种子、共 65,536 个
float32 样本的审计位于：

`reports/physical_fit_sampler_audit_20260806/`

该审计中非有限值、非正质量/惯量/时间常数、主惯量三角不等式违规均为 0。
这只是数学硬门，不代表 sampler 已充分覆盖真实无人机家族。

`archive_pre_fix_20260804/` 仅用于历史追溯，不可作为当前 sampler 的证据。
