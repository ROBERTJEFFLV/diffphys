"""Scoped native arithmetic boundaries for the verified PyTorch 2.10 compiler.

AOTAutograd differentiates the original ATen graph. The post-autograd pass
selects native kernels for numerically sensitive operations in both generated
graphs; it never replaces a derivative or mutates global lowering tables.
"""
from __future__ import annotations

import torch
from torch._inductor.compile_fx import compile_fx
from torch._inductor.custom_graph_pass import CustomGraphPass, get_hash_for_files
from torch._inductor.decomposition import decompositions
from torch._inductor.lowering import fallback_handler


def _native(op, value, *args, **kwargs):
    if isinstance(value, torch.Tensor):  # Fake-tensor propagation during compilation.
        return op(value, *args, **kwargs)
    return fallback_handler(op, add_to_fallback_set=False)(value, *args, **kwargs)


def _native_sum(value, *args, **kwargs):
    return _native(torch.ops.aten.sum.dim_IntList, value, *args, **kwargs)


def _native_cross(value, *args, **kwargs):
    return _native(torch.ops.aten.linalg_cross.default, value, *args, **kwargs)


def _native_norm(value, *args, **kwargs):
    return _native(torch.ops.aten.linalg_vector_norm.default, value, *args, **kwargs)


def _native_bmm(value, *args, **kwargs):
    return _native(torch.ops.aten.bmm.default, value, *args, **kwargs)


def _native_cumsum(value, *args, **kwargs):
    return _native(torch.ops.aten.cumsum.default, value, *args, **kwargs)


NATIVE_BOUNDARIES = {
    torch.ops.aten.sum.dim_IntList: _native_sum,
    torch.ops.aten.linalg_cross.default: _native_cross,
    torch.ops.aten.linalg_vector_norm.default: _native_norm,
    torch.ops.aten.bmm.default: _native_bmm,
    # PyTorch 2.10 Triton SplitScan fails for the H500 no-grad EVAL layout.
    torch.ops.aten.cumsum.default: _native_cumsum,
}
for _lower in NATIVE_BOUNDARIES.values():
    _lower._inductor_lowering_function = True


def compiler_contract() -> dict:
    return {"version": "native-arithmetic-v2", "torch": "2.10",
            "dynamo": {"recompile_limit": 64},
            "native_boundaries": [str(op) for op in NATIVE_BOUNDARIES],
            "options": {"triton.cudagraphs": False, "emulate_precision_casts": True,
                        "emulate_divison_rounding": True}}


def configure_compiler() -> None:
    # Singleton compaction, layouts, grad modes and two supported dtypes need
    # more than eight guarded variants. Keep a finite budget and fullgraph errors.
    torch._dynamo.config.recompile_limit = compiler_contract()['dynamo']['recompile_limit']


class NativeArithmeticPass(CustomGraphPass):
    def __call__(self, graph: torch.fx.Graph) -> None:
        for node in graph.nodes:
            if node.op == "call_function" and node.target in NATIVE_BOUNDARIES:
                node.target = NATIVE_BOUNDARIES[node.target]

    def uuid(self) -> bytes:
        return get_hash_for_files((__file__,))


def strict_inductor(graph: torch.fx.GraphModule, inputs):
    if not str(torch.__version__).split("+")[0].startswith("2.10."):
        raise RuntimeError("compiled physics requires the verified PyTorch 2.10 compiler")
    table = {op: fn for op, fn in decompositions.items() if op not in NATIVE_BOUNDARIES}
    return compile_fx(graph, inputs, decompositions=table, config_patches={
        **compiler_contract()["options"],
        "post_grad_custom_pre_pass": NativeArithmeticPass(),
    })
