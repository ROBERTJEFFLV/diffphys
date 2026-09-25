"""Native CUDA tests are separate from CPU stand-ins; skips are not GPU evidence."""
import threading
import uuid
import warnings

import pytest
import torch

from response_groups import (_make_batch_rule, _stack_native_backward,
                               _sparse_native_backward, native_gru_vmap)
from response_groups import GroupBalanceConfig, _batched_group_vjp
from tools.train_response_control import parse_args


def cpu_standin(grad, workspace, has_bias):
    """Shape/dispatch stand-in using the published gate formula, NOT the CUDA kernel."""
    r, z, n, h, hn = workspace.chunk(5, -1)
    gz = grad * (h - n) * (1 - z) * z
    hx = grad * z
    gn = grad * (1 - z) * (1 - n * n)
    gr = gn * hn * (1 - r) * r
    gi = torch.cat((gr, gz, gn), -1)
    gh = torch.cat((gr, gz, gn * r), -1)
    return gi, gh, hx, gi.sum(0) if has_bias else None, gh.sum(0) if has_bias else None


@pytest.mark.parametrize('dtype', [torch.float32, torch.float64])
@pytest.mark.parametrize('groups,scenes,hidden', [(1, 1, 1), (3, 7, 5), (16, 32, 64)])
@pytest.mark.parametrize('has_bias', [True, False])
def test_flattening_matches_separate_calls_cpu_standin(dtype, groups, scenes, hidden, has_bias):
    gen = torch.Generator().manual_seed(7)
    grad = torch.randn(groups, hidden, scenes, generator=gen, dtype=dtype).transpose(1, 2)
    workspace = torch.randn(scenes, 5 * hidden, generator=gen, dtype=dtype)[None].expand(groups, -1, -1)
    expected = [cpu_standin(grad[g], workspace[g], has_bias) for g in range(groups)]
    calls = []
    def counted(*args):
        calls.append(args[0].shape)
        return cpu_standin(*args)
    result = _stack_native_backward(counted, grad, workspace, has_bias)
    assert calls == [torch.Size((groups * scenes, hidden))]
    for i, value in enumerate(result):
        if value is None:
            assert all(row[i] is None for row in expected)
        else:
            assert torch.equal(value, torch.stack([row[i] for row in expected]))


def test_bias_keeps_groups_separate_and_verify_detects_corruption():
    g = torch.ones(2, 3, 4)
    w = torch.rand(2, 3, 20)
    _stack_native_backward(cpu_standin, g, w, True, verify=True)
    def corrupt(grad, workspace, has_bias):
        values = list(cpu_standin(grad, workspace, has_bias))
        if grad.shape[0] == 6:
            values[0] = values[0] + 1
        return tuple(values)
    with pytest.raises(RuntimeError, match='verification failed'):
        _stack_native_backward(corrupt, g, w, True, verify=True)


def test_dispatch_rule_reduces_calls_and_preserves_none_cpu_standin():
    # Temporary private namespace, never a CPU implementation of an aten op.
    ns = 'diffphys_test_' + uuid.uuid4().hex
    definitions = torch.library.Library(ns, 'DEF')
    definitions.define('gru(Tensor g, Tensor w, bool bias) -> (Tensor, Tensor, Tensor, Tensor, Tensor)')
    calls = []
    def counted(*args):
        calls.append(args[0].shape)
        return cpu_standin(*args)
    definitions.impl('gru', counted, 'CPU')
    op = getattr(torch.ops, ns).gru.default
    grad = torch.randn(4, 7, 5)
    workspace = torch.randn(7, 25)
    expected = tuple(torch.stack([op(grad[g], workspace, True)[i] for g in range(4)])
                     for i in range(5))
    implementation = torch.library.Library(ns, 'IMPL', 'FuncTorchBatched')
    try:
        implementation.impl('gru', _make_batch_rule(op, verify=False, owner_thread=threading.get_ident()))
        calls.clear()
        actual = torch.vmap(lambda g: op(g, workspace, True))(grad)
        assert calls == [torch.Size((28, 5))]
        assert all(torch.equal(a, b) for a, b in zip(actual, expected))
        # The wrapper discards undefined bias outputs just as native autograd does.
        actual = torch.vmap(lambda g: op(g, workspace, False)[:3])(grad)
        assert all(torch.equal(a, b) for a, b in zip(actual, expected[:3]))
        # Batched saved workspace and unbatched grad are also handled.
        ws = workspace[None].expand(4, -1, -1).clone()
        actual = torch.vmap(lambda w: op(grad[0], w, True))(ws)
        expected_ws = op(grad[0], workspace, True)
        assert all(torch.equal(a, b[None].expand_as(a)) for a, b in zip(actual, expected_ws))
    finally:
        implementation._destroy()
        definitions._destroy()


def test_registration_restored_on_exception_and_default_is_untouched():
    opname = 'aten::_thnn_fused_gru_cell_backward'
    has_kernel = torch._C._dispatch_has_kernel_for_dispatch_key(opname, 'FuncTorchBatched')
    with native_gru_vmap('fallback'):
        assert torch._C._dispatch_has_kernel_for_dispatch_key(opname, 'FuncTorchBatched') == has_kernel
    if has_kernel:
        with pytest.raises(RuntimeError, match='already has'):
            with native_gru_vmap('native'):
                pass
    else:
        with pytest.raises(ValueError, match='injected'):
            with native_gru_vmap('native'):
                assert torch._C._dispatch_has_kernel_for_dispatch_key(opname, 'FuncTorchBatched')
                raise ValueError('injected failure')
        assert not torch._C._dispatch_has_kernel_for_dispatch_key(opname, 'FuncTorchBatched')


def test_flags_are_bound_and_old_backend_is_default():
    assert GroupBalanceConfig.from_args(parse_args([])).gru_vmap_mode == 'fallback'
    for mode in ('native', 'verify'):
        args = parse_args(['--group-gru-vmap-mode', mode])
        assert GroupBalanceConfig.from_args(args).gru_vmap_mode == mode
    with pytest.raises(ValueError):
        GroupBalanceConfig(gru_vmap_mode='silent-auto')


@pytest.mark.skipif(not torch.cuda.is_available(), reason='needs actual CUDA fused GRU backward')
@pytest.mark.parametrize('dtype', [torch.float32, torch.float64])
@pytest.mark.parametrize('mode', ['verify', 'sparse-verify'])
def test_actual_cuda_gru_forward_boundary_raw_gradients_and_adam_equal(dtype, mode):
    # A recurrent test of the real CUDA workspace/kernel, not the stand-in.
    import copy
    from response_groups import normalize_group_rows
    torch.manual_seed(7)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    cell = torch.nn.GRUCell(7, 13).to(device='cuda', dtype=dtype)
    optimizer = torch.optim.Adam(cell.parameters(), lr=3e-4)
    sum(p.square().sum() for p in cell.parameters()).backward()
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)
    params = list(cell.parameters())
    x = torch.randn(9, 32, 7, device='cuda', dtype=dtype)
    h = torch.randn(32, 13, device='cuda', dtype=dtype)
    costs = torch.zeros(32, device='cuda', dtype=dtype)
    outputs = []
    for t in range(9):
        h = cell(x[t], h)
        outputs.append(h.detach().clone())
        costs = costs + h.square().sum(-1)
    seeds = torch.eye(4, device='cuda', dtype=dtype).repeat_interleave(8, dim=1)
    rng = torch.cuda.get_rng_state().clone()
    expected = _batched_group_vjp(costs, params, seeds, retain_graph=True)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        actual = _batched_group_vjp(costs, params, seeds, retain_graph=False, gru_vmap_mode=mode)
    assert all(torch.equal(a, b) for a, b in zip(actual, expected))
    assert not any('_thnn_fused_gru_cell_backward' in str(w.message) and 'performance drop' in str(w.message)
                   for w in caught)
    assert torch.equal(h.detach(), outputs[-1])
    assert torch.equal(rng, torch.cuda.get_rng_state())
    before_model = copy.deepcopy(cell.state_dict())
    before_optim = copy.deepcopy(optimizer.state_dict())
    results = []
    for gradients in (expected, actual):
        cell.load_state_dict(before_model)
        optimizer.load_state_dict(copy.deepcopy(before_optim))
        rows = torch.cat([g.reshape(4, -1) for g in gradients], 1)
        combined, _ = normalize_group_rows(rows, 1e-12)
        offset = 0
        for p in params:
            p.grad = combined[offset:offset+p.numel()].reshape_as(p).clone()
            offset += p.numel()
        optimizer.step()
        results.append((copy.deepcopy(cell.state_dict()), copy.deepcopy(optimizer.state_dict())))
    assert all(torch.equal(results[0][0][k], results[1][0][k]) for k in before_model)
    for pid, state in results[0][1]['state'].items():
        for key, value in state.items():
            assert torch.equal(value, results[1][1]['state'][pid][key])


@pytest.mark.parametrize('dtype', [torch.float32, torch.float64])
@pytest.mark.parametrize('has_bias', [False, True])
def test_sparse_gru_preserves_native_reductions_cpu_standin(dtype, has_bias):
    groups, scenes, hidden = 16, 37, 11
    generator = torch.Generator().manual_seed(11)
    owner = torch.arange(scenes) % groups
    dense = torch.randn(scenes, hidden, generator=generator, dtype=dtype)
    grad = torch.where((torch.arange(groups)[:, None] == owner[None])[:, :, None], dense[None], 0.)
    grad[:, 3] = 0.  # Entirely zero scene still keeps a zero contribution.
    workspace = torch.randn(scenes, 5*hidden, generator=generator, dtype=dtype)
    calls, flags = [], []
    def counted(*args):
        calls.append(args[0].shape)
        return cpu_standin(*args)
    actual = _sparse_native_backward(counted, grad, workspace, has_bias, flags=flags)
    assert calls == [torch.Size((scenes, hidden))]  # B rows, not G*B rows.
    assert bool(torch.stack(flags).all())
    expected = _stack_native_backward(cpu_standin, grad, workspace[None].expand(groups,-1,-1), has_bias)
    for a, b in zip(actual, expected):
        assert (a is b) if a is None else torch.equal(a, b)
    _sparse_native_backward(cpu_standin, grad, workspace, has_bias, flags=[], verify=True)


def test_sparse_guard_detects_non_disjoint_signals():
    grad = torch.ones(2, 3, 4)
    workspace = torch.rand(3, 20)
    flags = []
    _sparse_native_backward(cpu_standin, grad, workspace, True, flags=flags)
    assert not bool(torch.stack(flags).all())
    with pytest.raises(RuntimeError, match='verification failed'):
        _sparse_native_backward(cpu_standin, grad, workspace, True, flags=[], verify=True)


def test_graph_task_tag_routes_only_requested_backward_cpu_standin():
    ns = 'diffphys_task_' + uuid.uuid4().hex
    definitions = torch.library.Library(ns, 'DEF')
    definitions.define('gru(Tensor g, Tensor w, bool bias) -> (Tensor, Tensor, Tensor, Tensor, Tensor)')
    calls = []
    def counted(*args):
        calls.append(args[0].shape)
        return cpu_standin(*args)
    definitions.impl('gru', counted, 'CPU')
    op = getattr(torch.ops, ns).gru.default
    class Gate(torch.autograd.Function):
        @staticmethod
        def forward(ctx, x, workspace):
            ctx.save_for_backward(workspace)
            return x * workspace.chunk(5, -1)[1]
        @staticmethod
        def backward(ctx, gradient):
            workspace, = ctx.saved_tensors
            return op(gradient, workspace, False)[2], None
    x = torch.nn.Parameter(torch.randn(8, 3))
    workspace = torch.randn(8, 15)
    costs = Gate.apply(x, workspace).sum(-1)
    allowed = set()
    handle = costs.register_hook(lambda g: allowed.add(torch._C._current_graph_task_id()))
    implementation = torch.library.Library(ns, 'IMPL', 'FuncTorchBatched')
    try:
        implementation.impl('gru', _make_batch_rule(op, verify=False, allowed_graphs=allowed))
        seeds = torch.eye(2).repeat_interleave(4, dim=1)
        actual = _batched_group_vjp(costs, [x], seeds, retain_graph=False)[0]
        expected = seeds[:, :, None] * workspace.chunk(5, -1)[1][None]
        assert torch.equal(actual, expected)
        assert calls == [torch.Size((16, 3))]
        calls.clear()
        # Outside the tagged graph task this uses the unoptimized native path.
        torch.vmap(lambda g: op(g, workspace, True))(torch.randn(2, 8, 3))
        assert calls == [torch.Size((8, 3)), torch.Size((8, 3))]
    finally:
        handle.remove()
        implementation._destroy()
        definitions._destroy()
