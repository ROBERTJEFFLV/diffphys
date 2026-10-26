"""CPU storage must preserve CUDA values, layouts and full backward graphs."""
from __future__ import annotations

import pytest
import torch


def test_cpu_storage_is_a_noop():
    from tensor_storage import saved_tensor_storage
    x = torch.randn(8, requires_grad=True)
    with saved_tensor_storage('cpu') as stats:
        y = x.square().sum()
    y.backward()
    assert stats == {'offloaded_tensors': 0, 'offloaded_bytes': 0}
    assert torch.equal(x.grad, 2*x)


@pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA unavailable')
@pytest.mark.parametrize('compiled', [False, True])
def test_large_saved_values_and_strided_views_have_identical_gradients(compiled):
    from tensor_storage import saved_tensor_storage
    from compiled_execution import compiled_kernel
    def objective(x):
        return x.square().sum() + x.T.sin().square().sum()
    calculate = compiled_kernel(objective) if compiled else objective
    source = torch.randn(256, 256, device='cuda', dtype=torch.float64)
    gradients = []
    for offload in (False, True):
        x = source.clone().requires_grad_()
        if offload:
            with saved_tensor_storage('cuda') as stats:
                cost = calculate(x)
        else:
            cost = calculate(x)
        gradients.append(torch.autograd.grad(cost, x)[0])
    assert stats['offloaded_bytes'] >= source.numel()*source.element_size()
    assert stats['offloaded_tensors'] > 0
    assert torch.equal(*gradients)
