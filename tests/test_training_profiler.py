"""The reusable profiler is read-only and reports the real production backend."""
import pytest
import torch

from tools.profile_response_training import main
from response_policy import ARCHITECTURE, ResponseMotorPolicy
from response_training import model_hash


def test_cpu_profiler_never_steps_or_mutates_checkpoint(tmp_path, monkeypatch):
    policy = ResponseMotorPolicy()
    checkpoint = tmp_path/'actor.pt'
    torch.save({'architecture':ARCHITECTURE, 'policy_config':{'memory_dim':64, 'dt':.01},
                'model':policy.state_dict(), 'model_sha256':model_hash(policy)}, checkpoint)
    before = checkpoint.read_bytes()
    def forbidden(*args, **kwargs):
        raise AssertionError('profiler must not perform an Adam update')
    monkeypatch.setattr(torch.optim.Adam, 'step', forbidden)
    result = main(['--device','cpu','--scenes','128','--horizon','3','--warmup','0',
                   '--repeats','1','--epsilon-p','.01','--epsilon-a','.01','--lambda-R','.2',
                   '--checkpoint',str(checkpoint),'--report',str(tmp_path/'profile.json')])
    assert checkpoint.read_bytes() == before
    assert result['model_unchanged'] is True
    assert result['model_sha256'] == model_hash(policy)
    assert len(result['samples']) == 1
    assert result['scenes'] == 128
    assert result['samples'][0]['physical_transitions'] <= 384
    assert result['numerical_backend']['rotation'] == 'eager'
    assert result['median_seconds'] > 0
    assert result['samples'][0]['peak_bytes'] is None


def test_profiler_rejects_overwriting_input_checkpoint(tmp_path):
    checkpoint = tmp_path/'actor.pt'
    checkpoint.write_bytes(b'original')
    with pytest.raises(SystemExit):
        main(['--device','cpu','--checkpoint',str(checkpoint),'--report',str(checkpoint),
              '--epsilon-p','.01','--epsilon-a','.01','--lambda-R','.2'])
    assert checkpoint.read_bytes() == b'original'
