"""Device correctness with real updates, masking, history and checkpoint reload."""
import numpy as np
import pytest
import torch
from arcengine import GameAction
from DQN.dqn_agent import DQNAgent
from DQN.q_network import encode_frames, QNetwork


@pytest.mark.parametrize("device", ["cpu", pytest.param("cuda", marks=pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable"))])
def test_fixed_pool_matches_adaptive_forward_and_gradient(device):
    previous = torch.are_deterministic_algorithms_enabled()
    try:
        torch.use_deterministic_algorithms(True)
        torch.manual_seed(42)
        x = torch.randn(2, 64, 64, 64, device=device, requires_grad=True)
        fixed = torch.nn.AvgPool2d(16,16)(x)
        # Reference computed on CPU: CUDA adaptive backward is the failing op.
        reference_input = x.detach().cpu().clone().requires_grad_()
        reference = torch.nn.AdaptiveAvgPool2d((4,4))(reference_input)
        torch.testing.assert_close(fixed.cpu(), reference, rtol=1e-5, atol=1e-6)
        weights = torch.linspace(-1,1,fixed.numel()).reshape(fixed.shape)
        (fixed * weights.to(device)).sum().backward()
        (reference * weights).sum().backward()
        torch.testing.assert_close(x.grad.cpu(), reference_input.grad, rtol=0, atol=0)
        for frames in (1,4):
            current = QNetwork(frames)
            legacy = QNetwork(frames)
            legacy.discrete_head[0] = torch.nn.AdaptiveAvgPool2d((4,4))
            assert current.state_dict().keys() == legacy.state_dict().keys()
            legacy.load_state_dict(current.state_dict(),strict=True)
    finally:
        torch.use_deterministic_algorithms(previous)

@pytest.mark.parametrize("device", ["cpu", pytest.param("cuda", marks=pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable"))])
@pytest.mark.parametrize("frames", [1, 4])
def test_learning_device_padding_terminal_target_checkpoint(device, frames, tmp_path):
    torch.set_num_threads(1)
    a = DQNAgent(device=device, num_frames=frames, batch_size=2, target_update_interval=1)
    grid = np.zeros((frames, 64, 64), dtype=np.uint8)
    grid[:, 0, 0] = 16
    encoded = a.encode_batch([grid, grid])
    assert encoded.device.type == device and encoded.shape == (2, frames*16, 64, 64)
    assert encoded[:, :, 0, 0].sum().item() == 0
    torch.testing.assert_close(encoded.cpu(), encode_frames(np.stack([grid, grid])))
    assert a.select_action(grid, [GameAction.ACTION1], 0) == 0
    a.remember(grid, 0, 1, grid, False, [GameAction.ACTION1])
    a.remember(grid, 0, 1, grid, True, [])
    before = {k: v.detach().clone() for k, v in a.online.state_dict().items()}
    assert np.isfinite(a.learn())
    assert any(not torch.equal(before[k], v) for k, v in a.online.state_dict().items())
    for k, v in a.online.state_dict().items():
        assert v.device.type == device
        torch.testing.assert_close(v, a.target.state_dict()[k])
    for state in a.optimizer.state.values():
        assert state["exp_avg"].device.type == device
    path = tmp_path / "device.pt"
    torch.save({"online_state_dict": a.online.state_dict(), "target_state_dict": a.target.state_dict(),
                "optimizer_state_dict": a.optimizer.state_dict()}, path)
    saved = torch.load(path, map_location="cpu", weights_only=True)
    cpu = DQNAgent(num_frames=frames)
    cpu.online.load_state_dict(saved["online_state_dict"])
    cpu.target.load_state_dict(saved["target_state_dict"])
    cpu.optimizer.load_state_dict(saved["optimizer_state_dict"])
    assert cpu.select_action(grid, [GameAction.ACTION1], 0) == 0


def test_profiled_shared_trainer_outputs(monkeypatch, tmp_path):
    import json
    import sys
    import DQN.train as train
    from tests.fake_arc import FakeArcade
    monkeypatch.setattr(train.arc_agi, "Arcade", lambda **kw: FakeArcade())
    monkeypatch.setattr(sys, "argv", ["train", "--backend", "shared", "--steps", "12",
        "--warmup", "2", "--train-every", "2", "--batch-size", "2", "--episode-limit", "0",
        "--device", "cpu", "--profile", "--torch-threads", "1", "--epsilon-start", "0",
        "--epsilon-end", "0", "--run-suffix", "profiletest", "--output-root", str(tmp_path)])
    train.main()
    report = json.loads(next((tmp_path / "Logs/DQN").glob("*_timing.json")).read_text())
    assert report["gradient_updates"] == 6 and report["selected_actions"] == 12
    assert report["stages_calls"]["environment_step"] == 12
    assert report["stages_calls"]["policy_forward"] == 12
    assert report["stages_calls"]["gradient_batch_encode_transfer"] == 6
    assert report["stages_seconds"]["gradient_update"] >= report["stages_seconds"]["gradient_batch_encode_transfer"]
    saved = torch.load(next((tmp_path / "DQN/checkpoints").glob("*.pt")), weights_only=True)
    assert saved["training_backend"] == "shared" and saved["config"]["device"] == "cpu"
    with pytest.raises(FileExistsError):
        train.main()
