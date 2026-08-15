import sys
import types
from argparse import Namespace
from contextlib import ExitStack
from unittest.mock import MagicMock, patch

import pytest
import torch


def _stub_module(name: str, attrs: dict[str, object] | None = None, is_package: bool = False) -> types.ModuleType:
    module = types.ModuleType(name)
    if is_package:
        module.__path__ = []
    if attrs is not None:
        for attr_name, value in attrs.items():
            setattr(module, attr_name, value)
    sys.modules[name] = module
    return module


class _DummyDDP:
    pass


class _DummyModel:
    pass


class _DummyOptimizer:
    pass


class _DummyChainedOptimizer:
    pass


class _DummyDistributedOptimizer:
    pass


class _DummyScheduler:
    pass


class _DummyOptimizerConfig:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)


class _FakeModelChunk:
    role: str | None = None


@pytest.fixture(scope="module", autouse=True)
def _mock_megatron_environment():
    original_modules = dict(sys.modules)
    try:
        _stub_module("megatron", is_package=True)
        core_module = _stub_module("megatron.core", is_package=True)
        core_module.mpu = types.SimpleNamespace()
        core_module.tensor_parallel = types.SimpleNamespace(model_parallel_cuda_manual_seed=MagicMock())
        _stub_module(
            "megatron.core.distributed",
            {
                "DistributedDataParallel": _DummyDDP,
                "finalize_model_grads": MagicMock(),
            },
        )
        _stub_module(
            "megatron.core.enums",
            {"ModelType": types.SimpleNamespace(encoder_or_decoder="encoder_or_decoder")},
        )
        _stub_module("megatron.core.models", is_package=True)
        _stub_module("megatron.core.models.gpt", {"GPTModel": _DummyModel})
        _stub_module(
            "megatron.core.optimizer",
            {
                "OptimizerConfig": _DummyOptimizerConfig,
                "get_megatron_optimizer": MagicMock(),
            },
            is_package=True,
        )
        _stub_module("megatron.core.optimizer.muon", {"get_megatron_muon_optimizer": MagicMock()})
        _stub_module("megatron.core.optimizer.distrib_optimizer", {"DistributedOptimizer": _DummyDistributedOptimizer})
        _stub_module(
            "megatron.core.optimizer.optimizer",
            {
                "ChainedOptimizer": _DummyChainedOptimizer,
                "MegatronOptimizer": _DummyOptimizer,
            },
        )
        _stub_module("megatron.core.optimizer_param_scheduler", {"OptimizerParamScheduler": _DummyScheduler})
        _stub_module("megatron.core.packed_seq_params", {"PackedSeqParams": MagicMock()})
        _stub_module("megatron.core.pipeline_parallel", {"get_forward_backward_func": MagicMock()})
        _stub_module("megatron.core.transformer", is_package=True)
        _stub_module("megatron.core.transformer.utils", {"sharded_state_dict_default": MagicMock()})
        _stub_module("megatron.core.utils", {"get_model_config": MagicMock()})
        _stub_module("megatron.core.config", {"set_experimental_flag": MagicMock()})
        _stub_module("megatron.core.num_microbatches_calculator", {"init_num_microbatches_calculator": MagicMock()})
        _stub_module("megatron.training", is_package=True)
        _stub_module(
            "megatron.training.global_vars",
            {
                "get_args": MagicMock(),
                "_build_tokenizer": MagicMock(),
                "set_args": MagicMock(),
            },
        )
        _stub_module("megatron.training.training", {"get_model": MagicMock()})
        _stub_module(
            "megatron.training.checkpointing",
            {
                "load_checkpoint": MagicMock(),
                "save_checkpoint": MagicMock(),
            },
        )
        _stub_module("sglang.srt.debug_utils", is_package=True)
        _stub_module(
            "sglang.srt.debug_utils.dumper",
            {
                "DumperConfig": MagicMock(),
                "_get_rank": MagicMock(return_value=0),
                "dumper": MagicMock(),
            },
        )
        _stub_module(
            "miles.backends.megatron_utils.bridge_lora_helpers",
            {
                "_ensure_model_list": MagicMock(),
                "_setup_lora_model_via_bridge": MagicMock(),
            },
        )
        _stub_module("miles.backends.megatron_utils.model_provider", {"get_model_provider_func": MagicMock()})
        yield
    finally:
        sys.modules.clear()
        sys.modules.update(original_modules)


def _patch_initialize_side_effects(stack: ExitStack) -> None:
    stack.enter_context(patch("miles.backends.megatron_utils.model.clear_memory"))
    stack.enter_context(patch("miles.backends.megatron_utils.model.check_peak_gpu_memory_after_load"))
    stack.enter_context(patch("miles.backends.megatron_utils.model.check_model_hashes"))


def test_initialize_does_not_step_scheduler_restored_from_checkpoint():
    from miles.backends.megatron_utils.model import initialize_model_and_optimizer

    args = Namespace(use_checkpoint_opt_param_scheduler=True, global_batch_size=8)
    model = [_FakeModelChunk()]
    optimizer = object()
    opt_param_scheduler = MagicMock()

    with ExitStack() as stack:
        stack.enter_context(
            patch(
                "miles.backends.megatron_utils.model.setup_model_and_optimizer",
                return_value=(model, optimizer, opt_param_scheduler),
            )
        )
        stack.enter_context(patch("miles.backends.megatron_utils.model.load_checkpoint", return_value=(100, 0)))
        _patch_initialize_side_effects(stack)
        result = initialize_model_and_optimizer(args)

    assert result == (model, optimizer, opt_param_scheduler, 100)
    opt_param_scheduler.step.assert_not_called()


def test_initialize_steps_scheduler_when_checkpoint_did_not_restore_it():
    from miles.backends.megatron_utils.model import initialize_model_and_optimizer

    args = Namespace(use_checkpoint_opt_param_scheduler=False, global_batch_size=8)
    model = [_FakeModelChunk()]
    optimizer = object()
    opt_param_scheduler = MagicMock()

    with ExitStack() as stack:
        stack.enter_context(
            patch(
                "miles.backends.megatron_utils.model.setup_model_and_optimizer",
                return_value=(model, optimizer, opt_param_scheduler),
            )
        )
        stack.enter_context(patch("miles.backends.megatron_utils.model.load_checkpoint", return_value=(100, 0)))
        _patch_initialize_side_effects(stack)
        result = initialize_model_and_optimizer(args)

    assert result == (model, optimizer, opt_param_scheduler, 100)
    opt_param_scheduler.step.assert_called_once_with(increment=800)


def test_debug_module_backward_hook_matches_live_module_and_observes_gradient(monkeypatch):
    from miles.backends.megatron_utils import model as model_module

    model = torch.nn.Module()
    model.target = torch.nn.Linear(3, 2, bias=False)
    monkeypatch.setenv("MILES_DEBUG_MODULE_BACKWARD_SUFFIXES", "target")

    with patch.object(model_module, "_log_debug_module_tensor") as log_tensor:
        model_module._install_debug_module_backward_hooks([model])
        model.target(torch.ones(1, 3, requires_grad=True)).sum().backward()

    log_tensor.assert_called_once()
    assert log_tensor.call_args.kwargs["module_name"] == "chunk=0 target"
    assert log_tensor.call_args.kwargs["tensor_path"] == "output"
    assert log_tensor.call_args.kwargs["kind"] == "output_grad"


def test_debug_module_backward_hook_rejects_inert_suffix(monkeypatch):
    from miles.backends.megatron_utils import model as model_module

    monkeypatch.setenv("MILES_DEBUG_MODULE_BACKWARD_SUFFIXES", "missing")
    with pytest.raises(RuntimeError, match="matched no live modules"):
        model_module._install_debug_module_backward_hooks([torch.nn.Linear(2, 2)])


def test_debug_module_tensor_hook_observes_forward_and_parameter_grad(monkeypatch):
    from miles.backends.megatron_utils import model as model_module

    model = torch.nn.Module()
    model.adapter = torch.nn.Linear(3, 2, bias=False)
    monkeypatch.setenv("MILES_DEBUG_MODULE_TENSOR_SUFFIXES", "adapter")

    with patch.object(model_module, "_log_debug_module_tensor") as log_tensor:
        model_module._install_debug_module_tensor_hooks([model])
        model.adapter(torch.ones(1, 3, requires_grad=True)).sum().backward()

    kinds = [call.kwargs["kind"] for call in log_tensor.call_args_list]
    assert kinds == ["forward_input", "forward_output", "parameter_grad"]
    assert {call.kwargs["module_name"] for call in log_tensor.call_args_list} == {"chunk=0 adapter"}


def test_debug_module_tensor_hook_rejects_inert_suffix(monkeypatch):
    from miles.backends.megatron_utils import model as model_module

    monkeypatch.setenv("MILES_DEBUG_MODULE_TENSOR_SUFFIXES", "missing")
    with pytest.raises(RuntimeError, match="matched no live modules"):
        model_module._install_debug_module_tensor_hooks([torch.nn.Linear(2, 2)])
