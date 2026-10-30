"""AgentKVConnector: a vLLM 0.27.1 KV connector for agent tool-call pauses.

When a request finishes (the agent issued a tool call), its full blocks are copied to pinned CPU
memory in bf16, plus a packed low-bit copy unless mode is "bf16". When the next turn arrives,
blocks that are no longer in vLLM's GPU prefix cache are reloaded asynchronously. Block 0 and the
recent window come back in bf16, the rest at the mode's precision, dequantized into ordinary
bf16 vLLM blocks. Nothing in vLLM is patched.

    vllm serve <model> --attention-backend FLASH_ATTN --kv-transfer-config '{
        "kv_connector": "AgentKVConnector", "kv_connector_module_path": "kvm.connector",
        "kv_role": "kv_both",
        "kv_connector_extra_config": {"mode": "uniform_int2", "cpu_bytes": 68719476736}}'

kv_connector_extra_config:
    mode               bf16 | uniform_int4 | uniform_int2              (default bf16)
    cpu_bytes          pinned host memory per worker for both copies    (default 32 GiB)
    recent_tokens      newest history tokens always reloaded in bf16   (default 64)
    save_prompt_only   only save prompt blocks, like vLLM's OffloadingConnector default
                       (default true)
    chunk_blocks       blocks per transfer chunk                       (default 64)
    stats_path         append one JSON line of load/save stats per finished request

Requests may pass kv_transfer_params={"tag": "..."}; the tag is copied into the stats line.
Per-request stats are also returned in the response's kv_transfer_params["kvm"].
"""

import json
import math
import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from vllm.distributed.kv_transfer.kv_connector.v1.base import (
    KVConnectorBase_V1,
    KVConnectorMetadata,
    KVConnectorRole,
)
from vllm.logger import init_logger

from kvm.quant import block_bytes
from kvm.sched import MODES, LoadJob, SaveJob, SchedulerState

if TYPE_CHECKING:
    import torch

    from vllm.config import VllmConfig
    from vllm.forward_context import ForwardContext
    from vllm.v1.attention.backend import AttentionMetadata
    from vllm.v1.core.kv_cache_manager import KVCacheBlocks
    from vllm.v1.core.sched.output import SchedulerOutput
    from vllm.v1.kv_cache_interface import KVCacheConfig
    from vllm.v1.outputs import KVConnectorOutput
    from vllm.v1.request import Request

logger = init_logger(__name__)


@dataclass
class AgentKVMetadata(KVConnectorMetadata):
    loads: list[LoadJob] = field(default_factory=list)
    saves: list[SaveJob] = field(default_factory=list)


class AgentKVConnector(KVConnectorBase_V1):
    def __init__(self, vllm_config: "VllmConfig", role: KVConnectorRole,
                 kv_cache_config: "KVCacheConfig | None" = None):
        super().__init__(vllm_config, role, kv_cache_config)
        extra = self._kv_transfer_config.kv_connector_extra_config or {}
        self.mode = extra.get("mode", "bf16")
        if self.mode not in MODES:
            raise ValueError(f"AgentKVConnector mode must be one of {sorted(MODES)}, got {self.mode!r}")
        self.lo_bits = MODES[self.mode]
        m, p = vllm_config.model_config, vllm_config.parallel_config
        self.n_layers, self.heads = m.get_num_layers(p), m.get_num_kv_heads(p)
        self.block_size, self.head_dim = vllm_config.cache_config.block_size, m.get_head_size()
        self.bytes_per_block = {16: self.n_layers * block_bytes(16, self.block_size, self.heads, self.head_dim)}
        if self.lo_bits:
            self.bytes_per_block[self.lo_bits] = self.n_layers * block_bytes(
                self.lo_bits, self.block_size, self.heads, self.head_dim)
        self.n_slots = int(extra.get("cpu_bytes", 32 * 2**30)) // sum(self.bytes_per_block.values())
        self.chunk_blocks = int(extra.get("chunk_blocks", 64))
        self.save_prompt_only = bool(extra.get("save_prompt_only", True))
        self.stats_path = extra.get("stats_path")
        self.state: SchedulerState | None = None
        self.transfer = None
        if role == KVConnectorRole.SCHEDULER:
            recent = math.ceil(int(extra.get("recent_tokens", 64)) / self.block_size)
            self.state = SchedulerState(self.block_size, self.n_slots, self.mode, recent, self.bytes_per_block)
        logger.info("AgentKVConnector(%s): mode=%s, %d CPU slots of %d tokens", role.name, self.mode,
                    self.n_slots, self.block_size)

    @property
    def requires_kv_delivery(self) -> bool:
        # Best-effort cache: a dropped save is just a later cache miss.
        return False

    # ---------------------------------------------------------------- scheduler side

    def get_num_new_matched_tokens(self, request: "Request", num_computed_tokens: int) -> tuple[int, bool]:
        if request.skip_reading_prefix_cache:  # e.g. prompt_logprobs: every prompt token is recomputed
            return 0, False
        n = self.state.num_matched_tokens(request.request_id, request.block_hashes, num_computed_tokens)
        return n, n > 0

    def update_state_after_alloc(self, request: "Request", blocks: "KVCacheBlocks",
                                 num_external_tokens: int) -> None:
        self.state.on_alloc(request.request_id, request.block_hashes, blocks.get_block_ids()[0],
                            num_external_tokens)

    def build_connector_meta(self, scheduler_output: "SchedulerOutput") -> KVConnectorMetadata:
        loads, saves = self.state.take_jobs()
        return AgentKVMetadata(loads=loads, saves=saves)

    def update_connector_output(self, connector_output: "KVConnectorOutput") -> None:
        self.state.on_worker_output(connector_output.finished_sending, connector_output.finished_recving)

    def request_finished(self, request: "Request", block_ids: list[int]) -> tuple[bool, dict[str, Any] | None]:
        limit = request.num_prompt_tokens if self.save_prompt_only else request.num_computed_tokens
        n = min(len(request.block_hashes), request.num_computed_tokens // self.block_size,
                limit // self.block_size)
        delay, stats = self.state.on_finished(request.request_id, request.block_hashes[:n], block_ids[:n])
        params = request.kv_transfer_params or {}
        stats.update(mode=self.mode, tag=params.get("tag"), prompt_tokens=request.num_prompt_tokens,
                     time=time.time())
        if self.stats_path:
            with open(self.stats_path, "a") as f:
                f.write(json.dumps(stats) + "\n")
        return delay, {"kvm": stats}

    # ---------------------------------------------------------------- worker side

    def register_kv_caches(self, kv_caches: dict[str, "torch.Tensor"]) -> None:
        from kvm.transfer import KVTransfer

        layers = list(kv_caches.values())
        want = (self.heads, self.block_size, 2 * self.head_dim)
        bad = [name for name, t in kv_caches.items() if t.dim() != 4 or tuple(t.shape[1:]) != want]
        if bad or len(layers) != self.n_layers:
            raise ValueError(
                f"AgentKVConnector expects {self.n_layers} FlashAttention KV tensors shaped "
                f"(num_blocks, {want[0]}, {want[1]}, {want[2]}); got {len(layers)} tensors, "
                f"mismatched: {bad[:3]}. Run vLLM with --attention-backend FLASH_ATTN.")
        self.transfer = KVTransfer(layers, self.n_slots, self.lo_bits, self.chunk_blocks)

    def start_load_kv(self, forward_context: "ForwardContext", **kwargs: Any) -> None:
        meta = self._get_connector_metadata()
        assert isinstance(meta, AgentKVMetadata)
        for job in meta.saves:
            self.transfer.save(job)
        for job in meta.loads:
            self.transfer.load(job)

    def wait_for_layer_load(self, layer_name: str) -> None:
        return

    def save_kv_layer(self, layer_name: str, kv_layer: "torch.Tensor",
                      attn_metadata: "AttentionMetadata", **kwargs: Any) -> None:
        return

    def wait_for_save(self) -> None:
        return

    def get_finished(self, finished_req_ids: set[str]) -> tuple[set[str] | None, set[str] | None]:
        sent, recved = self.transfer.poll()
        return sent or None, recved or None
