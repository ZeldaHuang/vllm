# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Unit tests for the scheduler-driven heartbeat / lease-renewal system."""

import logging
import time
from collections import defaultdict
from unittest.mock import MagicMock

import pytest

from vllm.v1.outputs import KVConnectorOutput

from .utils import create_request, make_nixl_scheduler

_ENGINE_A = "my-engine-id"


def _sched(kv_lease_duration: int = 30):
    return make_nixl_scheduler(heartbeat=True, kv_lease_duration=kv_lease_duration)


def _req(request_id: int = 1):
    return create_request(request_id=request_id, do_remote_prefill=True)


def _worker_stub():
    from vllm.distributed.kv_transfer.kv_connector.v1.nixl.worker import (
        NixlConnectorWorker,
    )

    w = object.__new__(NixlConnectorWorker)
    w._reqs_to_send = {}
    w._lease_extension = 20
    return w


# ===================================================================
# Scheduler: on_new_request
# ===================================================================


def test_on_new_request_tracks_and_groups():
    """Add two reqs to same engine, one to another; verify grouping."""
    s = _sched()
    s.on_new_request(_req(1))
    s.on_new_request(_req(2))

    assert s._heartbeat_by_engine[_ENGINE_A].req_ids == {"prefill-1", "prefill-2"}
    info = s._heartbeat_by_engine[_ENGINE_A]
    assert (info.host, info.port, info.tp_size) == ("my-host", 1234, 1)
    assert s._heartbeat_req_engine["id-1"] == (_ENGINE_A, "prefill-1")

    # Different engine.
    r3 = _req(3)
    r3.kv_transfer_params["remote_engine_id"] = "engine-b"
    s.on_new_request(r3)
    assert len(s._heartbeat_by_engine) == 2


@pytest.mark.parametrize(
    "make_req",
    [
        lambda: create_request(request_id=2, do_remote_decode=True),
        lambda: create_request(request_id=3),  # no kv_transfer_params
    ],
    ids=["decode", "plain"],
)
def test_on_new_request_ignores_non_prefill(make_req):
    s = _sched()
    s.on_new_request(make_req())
    assert len(s._heartbeat_by_engine) == 0


# ===================================================================
# Scheduler: _stop_heartbeat
# ===================================================================


def test_stop_heartbeat_partial_and_full():
    """Stop one of two reqs on same engine, then stop the other."""
    s = _sched()
    s.on_new_request(_req(1))
    s.on_new_request(_req(2))

    s._stop_heartbeat("id-1")
    assert s._heartbeat_by_engine[_ENGINE_A].req_ids == {"prefill-2"}
    assert "id-1" not in s._heartbeat_req_engine

    s._stop_heartbeat("id-2")
    assert len(s._heartbeat_by_engine) == 0
    assert len(s._heartbeat_req_engine) == 0


# ===================================================================
# Scheduler: build_connector_meta throttling
# ===================================================================


def test_build_connector_meta_heartbeat_throttling():
    # kv_lease_duration=30 => _heartbeat_interval = 30 // 6 = 5
    s = _sched(kv_lease_duration=30)
    s.on_new_request(_req(1))

    # Ensure the first call triggers by placing last_heartbeat far in the past.
    s._last_heartbeat_time = time.perf_counter() - 10
    meta1 = s.build_connector_meta(MagicMock())
    assert _ENGINE_A in meta1.heartbeat_by_engine

    # Immediate second call is throttled (< 5s since last).
    meta2 = s.build_connector_meta(MagicMock())
    assert len(meta2.heartbeat_by_engine) == 0


# ===================================================================
# Scheduler: cleanup paths (update_connector_output / request_finished)
# ===================================================================


def test_update_connector_output_stops_heartbeat():
    s = _sched()
    s.on_new_request(_req(1))

    s.update_connector_output(
        KVConnectorOutput(
            finished_sending=None,
            finished_recving={"id-1"},
            invalid_block_ids=set(),
        )
    )

    assert len(s._heartbeat_by_engine) == 0
    assert len(s._heartbeat_req_engine) == 0


def test_request_finished_stops_heartbeat():
    s = _sched()
    r = _req(1)
    s.on_new_request(r)

    # Simulate update_state_after_alloc having consumed do_remote_prefill.
    r.kv_transfer_params["do_remote_prefill"] = False
    s.request_finished(r, block_ids=())

    assert len(s._heartbeat_by_engine) == 0
    assert len(s._heartbeat_req_engine) == 0


# ===================================================================
# Worker: _handle_heartbeat
# ===================================================================


def test_handle_heartbeat():
    w = _worker_stub()
    far_future = time.perf_counter() + 99999
    w._reqs_to_send = {"req-a": 100.0, "req-b": far_future}

    before = time.perf_counter()
    w._handle_heartbeat("req-a,req-b,req-unknown")

    # req-a: pushed forward to ~now+20.
    assert w._reqs_to_send["req-a"] >= before + 20
    # req-b: already far out, max() keeps it.
    assert w._reqs_to_send["req-b"] >= far_future
    # req-unknown: not added.
    assert "req-unknown" not in w._reqs_to_send


# ===================================================================
# Worker: abort-cleanup ("CANCEL:") notifications
# ===================================================================


def _pull_worker_stub():
    """Bare pull worker for notif-path tests (no NIXL, no torch)."""
    from types import SimpleNamespace

    from vllm.distributed.kv_transfer.kv_connector.v1.nixl.pull_worker import (
        NixlPullConnectorWorker,
    )

    w = object.__new__(NixlPullConnectorWorker)
    w.nixl_wrapper = MagicMock()
    w.world_size = 1
    w._remote_agents = {"prefill-engine": {(0, 0): "agent_p0"}}
    w._reqs_to_send = {}
    w._reqs_to_process = set()
    w.consumer_notification_counts_by_req = defaultdict(int)
    topo = MagicMock()
    topo.tp_ratio.return_value = 1
    topo.block_size_ratio.return_value = 1
    topo.get_engine_info.return_value = SimpleNamespace(remote_block_size=16)
    w.transfer_topo = topo
    return w


def _read_empty_blocks(w, **kwargs) -> None:
    """Drive the len(local_block_ids) == 0 fast path of _read_blocks."""
    from vllm.distributed.kv_transfer.kv_connector.v1.nixl.tp_mapping import (
        ReadSpec,
    )

    w._read_blocks(
        read_spec=ReadSpec(remote_rank=0, local_block_ids=[], remote_block_ids=[]),
        dst_engine_id="prefill-engine",
        request_id="decode-req",
        remote_request_id="prefill-req",
        local_xfer_side_handle=0,
        remote_xfer_side_handle=0,
        **kwargs,
    )


def test_read_blocks_abort_cleanup_sends_cancel_notif():
    """Abort cleanup (never pulled) sends a CANCEL-tagged notification so the
    prefill side can tell it apart from a real pull completion."""
    w = _pull_worker_stub()
    _read_empty_blocks(w, abort_cleanup=True)
    w.nixl_wrapper.send_notif.assert_called_once_with(
        "agent_p0", notif_msg=b"CANCEL:prefill-req:1"
    )


def test_read_blocks_full_cache_hit_notif_untagged():
    """A genuine full prefix cache hit keeps the plain notification format."""
    w = _pull_worker_stub()
    _read_empty_blocks(w)
    w.nixl_wrapper.send_notif.assert_called_once_with(
        "agent_p0", notif_msg=b"prefill-req:1"
    )


def test_get_new_notifs_cancel_releases_lease():
    """A CANCEL notif for a leased request releases the lease immediately and
    reports the completion so the scheduler's deferred free can proceed."""
    w = _pull_worker_stub()
    w._reqs_to_send["prefill-req"] = time.perf_counter() + 60
    w._reqs_to_process.add("prefill-req")
    # Partial consumer count from an earlier, now-moot completion notif.
    w.consumer_notification_counts_by_req["prefill-req"] = 1
    w.nixl_wrapper.get_new_notifs.return_value = {"agent_p0": [b"CANCEL:prefill-req:1"]}

    assert w._get_new_notifs() == {"prefill-req"}
    assert "prefill-req" not in w._reqs_to_send
    assert "prefill-req" not in w._reqs_to_process
    assert "prefill-req" not in w.consumer_notification_counts_by_req


def test_get_new_notifs_cancel_unleased_is_ignored(caplog_vllm):
    """A CANCEL notif for a request that was never leased (e.g. aborted while
    another transfer was still in flight) reports no completion: reporting one
    would let the scheduler delete the request out from under that transfer."""
    w = _pull_worker_stub()
    w.nixl_wrapper.get_new_notifs.return_value = {"agent_p0": [b"CANCEL:ghost:1"]}

    with caplog_vllm.at_level(logging.DEBUG):
        assert w._get_new_notifs() == set()

    assert not [r for r in caplog_vllm.records if r.levelno >= logging.ERROR]
    assert w._reqs_to_send == {}
    assert w._reqs_to_process == set()
    assert "ghost" not in w.consumer_notification_counts_by_req


def test_get_new_notifs_normal_notif_unchanged():
    """An untagged pull-completion notif for a leased request is handled
    exactly as before."""
    w = _pull_worker_stub()
    w._reqs_to_send["prefill-req"] = time.perf_counter() + 60
    w._reqs_to_process.add("prefill-req")
    w.nixl_wrapper.get_new_notifs.return_value = {"agent_p0": [b"prefill-req:1"]}

    assert w._get_new_notifs() == {"prefill-req"}
    assert "prefill-req" not in w._reqs_to_send
    assert "prefill-req" not in w._reqs_to_process
