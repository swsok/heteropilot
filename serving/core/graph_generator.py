import os
from time import time
from .request import *
from .logger import get_logger
from .run_paths import input_path

logger = get_logger("GraphGenerator")

_LLM_CONVERTER = None


def _llm_converter():
    """Import LLMConverter once, on first use (deviations.md D27).

    Deferred rather than imported at module scope so that importing `serving`
    stays cheap for callers that never build a graph -- the planner imports
    `serving.core.memory_model` for its own feasibility arithmetic.
    """
    global _LLM_CONVERTER
    if _LLM_CONVERTER is None:
        from chakra.src.converter.llm_converter import LLMConverter
        _LLM_CONVERTER = LLMConverter
    return _LLM_CONVERTER


def generate_graph(batch, hardware, num_npus, node_id=0, instance_id=0, npu_offset=0, enable_local_offloading=False, event=False, workload_name=None, inputs_root=None, cleanup_trace=True):

    cwd = os.getcwd()
    if inputs_root is None:
        inputs_root = os.path.join(cwd, "inputs")

    if event:
        file_name = 'event_handler'
    else:
        file_name = f'{hardware}/{batch.model}/instance{instance_id}_batch{batch.batch_id}'

    # For DP groups, all instances write .et files to a shared workload folder
    output_name = workload_name if workload_name else file_name

    trace_path = input_path(inputs_root, "trace", f"{file_name}.txt")
    output_path = input_path(inputs_root, "workload", output_name, "llm")
    workload_dir = os.path.dirname(output_path)
    os.makedirs(workload_dir, exist_ok=True)

    # In-process, not a subprocess (deviations.md D27). This is called once per
    # instance per iteration, and profiling put it at 28.9 % of wall time at 10 rps
    # -- of which only 7.4 ms per call is the conversion itself and 38.6 ms is fork,
    # exec, interpreter start and imports. Calling the converter directly returns
    # about 24 % of total wall time (docs/sim_cost_profile.md).
    #
    # LLMConverter directly rather than converter.main(), which is what makes this
    # safe as well as fast: main() parses sys.argv and calls setup_logging(), whose
    # logging.basicConfig would reconfigure the *frontend's* root logger on every
    # call. convert_llm() is a three-line wrapper around this constructor, so
    # nothing is skipped. Verified byte-identical to the subprocess path, and
    # unchanged across repeated calls -- the converter holds no module-level state.
    #
    # D26 is subsumed: there is no longer an interpreter to resolve. It stays in the
    # ledger because a node whose venv lacks the right chakra now fails at import
    # here rather than silently converting with the wrong one.
    logger.debug("Generating graph: input=%s output=%s num_npus=%s npu_offset=%s offload=%s",
                 trace_path, output_path, num_npus, npu_offset, enable_local_offloading,
                 extra={"node_id": node_id, "instance_id": instance_id})

    _llm_converter()(
        trace_path, output_path, num_npus, npu_offset, enable_local_offloading
    ).convert()
    if cleanup_trace:
        try:
            os.remove(trace_path)
        except FileNotFoundError:
            pass
    return
