# Stage E.6 bounded HTGNN resource pilot

Stage E.6 measures the complete untrained frozen production path on bounded
real train data: D4 lazy access, D6 logical batching, E.2 `HeteroData`
conversion, host-to-device transfer, and the E.3/E.4/E.5 classifier forward.
It does not alter the frozen architecture, train, prune, compute predictive
metrics, choose a training batch size, or inspect labels for optimization.

The default pilot uses one deterministic capture-order-spanning batch of 32
real samples for each of E1, E2, and E3. Each CUDA mode has five warmup calls
and twenty measured calls. The harness refuses batch sizes above 64, more than
10 warmups, or more than 50 measured calls. It opens only the three train
memory maps and requests exactly the selected samples through D4/D6; it never
iterates or materializes a split.

The modes are measured separately:

1. frozen E.2 CPU conversion, with its first call separated from repeated
   calls and process RSS retained only as noisy observational context;
2. synchronous CPU `HeteroData` to CUDA transfer, excluding preparation of the
   protective CPU clone from the timed interval;
3. complete normal E.5 forward; and
4. complete attention-return E.5 forward.

CUDA wall timing uses a monotonic clock with explicit synchronization directly
before and after every measured operation. Warmup precedes measurement. CUDA
peak allocated and reserved counters are reset for each mode. Reports retain
the median, nearest-rank p95, minimum, maximum, and raw bounded samples without
claiming precision or portability beyond this machine.

Normal and attention-return logits use bitwise equality on CPU. On CUDA they
use the frozen E.5 comparison policy (`rtol=1e-5`, `atol=1e-6`) because repeated
PyG scatter reductions can differ at float32 rounding scale even for the same
mode and unchanged model state. The observational report separately records
whether a measured pair was bitwise equal and its maximum absolute difference;
those noisy numerical details are not canonical contract facts.

`results/stage_e6/htgnn_resource_pilot_contract.json` is canonical evidence.
It contains environment/configuration identity, sample bounds, parameter
counts, shapes, finiteness, normal-versus-attention equality, topology and
source immutability, integrity results, and test outcomes. Runtime timings and
memory observations are deliberately excluded from its canonical hash.

`results/stage_e6/htgnn_resource_pilot_observations.json` is explicitly
non-canonical hardware-specific observational evidence. It contains the noisy
latency, throughput, RSS, CUDA allocation/reservation, and frozen E.2
provenance-validation component measurements. The latter is measured, not
optimized or cached; the report recommends investigation before E.7 only when
its median is at least 0.05 ms and at least 5% of repeated conversion median.

The pilot command is:

```powershell
.\.venv\Scripts\python.exe scripts\run_stage_e6_resource_pilot.py --run-tests
```

A passing bounded forward pilot establishes operational feasibility only for
the stated batch, hardware, software environment, and untrained forward path.
It does not establish training feasibility, predictive performance, an optimal
batch size, or behavior on another GPU.
