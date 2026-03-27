# Architecture Plan: Scaling Generate to 10+ Concurrent Users

## The Problem

The `GeneratorPool` approach works correctly — each slot gets its own model instance so there are no race conditions. But it's memory-expensive: every `ClimbDDPMGenerator` carries its own `Noiser` weights (~100 MB), so `GENERATOR_POOL_SIZE=10` costs ~1 GB RAM before serving a single request.

The root cause: we're treating **model instances** as the scarce resource when the actual scarce resource is **compute time**. Ten model copies don't run any faster than one; they just consume 10× the memory.

---

## The Key Insight

During inference, the DDPM model weights are **read-only**. The `ddpm.eval()` call disables all mutable state (dropout, BatchNorm running stats). The only mutable per-request state is:

- `noisy` — the current noise tensor for this request's climbs, shape `[n, 20, 12]`
- `cond_t` — the grade/angle condition tensor, shape `[n, cond_dim]`
- `t_tensor` — the current diffusion timestep, shape `[n, 1]`

These are local variables inside `generate()`. They are already different per-call. Multiple requests don't need separate model instances — they just need separate *tensors*.

This means we can run a **single model** and merge tensors from multiple concurrent requests into one batched forward pass. PyTorch's matrix operations handle arbitrary batch sizes transparently.

This is the standard approach in production ML serving systems (vLLM, TorchServe, NVIDIA Triton) and is called **dynamic batching**.

---

## Option A: Async Serial Queue (Simple)

Keep one model. Feed all requests through a single async queue processed one at a time.

```
Request 1 ──┐
Request 2 ──┤──► asyncio.Queue ──► single model ──► results
Request 3 ──┘
```

**Implementation** (~40 lines):
- Replace `GeneratorPool` with a single generator instance
- Add an `asyncio.Queue` and a background asyncio task that drains it
- Each request submits an `asyncio.Future` and awaits it
- The background task calls `gen.generate()` and sets futures

**Memory:** ~100 MB (1 model)
**Latency:** Serial. If each request takes T seconds, the Nth user waits (N-1)×T before their request even starts. With 10 users at 3s each: user 10 waits 27s in the queue, then 3s to generate = 30s total.
**Code change:** Small (~40 lines, no changes to DDPM internals)
**Verdict:** Solves the memory problem completely, but latency degrades linearly with queue depth. Acceptable for light/occasional usage, unacceptable if multiple users actually press generate simultaneously.

---

## Option B: Dynamic Request Batching (Recommended)

One model. When multiple requests are pending simultaneously, merge their tensors along the batch dimension and run the DDPM loop once for all of them. Split the result and deliver each user's slice.

```
User A (3 climbs) ──┐                              ┌── 3 climbs to A
User B (2 climbs) ──┤──► merge ──► DDPM(5 climbs) ──┤
User C (1 climb)  ──┘                              └── 1 climb to C + 2 climbs to B
```

The DDPM core loop runs once with `batch_size = 3+2+1 = 6`. After the loop, results are split by user.

### Why this is straightforward

Looking at the current `generate()` loop:

```python
# Each timestep:
gen_climbs = self.ddpm.predict_cfg(noisy, cond_t, t_tensor, guidance_value)
projected  = self._project_onto_manifold(gen_climbs, offset_manifold)
noisy      = self.ddpm.forward_diffusion(gen_climbs, t_tensor, ...)
```

- `predict_cfg` and `forward_diffusion` already operate on the full batch dimension and have no coupling between batch items. Concatenating tensors from different users is trivially valid.
- `cond_t` is already per-climb (grade/angle differ per climb within one request). Concatenating conditions from different users is equally valid.
- `_project_onto_manifold` uses a layout-specific manifold. If users share a layout (common for a homewall app), this is also directly batchable. If users are on different layouts, the projection step runs per-layout sub-slice (a small loop over layout groups).

### Parameters that are currently scalars but need to become per-item

| Parameter | Current | Batched |
|-----------|---------|---------|
| `guidance_value` | scalar float | tensor `[B, 1, 1]` — 1-line change to `predict_cfg` |
| `t_start_projection` | scalar float | easiest to normalize to default; or per-item mask |
| `timesteps` | int | must match across batched requests (normalize to default, or group by value) |

The only real constraint is **`timesteps`**: all requests in one merged batch must run the same number of denoising steps, because all items in the batch share the same loop counter. In practice, almost all requests will use the default (100). Requests with custom `timesteps` can either be normalized or excluded from the batch and run serially after.

### How the batching coordinator works

```python
@dataclass
class PendingGenRequest:
    layout_id:          str
    noisy:              Tensor        # [n, 20, 12]
    cond_t:             Tensor        # [n, cond_dim]
    n:                  int
    guidance_value:     float
    x_offset:           float | None
    timesteps:          int
    t_start_projection: float
    future:             asyncio.Future

class DynamicBatchRunner:
    BATCH_WINDOW_MS = 50   # wait up to 50ms to collect a batch
    DEFAULT_TIMESTEPS = 100

    def __init__(self, generator: ClimbDDPMGenerator):
        self._gen = generator
        self._queue: asyncio.Queue[PendingGenRequest] = asyncio.Queue()

    async def submit(self, req: PendingGenRequest) -> list:
        await self._queue.put(req)
        return await req.future

    async def run(self):
        """Background asyncio task — call once at startup."""
        while True:
            batch = [await self._queue.get()]
            deadline = asyncio.get_event_loop().time() + self.BATCH_WINDOW_MS / 1000
            while asyncio.get_event_loop().time() < deadline:
                try:
                    batch.append(self._queue.get_nowait())
                except asyncio.QueueEmpty:
                    await asyncio.sleep(0.005)

            # Partition by timesteps (only merge same-timestep requests)
            groups: dict[int, list[PendingGenRequest]] = {}
            for req in batch:
                groups.setdefault(req.timesteps, []).append(req)

            loop = asyncio.get_event_loop()
            for timesteps, group in groups.items():
                await loop.run_in_executor(None, self._process_group, group, timesteps)

    def _process_group(self, reqs: list[PendingGenRequest], timesteps: int):
        """Merge tensors, run DDPM, split results. Runs in a thread."""
        # Merge along batch dim
        all_noisy = torch.cat([r.noisy for r in reqs], dim=0)
        all_cond  = torch.cat([r.cond_t for r in reqs], dim=0)
        all_gv    = torch.tensor([r.guidance_value for r in reqs]
                        ).repeat_interleave(torch.tensor([r.n for r in reqs]))
                        .view(-1, 1, 1)   # [total_n, 1, 1]

        t_start = reqs[0].t_start_projection  # normalize or take first

        # --- DDPM loop (same logic as ClimbDDPMGenerator.generate) ---
        t_tensor = torch.ones((all_noisy.shape[0], 1))
        for _ in range(timesteps):
            gen_climbs = self._gen.ddpm.predict_cfg(all_noisy, all_cond, t_tensor, all_gv)
            if t_tensor[0].item() < t_start:
                # Project per-layout group
                offset = 0
                for req in reqs:
                    sl = slice(offset, offset + req.n)
                    manifold = self._gen._get_offset_manifold(req.layout_id, req.x_offset)
                    alpha    = self._gen._projection_strength(t_tensor[sl], t_start)
                    proj     = self._gen._project_onto_manifold(gen_climbs[sl], manifold)
                    gen_climbs[sl] = alpha * proj + (1 - alpha) * gen_climbs[sl]
                    offset += req.n
            t_tensor -= 1.0 / timesteps
            all_noisy = self._gen.ddpm.forward_diffusion(gen_climbs, t_tensor,
                            torch.randn_like(all_noisy))

        # Split and deliver
        offset = 0
        for req in reqs:
            sl = slice(offset, offset + req.n)
            manifold = self._gen._get_offset_manifold(req.layout_id, req.x_offset)
            if req.x_offset is None:
                result = self._gen._project_onto_indices_with_translation(
                    gen_climbs[sl], manifold, req.layout_id)
            else:
                result = self._gen._project_onto_indices(
                    gen_climbs[sl], manifold, req.layout_id)
            req.future.get_loop().call_soon_threadsafe(req.future.set_result, result)
            offset += req.n
```

**Memory:** ~100 MB (1 model)
**Latency for single user:** identical to today — they're the only request in their batch
**Latency for N simultaneous users with the same `timesteps`:** all N users get results at the same time after ≈ `N × single_request_time / parallelism`. On CPU, linear with batch size, so total compute is roughly the same as serial — but all users wait the same amount instead of user N waiting N× longer.
**Code change:** ~100 lines. Requires a small change to `predict_cfg` (scalar `guidance_value` → broadcastable tensor); the rest of the change is additive.

---

## Comparison

| Approach | Memory | Single-user latency | 10-user latency | Code added |
|----------|--------|--------------------|--------------------|------------|
| Current pool (10 instances) | ~1 GB | Same | Same (concurrent) | Already done |
| Option A: Serial queue | ~100 MB | Same | User 10 waits 9× | ~40 lines |
| **Option B: Dynamic batching** | **~100 MB** | **Same** | **All wait ≈ same (fair)** | **~100 lines** |

---

## Recommendation

**Implement Option B (dynamic batching).**

The reason Option A falls short: if the product is working well and multiple users are generating at the same moment, linear queue wait is a bad user experience. Option B handles this fairly — no user's experience degrades just because others are using the app.

The implementation fits the "lean" goal: ~100 lines of new code, no new dependencies, no infrastructure changes. The one DDPM internal change (`guidance_value` as a broadcastable tensor in `predict_cfg`) is a 1-line diff.

The batch runner starts as an asyncio background task in `app/main.py` alongside `init_db()`. The `DynamicBatchRunner` replaces `GeneratorPool` entirely (back to a single model instance).

### One edge case: `deterministic=True` requests

Deterministic generation seeds the generator's `torch.Generator`. In a merged batch, per-request seeding would need careful handling (the `deterministic_noise_generator` is per-`ClimbDDPMGenerator` instance). **Simplest fix:** deterministic requests bypass the batch runner and execute directly on the single model with a short lock, since deterministic generation is a power-user feature unlikely to be used by multiple users simultaneously. Alternatively, initialize the noisy tensor externally (before submitting to the batch runner) so seeding is done by the caller before the request enters the queue.

### Fallback / phased approach

If the dynamic batching refactor is more than you want to take on right now, Option A (serial queue) is a valid stepping stone. It solves the memory problem immediately with minimal code, and Option B can be layered on top later without changing the public API.
