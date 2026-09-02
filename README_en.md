<div align="center">

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="./website/assets/ouro/ouro-lockup-tagline-light.svg">
  <img src="./website/assets/ouro/ouro-lockup-tagline-ink.svg" alt="Ouro — State is all you need">
</picture>

</div>

<div align="center">

[中文](./README.md) | English

</div>

# Ouro

Ouro is a broadly recurrent architecture, but it is not a traditional token-by-token RNN. It combines chunk-local causal attention with three explicit forms of mutable state: recurrent vectors, a temporal state queue, and matrix fast state. Gridman is the byte-level language-model implementation used to study the architecture.

At a fixed model configuration, Ouro's persistent-state structure does not grow with processed history length. This is a property shared with stateful RNNs, not a claim that Ouro invented fixed-state recurrence. Ouro's research differences are its chunk-level computation, heterogeneous state topology, width-state scaling behavior, and a lifecycle contract that preserves state by default within a compatible model lineage.

## Evidence boundary

- `IMPLEMENTED`: state updates across input chunks, registered state in `state_dict`, explicit ablation reset, versioned checkpoint metadata, and separate Pretrain/SFT code paths.
- `PRELIMINARY`: one external 1280-width Pretrain checkpoint and its reported training run.
- `STATIC CONFIGURATION`: controlled width-family configurations that can be instantiated and counted but have not been trained.
- `PLANNED`: MaPhY Runtime factory state, snapshot, restore, fork, lineage management, and production isolation.
- `VALIDATED`: reserved for results that pass a predefined experiment and retain reviewable artifacts. No current width-scaling or long-memory claim uses this label.

The repository does not claim infinite context, lossless memory, constant total VRAM, permanent identity, consciousness, AGI, production readiness, or a proven scaling law.

## Ouro and typical RNNs

| Dimension | Typical RNN / LSTM / GRU | Ouro |
|---|---|---|
| Recurrent granularity | Usually token / timestep | Fixed small byte chunks |
| Local relationship modeling | Primarily recurrent updates | Chunk-local causal attention |
| State structure | Usually hidden/cell vectors | Vector state + temporal queue + matrix state |
| History-length relation | Fixed hidden shape at a fixed configuration | Fixed persistent-state topology at a fixed configuration |
| Scaling with width | Hidden state normally grows with model width | Vector state grows with `d`; matrix state grows with `d²` |
| Lifecycle | Reset policy is task/runtime dependent | Normal compatible-lineage paths preserve state by contract |

RNNs already have state. The comparison concerns state structure, computation granularity, and lifecycle. It does not claim that state alone makes Ouro superior to RNNs.

## State continuity and TBPTT

Ouro training uses **continuous state values with truncated gradient history**:

```text
forward state values: continue into the next chunk
gradient history: detach at controlled BPTT boundaries
```

`mem_detach()` does not reset values. Normal Pretrain, SFT, compatible post-training, and runtime entry points must not call `mem_clear()` implicitly. `reset_state_for_ablation()` is an explicit diagnostic operation.

A new SFT run loads a compatible Pretrain checkpoint. Resuming SFT loads an SFT checkpoint from the same lineage. A versioned checkpoint records the stage, step, model configuration, registered state, optimizer, scheduler, RNG state, lineage identifiers, data manifest identifier, and environment manifest. Loading an incompatible width is rejected; there is no implicit 1280→2624 state migration.

## Current reference checkpoint

| Field | Value | Status |
|---|---:|---|
| Model | Gridman-Naxi v0d1 Experimental Reference Model | `PRELIMINARY` |
| Stage | Pretrain checkpoint | Not an SFT result |
| Trainable parameters | 355,491,887 | Public model-size convention |
| Matrix-state elements | 9,830,400 | Structural count |
| Registered persistent-state elements | 15,400,960 @64 state slots | Structural footprint, not memory quality |
| Repository tracked scale | 365,322,287 | Technical-only convention: parameters + matrix state |
| Width / blocks / layers per block | 1280 / 2 / 4 | Current configuration |
| Patch / state slots / BPTT | 64 / 64 / 7 | Current configuration |
| Training run | ~7 GB; 2× RTX 4090; ~69 wall-clock hours / ~138 aggregate GPU-hours | Duration verified from the TensorBoard event span; corpus definition remains incomplete |
| Late-run train CE | 0.5620 nats per unmasked target token | Mean of the final 500 TensorBoard points; preliminary |

The checkpoint exists outside this repository. A public artifact still needs a stable download location, license, checksum, data manifest, and complete environment manifest.

## Controlled width-state family

Only `embed_dim` changes in this family. `blocks=2`, `block_layers=4`, `patch_size=64`, `state slots=64`, and `bptt_size=7` stay fixed.

| Width | Trainable parameters | Matrix state | All persistent state @64 | Evidence |
|---:|---:|---:|---:|---|
| 512 | 58.05M | 1.57M | 3.80M | `STATIC CONFIGURATION · NOT TRAINED` |
| 768 | 126.79M | 3.54M | 6.88M | `STATIC CONFIGURATION · NOT TRAINED` |
| 1280 | 355.49M | 9.83M | 15.40M | `WORKING CHECKPOINT · PRELIMINARY` |
| 1856 | 744.24M | 20.67M | 28.75M | `STATIC CONFIGURATION · NOT TRAINED` |
| 2624 | 1.483B | 41.31M | 52.73M | `PLANNED MODEL LINEAGE` |

These are structural counts for one model object configured with 64 state slots. They are not validated memory-quality scores. The intended study is to characterize width-state scaling behavior, not to assert a scaling law before controlled training, multiple seeds, and error intervals exist.

## Repository contents

- Native PyTorch implementation of Ouro-Naxi and Gridman.
- ByteTokenizer and streaming data loader.
- Pretrain / SFT training code paths with explicit checkpoint-stage semantics.
- DDP/NCCL matrix-state synchronization code path; correctness and scaling efficiency still require systematic validation.
- Research notes, experiment protocol, evidence ledger, public claims manifest, and width-family manifest.
- Website and NVIDIA Inception submission materials that use the same evidence labels.

The repository does not currently provide an auditable full training-data package or a public checkpoint artifact with complete release metadata.

## State, Self, memory, and instances

- **State** means Ouro's current mutable internal state.
- **Self** is the product-language name for that state as a whole; it does not imply consciousness, personality, or psychological identity.
- **Matrix state** refers to the mutable `mem` buffers; element count is not equivalent to validated long-term memory quality.
- **Model lineage** means a compatible architecture, width, state topology, queue length, patch size, and tokenizer carried across training stages.
- **Instance** currently means an independent model object with its own state. Factory state, snapshot, restore, fork, and production lineage management are planned MaPhY Runtime capabilities.

The 355M and planned 1.5B models use different widths and therefore belong to different model lineages.

## License

The source code is released under the [Apache License 2.0](LICENSE). Dataset and checkpoint release terms must be documented separately when those artifacts are published.
