# Observatorio — design

Observatorio is a **Clio-style, privacy-preserving usage-insights pipeline**,
built as a proof of value on the public
[WildChat-1M](https://huggingface.co/datasets/allenai/WildChat-1M) dataset and
running entirely on open Qwen models. It answers *"what do people use AI
assistants for?"* with a browsable hierarchy of task clusters, and it never
shows an individual conversation.

The data source is a thin adapter, so the same pipeline can run on other chat
corpora; any stable pseudonymous user key can play the role of `hashed_ip`.

Reference: *Clio: Privacy-Preserving Insights into Real-World AI Use*
(Tamkin et al., 2024, [arXiv 2412.13678](https://arxiv.org/abs/2412.13678)).
Section numbers like §G.7 below refer to that paper's appendix.

## 1. Decisions

| Decision | Choice | Why |
|---|---|---|
| Dataset | `allenai/WildChat-1M` (837,989 convos), not `allenai/WildChat` | The 529k release has no user key; 1M has `hashed_ip`, needed for Clio's unique-user threshold. Both ODC-BY, ungated. |
| Sample | 100k random (reservoir, seed 42) → 45,410 unique `hashed_ip` | Clio's reference run size; ~half a day on one H100. |
| Language | **English everywhere** (prompts, summaries, labels) | Same as Clio. Collapses the same request across languages (~53% of WildChat isn't English) into one cluster. |
| LLM | **Qwen3.8-27B** for every generative step, thinking off | One model, one chat template, best quality available locally. Thinking would multiply output tokens 10–50×. |
| Embeddings | **Qwen3-Embedding-8B** | Strong multilingual/English embedder; summaries are short, so the 8B is cheap. |
| Serving | vLLM **offline** batch (`LLM.generate` / `LLM.embed`) | Batch job, no server or port to share. |
| Output | Aggregate-only | WildChat is public, but the point is a pipeline that is also safe on private corpora, where individual views aren't allowed. |

## 2. Pipeline

```
WildChat-1M ─► 0 ingest      sample · normalize · XML transcript (§G.2) · free facets
                  │           (hashed_ip, country, model, turns, language, moderation)
                  ▼
              1 extract     Qwen3.8-27B · per-conversation facets, PII-free, no proper nouns
                  │           request · task · concern(1–5)                ── privacy layer 1
                  ▼
              2 embed       Qwen3-Embedding-8B on the `request` summary
                  ▼
              3 cluster     k-means, k ≈ n/100 (~1,000 base clusters)
                  │         + 2-D UMAP (cosine, n_neighbors 15, min_dist 0; §G.6)
                  │         drop clusters with < 10 unique hashed_ip or < 20 convos
                  ▼                                                      ── privacy layer 2
              4 label       Qwen3.8-27B · contrastive naming (§G.5):
                  │         50 in-cluster + 50 nearest out-of-cluster summaries
                  ▼                                                      ── privacy layer 3
              5 hierarchize Qwen3.8-27B · §G.7: neighborhoods of ~40 → propose →
                  │         dedup → assign (shuffled) → rename; ~1,000 → ~150 → ~20
                  ▼
              6 audit       Qwen3.8-27B · 1–5 privacy rubric (§G.8), keep ≥ 3
                  ▼                                                      ── privacy layer 4
              7 publish     serve.duckdb with cluster-level tables only
                  ▼
              8 viewer      FastAPI + deck.gl: cluster map + tree, colour by facet
```

Every stage reads and writes Parquet under `runs/<run_id>/`, records its
parameters in `runs/<run_id>/config.json`, and is resumable (chunked outputs are
skipped on rerun).

### 2.1 Extract (stage 1)
- Clio's §G.4 extractor prompt, adapted to Qwen's chat format: the
  conversation is the first user turn, `"I understand."` the assistant turn, the
  facet question the second user turn, and Clio's assistant prefill is appended
  after the generation prompt.
- Because the conversation comes first, every facet for a conversation shares
  one prompt prefix. Requests are ordered so a conversation's 2nd/3rd facets are
  queued `lag` (256) conversations after its first, so vLLM's prefix cache
  serves them instead of re-reading the conversation.
- Transcripts over 4,000 tokens keep the first 3,500 and last 500 tokens (<1%
  of conversations; mean length is ~1,040 tokens).
- The PII / no-proper-nouns block is included only for open-ended facets, as in Clio.
- Temperature 0.2 (Clio's value), seed 0.

### 2.2 Privacy layers
1. **Summaries** are PII-free and have no proper nouns (prompted).
2. **Aggregation threshold**: a cluster is kept only with ≥ 10 unique
   `hashed_ip` **and** ≥ 20 conversations. Clio withholds its values; ours
   are k-anonymity-style defaults (k≈10).
3. **Cluster names/descriptions** are prompted to exclude private information.
4. **Auditor** rates each name + description 1–5 (§G.8 rubric) and drops
   anything below 3.

`hashed_ip` is an imperfect user key (shared IPs, VPNs), which is fine for a
proof of value; on another corpus, use its own pseudonymous user key.

## 3. Evaluation
- **Concern facet vs. ground truth**: WildChat ships OpenAI-moderation and
  Detoxify labels per message. Agreement between our 1–5 `concern` and
  `mod_flagged` / categories is a free check on the concern facet.
- **Privacy by stage** (Clio Fig. 5): run the auditor over a 5k sample of raw
  transcripts, then summaries, then cluster labels; report the share rated 1–2
  at each stage.
- **Reconstruction** (later): synthetic conversations with a known topic
  hierarchy, checking the pipeline recovers it (Clio: 94%).

## 4. Compute (single H100 NVL, 94 GB)
Measured on 3,000 WildChat-1M conversations with the Qwen3.8 tokenizer: mean
1,040 tokens (p50 584, p90 2.7k, p99 6.3k).

**Benchmark (2026-09-28, 1,000 conversations, bf16, 3 facets):** 1.82–1.92
conversations/s at ~8.1–8.6k prompt tokens/s (GPU compute-bound on reading
prompts; ~83 output tokens per conversation). Prompt-prefix reuse across facets
stays at ~18%. Qwen3.8 is a hybrid model, so vLLM 0.30 only keeps its
linear-attention state in coarse "align" mode (`mamba_cache_mode=all` is
unsupported and falls back to align). Moving the later facets closer (`lag`
256 → 32) raised reuse from 13% to 18%. In practice each facet re-reads the conversation.

| Stage | 100k (≈99k unique) |
|---|---|
| Extract, 3 facets (chosen: Clio-faithful) | **~14.5 h** measured rate |
| Embed | ~5–10 min |
| Cluster + UMAP | ~15–20 min |
| Label, hierarchy, audit | ~45 min |

Options not taken: dropping `task` (~9.5 h), one combined facet call (~5 h), FP8 (~1.6×, unmeasured).
Full WildChat-1M would be ~8× the extract time.

**Checkpointing:** extraction writes `parts/extract_NNNN.parquet` every 1,000
conversations (~9 min) and a stats line per chunk to `extract_stats.jsonl`;
rerunning `scripts/run_gpu.sh <run>` resumes from the last finished chunk.

## 4b. Data notes (found while benchmarking)
- **Duplicate conversations.** 1,549 `conversation_hash` values appear 13,219
  times across 6,984 IPs in WildChat-1M (shared templates, e.g. image-prompt
  generators); 949 duplicate rows in our sample. Extraction runs each hash once;
  duplicates keep counting toward cluster size and unique users.
- **Toxic conversations were removed from WildChat-1M**: `toxic` and
  moderation `flagged` are false for all 837,989 rows. The concern evaluation
  therefore uses the continuous OpenAI-moderation `category_scores`, not flags.
- The first 1,000 summaries were all English, none hit the length limit, and
  `request`/`task` often coincide. ~7% contain benign proper nouns (brands,
  products), which is what privacy layers 3–4 are for.

## 5. Layout
```
observatorio/                     code (uv project)
├── env.sh                        source first: pins caches, models and the venv to $OBS_ROOT (default ~/.observatorio)
├── observatorio/
│   ├── config.py                 paths, model ids, run config
│   ├── ingest.py                 stage 0
│   ├── prompts.py                prompts (adapted from Clio's appendix G) + JSON schemas
│   ├── llm.py                    vLLM helpers: chat + prefill, structured outputs, debug logging
│   ├── quality.py                deterministic quality gate for generated names/descriptions
│   ├── extract.py                stage 1 (GPU)
│   ├── embed.py                  stage 2 (GPU)
│   ├── cluster.py                stage 3: k-means + UMAP + privacy threshold (CPU)
│   ├── label.py                  stage 4: contrastive cluster naming (GPU)
│   ├── hierarchy.py              stage 5: multi-level hierarchy (GPU)
│   ├── audit.py                  stage 6: privacy auditor with a content-keyed cache (GPU)
│   ├── publish.py                stage 7: aggregates → serve.duckdb, last privacy gate (CPU)
│   ├── publish_schema.py         the serve.duckdb contract + check_schema()
│   ├── serve.py, web/index.html  stage 8: aggregate-only viewer
│   ├── export_static.py          static export of a publication for a server-less demo
│   └── eval_coherence.py         evaluation set for the hierarchy coherence judgment
├── scripts/
│   ├── run_full.sh <run>         everything: extract → embed → cluster → label → hierarchy → audit → publish
│   ├── run_gpu.sh <run>          extract → embed (checkpointed every 1k conversations)
│   ├── run_pipeline.sh <run>     embed → … → publish; FROM=<stage> resumes
│   ├── make_fixture.py           synthetic serve.duckdb for viewer development
│   ├── make_dev_run.py           CPU-only dev run from finished extract chunks
│   └── shoot.mjs                 Playwright screenshot of the viewer
├── tests/                        CPU-only unit tests (python -m unittest) + viewer tests (node --test)
└── docs/                         reviews of each run, evaluation set and rubric, images
$OBS_ROOT/                        everything heavy (not in git)
├── venv/ hf/ uv-cache/ python/ vllm-cache/ logs/
└── data/runs/<run_id>/           conversations.parquet, parts/, facets.parquet, embeddings.npy,
                                  clusters.parquet, labels.parquet, hierarchy.parquet, audit.parquet,
                                  serve.duckdb, config.json, debug/
```

## 6. Status
1. Ingest, extraction and embeddings for a 100k sample: done (validated first on a 10k sub-run).
2. Clustering, privacy threshold, contrastive labels, auditor, publish: done.
3. Hierarchy and viewer: done; several iterations, each reviewed in `docs/reviews/`.
   **v3 is the reference publication** (structured outputs, quality gate, fail-closed checks);
   v4's separate coherence check was not adopted (it over-dissolved groups; see the v4 review).
4. Open work: calibrate the coherence judgment on the labeled set in `docs/eval/` (labels not yet
   confirmed); calibrate the privacy auditor; extraction-level issues from the first review
   (instruction leakage, long multi-topic sessions, concern rubric).
5. Source adapters for other chat corpora.
