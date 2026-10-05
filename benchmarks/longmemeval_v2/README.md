# LongMemEval-V2 frozen adapter

Core retrieval snapshot: `v2-untouched-core-20261006` / `8aa584d523f587d9a918f9ec9321ffe90532c50a`.

The adapter is schema-only glue. It consumes trajectory fields (`goal`, `outcome`, `url`, `action`, `thought`, `accessibility_tree`) and never reads question ids, answers, eval functions, or gold evidence.

Default query policy is fixed before V2 scoring: hybrid vector+BM25, candidate_k=64, CE rerank, top_k=16, max 4 chunks per trajectory. The first official V2 result must be recorded before any parameter changes.
