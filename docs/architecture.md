# CA-HNM architecture and controls

## Offline mining pipeline

For each query, the implementation:

1. retrieves the top `K` documents with BM25 and the initial dense encoder;
2. forms their union and removes every judged positive;
3. resolves the query's target concept and extracts its local typed context;
4. applies deterministic target, relation, scope/granularity, context, and level
   mismatch rules;
5. excludes target matches, ambiguous candidates, and candidates without an
   explicit mismatch;
6. groups eligible negatives by their primary violation type and selects them
   round-robin while preserving candidate order within each group; and
7. materializes query-positive-negative examples for a standard dual encoder.

Structured information changes only the negative set. It is not provided to the
encoder during training or inference.

## RQ1: pure selection and rank-matched control

`CA-HNM-full` is the implementation name for CA-HNM-Pure. It retains at most four
constraint-aware negatives per query in the paper configuration.

`CA-HNM-rank-matched` first mines the CA-HNM-Pure set. For every selected
constraint-aware item, it chooses one unused, non-positive candidate from the
same merged retrieval pool. The merged list is ordered by descending maximum of
the raw BM25 and dense scores; document ID breaks equal-score ties. The
implementation then minimizes, in order:

1. mismatch with the stored retrieval-source signature;
2. absolute distance from the selected item's merged rank;
3. the candidate's merged rank; and
4. document ID.

Selection is without replacement and every CA-HNM-Pure document is excluded from
the control. This is a retrieval-position control, not a match on all candidate
properties.

## RQ2: mixed negative pools

`CA-HNM-mixed` combines random, BM25, dense, and constraint-aware components.
`CA-HNM-matched-mixed` replaces only the constraint-aware component with the
rank-matched component.

Components are traversed round-robin after duplicate removal until the
four-negative budget is filled or all components are empty. Component keys are
sorted deterministically. With all components available, this gives one item per
component in the first pass. Empty or duplicate components leave capacity for
later passes.

## RQ3: structural sensitivity

| Paper variant | Implementation | Operation |
| --- | --- | --- |
| Label-only | `CA-HNM-label-only` | Retains ontology nodes/labels but removes relations. |
| Shuffled graph | `CA-HNM-shuffled-graph` | Deterministically shuffles relation targets within each relation type while preserving typed in/out degree marginals. |
| No structure | `CA-HNM-no-ontology` | Selects retrieval candidates without ontology constraints. |

The shuffle uses the mining seed (13 in the paper) and is fixed because mining is
performed once before the ten training-seed runs. These variants can change both
structured evidence and candidate membership; they are end-to-end sensitivity
analyses rather than fixed-candidate causal interventions.

## Metrics and inference

The reported metrics are NDCG@10, MRR@10, MAP@100, and Recall@100. Both datasets
use binary qrels. Paired CA-HNM-Pure-minus-rank-matched effects are computed by
training seed. Hierarchical 95% bootstrap intervals resample seeds and queries.

## Reproducibility invariants

- train, development, and test queries are target-disjoint;
- judged positives are never eligible as explicit negatives;
- the same mined pools are reused across training seeds;
- all compared strategies use the same explicit-negative budget;
- BGE/cached-MNRL values repeated across paper tables come from the expanded
  execution; and
- model checkpoints are unnecessary for checking the published aggregate values,
  but are required to re-evaluate rankings from scratch.
