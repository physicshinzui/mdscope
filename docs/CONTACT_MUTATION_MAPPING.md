# Contact residue correspondence for point mutations

## Scope

This note records how `mdscope` matches residues between the two groups in a
contact comparison when their sequences contain a point mutation, such as
WT R139 versus R139C C139.

## History of the R139C issue

The initial contact implementation (`04df41c`) identified corresponding
residues by `(chain, topology resid)`. It therefore allowed the residue name
to differ between WT and R139C.

While support for trajectory-specific selections and biological residue
numbering was added in `8038196`, the compatibility check was changed to
`(chain position, residue name)`. Requiring the residue name was unintended:
WT ARG139 and mutant CYS139 occupy the same sequence position but were judged
incompatible. A fresh run consequently stopped at the first mutant
trajectory, `R139C_r01`, with:

```text
RuntimeError: Contact residue positions do not match reference trajectory: R139C_r01
```

The earlier analysis had succeeded because its archived contact results were
generated on 2026-07-07, before this change. During later reproduction,
configuration changes invalidated the cached step and forced contact
occupancies to be recalculated, exposing the regression.

## Corrected correspondence rule

The compatibility signature is now `(chain, chain_position)`, where
`chain_position` is the one-based ordinal position of a residue within its
selected chain. Thus, compared trajectories must have the same selected
chains, residue counts, and residue order, but the residue name at a
corresponding position may differ.

Residue names are still stored separately for each group. Consequently, the
comparison tables preserve the identity change and can label the mutation as
`R139→C`; allowing the name difference does not erase it.

This rule is suitable for point substitutions in sequences of equal length.
Insertions, deletions, chain rearrangements, or selections with missing
residues require an explicit residue mapping and are not safely handled by
ordinal chain positions alone.

## Validation

- The contact regression test constructs an independent comparison
  trajectory with an ALA-to-CYS substitution at the same chain position.
- The complete `reproduce_all.sh --resume` workflow succeeded end to end on
  2026-07-23 after the correction.
- The `mdscope` test suite completed with 26 passed tests and 1 skipped test.
- Canonical WT-versus-R139C contact outputs were regenerated after the fix.

## Scientific impact

The correction changes only the cross-trajectory compatibility check. Contact
atom selection, distance cutoff, periodic-boundary treatment, sequence
separation filter, occupancy calculation, replicate-level statistics, and
ranking criteria are unchanged.
