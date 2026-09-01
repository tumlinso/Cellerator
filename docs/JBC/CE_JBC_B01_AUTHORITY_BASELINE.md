# CE-JBC-B01 authority baseline

This record freezes the live inputs for the Cellerator side of the joint
biological compiler bootstrap.  It is evidence, not a claim that Cellerator
and CellShard can be observed atomically.

## Observation 1

Project Control observed the two registered workspaces independently at
`2026-09-01T05:45:30Z` (Cellerator) and `2026-09-01T05:45:31Z`
(CellShard), with one second of reported cross-workspace skew.

| Authority | Source cursor | Todo/workflow cursor | Worktree state | Active JBC claim |
| --- | --- | --- | --- | --- |
| Cellerator | `d735173a8fefcab127a0d742efc0b17355c4550a` | revision `3603`; semantic/workflow fingerprint `770c4974b05c415d08cf65d35ad2bfbc9a662f73003f61d0dc94cc7c09ad4d12` | `wt-d3fea47b1244da16`; fingerprint `4b0f96205da8940d2d100ef504d681595d889d69e3cc6025c639e8f732d09f6b`; dirty because the nested CellShard authority had an active claim | `CE-JBC-RUN-V1 / CE-JBC-L-BOOTSTRAP / CE-JBC-B01` |
| CellShard | `5f6a502b4355732c4ed3cc873a25b8aec66d8338` | revision `314`; semantic/workflow fingerprint `c5aaf9ccece7fe50283b8052b860b64d868cf8e5301e307ca45002c1c3f372c6` | `wt-b27d2b8b5924701d`; clean fingerprint `e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855` in the CellShard observer | `CS-JBC-RUN-V1 / CS-JBC-L-BOOTSTRAP / CS-JBC-B01` |

The parent dirty state is not normalized into CellShard cleanliness.  It is a
cross-authority effect of CellShard's independently claimed workflow and is
recorded separately from either source commit.

## Observation 2

Project Control repeated the independent read at `2026-09-01T05:46:06Z`
(Cellerator) and `2026-09-01T05:46:07Z` (CellShard), again with one second of
reported skew.

| Authority | Source cursor | Todo/workflow cursor | Worktree state | Active JBC claim |
| --- | --- | --- | --- | --- |
| Cellerator | `d735173a8fefcab127a0d742efc0b17355c4550a` | revision `3603`; semantic/workflow fingerprint `574e0acb1ec148ca930bafb25e0ca0765581deb2e72197046142ae968b9552d5` | `wt-d3fea47b1244da16`; fingerprint `4b0f96205da8940d2d100ef504d681595d889d69e3cc6025c639e8f732d09f6b`; parent still reports the claimed submodule dirty | `CE-JBC-RUN-V1 / CE-JBC-L-BOOTSTRAP / CE-JBC-B01` |
| CellShard | `5f6a502b4355732c4ed3cc873a25b8aec66d8338` | revision `314`; semantic/workflow fingerprint `14b07cee56f5cc1e754f5ac46ddc2c2ab6b2f4a452b1df93da7c78ac1701b208` | `wt-b27d2b8b5924701d`; clean fingerprint `e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855` in the CellShard observer | `CS-JBC-RUN-V1 / CS-JBC-L-BOOTSTRAP / CS-JBC-B01` |

The semantic fingerprints differ between observations while the revisions do
not because active claim/session state is included in the live workflow read.
The revisions and each observation's fingerprint are therefore retained as a
pair; neither fingerprint supersedes the other.

## Frozen interfaces and interlock

Cellerator currently publishes no JBC interface.  CellShard retains these
frozen CS-FOUND interfaces for compatibility evidence:

| Interface | Version | Content hash |
| --- | --- | --- |
| `CS-FOUND-I1` | 1 | `7ecf116c9568d646b180e28a59ddf1ab9adbd10017a29b2a26a12255842adb30` |
| `CS-FOUND-I2A` | 1 | `217a095db360a5ff8ac6fbc9edc9e20acbf2833463c7c988be124c6ef4efe4f3` |
| `CS-FOUND-I2B` | 1 | `7c86e72818699a566787ed1561d97f8448ce199974b827a81712f4ff5777696a` |
| `CS-FOUND-I2C` | 1 | `840ba657df09cfac3a186d23f088da5bfa99a1df5bbcf66c11cf6dae8052330c` |
| `CS-FOUND-I3A` | 1 | `b09fd2ca65112edfea3739199be618a0b2b547550733b0d01ce771ba658b0914` |
| `CS-FOUND-I3B` | 1 | `32c45f1dc09643d711f04e7e9f086cda22f25a67b986c8e34e4bad3d97bd229a` |
| `CS-FOUND-I4` | 2 | `32c45f1dc09643d711f04e7e9f086cda22f25a67b986c8e34e4bad3d97bd229a` |
| `CS-FOUND-I5A` | 1 | `3d5c4533eeb14b93bddc85067338cea5745790c81bff06407489411f902df82e` |
| `CS-FOUND-I5B` | 1 | `e5a76d38652886e16a8b7b06c3068c0fac5b1736b2216465d7e82d1386210b24` |

`CE-GEO-COMPLETE` and `CE-EXOP-COMPLETE` are reached, but the
human-controlled `CE-AMP-PERMISSION` decision remains `not_granted` at
revision `3281`.  Consequently `CE-AMP-00` remains planned and inactive.  JBC
does not change, claim, or repair that program.

## Transition classification

- Preserve the biological identity, order, structure/value separation,
  semantic-geometry, complete-cost planner, and direct-native-execution
  contracts already owned by Cellerator.
- Extend those contracts adjacently with atom coverage, fragments, partial
  algebra, external cost, and bounded compiler-exchange surfaces.
- Keep retained CP-Math v1 and legacy formats as compatibility evidence while
  new ownership converges on the operation core; do not grant them new
  authority.
- Treat the registered CellShard checkout as an adjacent privileged compiler
  component for the JBC program while preserving CellShard ownership of
  persistence, delivery, placement, and global scheduling.
- Preserve the CE-AMP interlock unchanged and ignore non-JBC runs except as
  source evidence.

No runtime behavior, wire format, historical program, or non-JBC authority is
changed by this baseline.
