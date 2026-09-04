# Sectioned profile storage v1

`CELLPRF1` stores a fixed 112-byte header, a 64-byte aligned section directory,
and individually aligned payloads. Directory entries carry stable 128-bit
identities, schema versions, stored and logical sizes, compression identifiers,
and checksums. The complete image also has a checksum whose field is treated as
zero during calculation.

The builder writes into caller-owned aligned memory. The validator produces a
non-owning view over the same bytes, so a read-only memory mapping needs no
relocation or allocation. Known required sections are validated strictly.
Unknown sections are accepted only when marked optional and remain discoverable
by numeric kind, allowing newer producers to extend the format without making
older readers interpret unknown evidence. Compression is explicit: uncompressed
sections require equal stored and logical sizes; compressed payload bytes remain
owned by the selected codec and are never silently decoded.

Validation fails closed for bad charter identity, endian or alignment markers,
truncation, directory bounds, duplicate stable identities, unsupported required
sections, inconsistent compression sizes, per-section corruption, and whole-image
corruption.
