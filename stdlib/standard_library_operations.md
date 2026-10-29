# Operation wrappers v1

`operations.cell` expresses the eleven standard operation constructions as
constexpr, inlineable descriptors over base semantic operation kinds. Wrappers
add no storage, dispatch, conversion, or policy and therefore disappear before
Semantic IR or lower transparently to the identical typed operation.
