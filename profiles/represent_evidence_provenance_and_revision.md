# Evidence provenance and revision v1

Provenance independently identifies dataset, source, sampling method, observation
window, transformation stage, producer/tool version, confidence, revision, and a
validity-predicate set. Its cache identity includes every evidence metadata field
but keeps the semantic subject identity separate. Any evidence revision or method
change therefore invalidates cached evidence without changing semantic identity.
