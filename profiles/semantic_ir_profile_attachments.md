# Semantic IR profile attachments v1

Semantic IR nodes refer to profile environments, states, and evidence only by
stable identities. Small evidence may be embedded by offset and size; large
evidence uses an external artifact identity. The fixed record is pointer-free,
path-free, trivially serializable, and rejects mixed embedded/external forms.
