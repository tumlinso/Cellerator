# Package manifest and resource hashes v1

The installed manifest begins with CEIR, profile, and schema revisions, then
records every regular installed resource as a lexically sorted relative path
and lowercase SHA-256. Backend identity metadata and standard-library files use
the same record form. Absolute prefix, timestamps, and directory enumeration
order are excluded so equal clean installations compare byte-for-byte.
