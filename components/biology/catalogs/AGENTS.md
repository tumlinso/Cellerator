# Biology reference catalogs

Inspect the release manifest, dictionary inputs, generator, and tests for
routine work. Generated catalog tables are bulk data in the build directory.
Open a generated table only to debug a specific record or generation failure.
Keep annotation downloads and generated tables out of the source tree.

Pin the source release and its checksum in the manifest. Dictionary lines use
exact canonical gene IDs from that release. Keep dictionary order explicit.
Generated indices address catalog rows. Model token assignments, packing
positions, and external dataset column positions have separate owners.
