# Installation layout and RPATH policy v1

Executables, component libraries, headers, and versioned shared resources use
GNUInstallDirs. Stdlib, profiles, schemas, backends, docs/examples, and debug
metadata live below `${datadir}/cellerator`. Runtime lookup uses an
executable-relative `$ORIGIN` RPATH and never captures a staging or source path,
so DESTDIR and moved-prefix installs remain valid.
