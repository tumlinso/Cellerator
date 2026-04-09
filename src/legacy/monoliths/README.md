# `legacy/monoliths`

Reserved home for the large historical entrypoints and all-in-one source files.

For now those files still live at their existing paths so callers do not break,
but new reusable compute should be extracted into `src/compute/*` instead of
growing the monoliths further.
