# Superseded job reconciliation

The final owner-scoped audit found three superseded jobs still marked blocked
after their earlier precondition failures. Each had verified resource cleanup;
none was needed for current qualification. Owner `cancel` initially returned
unknown because no execution handle existed. Supported owner `recover` then
fenced/reconciled each to `cancelled` with `cleanup_verified=true`.

The exact cancel and recover receipts are retained here. This audit covers only
the three listed sparse-support jobs, and makes no claim about unrelated jobs
or global scheduler state. The subsequently admitted robust performance run has
its own final receipt under its experiment record.
