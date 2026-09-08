# Cross-repository dependencies and closure

## Why there are two packages

Cellerator source additions are real work with their own authority, regression obligations and integration history. GlassHelix is the orchestrating consumer, not an alternate owner of that source. Both native plans are independently additive. A combined ZIP is a delivery convenience, not a merged authority.

No native task dependency points at a foreign project's task id. Instead, a producer publishes a capability receipt; a local consumer import task verifies it and reaches a local checkpoint/accepted task state. Dependent tasks reference that local task. The combined offline validator additionally checks the true cross-project edges to catch cycles invisible to each local plan.

## Receipt sequence

CE M10 publishes developmental contracts; GH X01 imports them. This establishes interface agreement, not runtime execution. CE M20 publishes the linked core; GH X02 imports it. CE M30 publishes n-ary/support capability; GH X03 imports it. CE M40 publishes differential execution; GH X04 imports it. CE M50 publishes the qualified consumer-ready bundle; GH X05 imports it.

GH M60 then executes and publishes real consumer acceptance against that Cellerator bundle. CE X01 imports GH M60; CE M90 closes. GH X06 imports CE M90; GH M90 closes the final pair.

**Do not make GH M60 wait for CE M90. Do not make CE M50 wait for GH M60.** Either change creates an avoidable mutual-final-acceptance deadlock.

## What a producer receipt proves

A qualified receipt identifies the edge, capability, producer project UUID, run, task, source commit, relevant source dependency hashes, interface versions and file hashes. Runtime capabilities include references to actual passing test receipts with exact source/binary/build/tool/device information. Developmental-contract receipt is explicitly weaker and need not pretend that a kernel ran.

The receipt identifies all relevant source dependencies, not just one header chosen because it stayed unchanged. A consumed descendant is allowed only when those dependencies and interface hashes still match, or when qualification is repeated and a new receipt published.

The consumer also captures a fresh authoritative observation of producer task completion and checkpoint state. The shipped authority wrapper contains normalized fields plus the raw tool response supporting them. A handwritten `done: true` is not authority. Checksums protect integrity and identity; they do not make a forged human claim true. The controller and independent verifier remain responsible for evidence quality.

## Artifact and source commit order

Freeze the executable source commit, build and run qualification at that commit, then publish external capability/test receipts. The producer task can subsequently be marked complete with the external receipt reference and checkpoint through the native workflow. This avoids requiring a source file to contain its own commit hash.

Do not require an impossible self-referential receipt commit. Runtime receipts belong in an external evidence directory, optionally archived later with an explicit source-under-test identity. The consumer reads the fresh authority after completion.

## Drift and renewal

A changed public contract, affected numerical implementation, build configuration, support semantics or arithmetic policy invalidates the relevant receipt. Stop dependent use, publish a revised contract, update consumers and rerun affected qualification. A changed unrelated README does not by itself require rerunning every numerical test, but compatibility must still be recorded deliberately.

## Two imports are not atomic

Manually import Cellerator first, verify it, then take a fresh GlassHelix preview and import GlassHelix. If the second import fails, preserve the first. Reconcile actual authority before any retry. Do not roll back or delete the successful first import automatically. Plan ingestion may create active run metadata, but no script in this package claims or dispatches workers.
