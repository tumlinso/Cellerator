# Root initial interpretation, before independent result review

Recorded 2026-11-03T15:04:17.210358+00:00. This interpretation
uses the operator's finished receipt/cleanup and numerically validated measurement;
the verifier's independent measurement artifact acceptance is pending.

The matched circuit median is 47.7281 microseconds per full resident
forward/VJP versus 50.8444 for the unfactored control: 1.065293x,
or 6.129% less time. KDA's paired seven-block exploratory bootstrap
interval is [1.044346, 1.069254]x (95%, 10,000 resamples).
The primary comparison therefore supports a modest factoring benefit in this
fixed motif/mapping on the measured run. It does not meet or invoke the old
packet-promotion policy and does not change a production default.

Separate event medians report equal forward time (9.216 microseconds) and
backward 44.032 versus 41.984 microseconds. Those diagnostics are consistent
with the benefit arising mainly in backward, but are separately instrumented
and do not establish a kernel-level causal profile or sum into the primary metric.

The descriptive native request selected stable and measured 6.7223125 milliseconds,
about 140.85x the circuit time. This is NOT a factoring speedup: both private
arms share a specialized owner/reduction design that differs from the generic
native path. Reference compilation configuration and implementation differences
must be checked before proposing an explanation for that large gap. No new
GPU measurement or performance revision is authorized at this checkpoint.

The two private arms have identical reported buffer/scratch sizes: 3,276,800
bytes of tape+scratch each; the native stable reference reports 98,304 bytes.
The private reduction strategy thus spends about 33.33x more scratch, and its
performance cannot be separated from that trade by this experiment. Setup and
upload are outside the primary resident interval and are not amortized into a
claimed complete caller or optimizer-step speedup.

All three final output/VJP sets passed the frozen numeric check. Primary signal
is based on only seven serial paired blocks, one shape/topology/precision,
fixed warmed values, and pre/post observations rather than continuous foreign
process monitoring. Treat it as one useful experimental result. Stop, discuss
with an independent reviewer and the user, and choose no next experiment yet.

Raw measurement summary SHA256: ca98ac7f4ba66c325fabcd0f204fd83e8984a8690cab8a19f18d00edfeead9a5.
Analysis SHA256: 35d78864f0058d6911aba616013884b29c1bb7fd1a2ab093ff72f674fab03bcc.
