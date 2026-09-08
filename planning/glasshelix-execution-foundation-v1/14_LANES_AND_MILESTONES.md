# Lane and milestone guide

This project contains 14 first-class lanes, including the coordinator. Each queue below is ordered. The combined graph is in `machine/joint_topology.json`.

| Lane | Role | Tasks |
|---|---|---|
| CE-NF1-L-A | implementer | A01, A02, A03, A04 |
| CE-NF1-L-C | implementer | C01, C02, C03, C04, C05, C06 |
| CE-NF1-L-B | implementer | B01, B02, B03, B04, B05, B06 |
| CE-NF1-L-P | implementer | P01, P02, P03, P04, P05, P06, P07 |
| CE-NF1-L-V | implementer | V01, V02, V03, V04, V05, V06, V07 |
| CE-NF1-L-N | implementer | N01, N02, N03, N04, N05, N06 |
| CE-NF1-L-H | implementer | H01, H02, H03, H04, H05, H06, H07 |
| CE-NF1-L-S | implementer | S01, S02, S03, S04, S05, S06, S07 |
| CE-NF1-L-D | implementer | D01, D02, D03, D04, D05, D06, D07, D08 |
| CE-NF1-L-T | validator | T01, T02, T03, T04, T05, T06 |
| CE-NF1-L-O | specialist | O01, O02, O03, O04, O05, O06, O07 |
| CE-NF1-L-X | integrator | X01 |
| CE-NF1-L-M | integrator | M00, M10, M20, M30, M40, M50, M90 |

## Integration does not wait behind its own build hooks

The first baseline integration may seed a small optional build-fragment hook before feature fan-out, with no placeholder capabilities. Provider/build lanes then register real tests through their assigned fragments. If an existing build requires a central change not yet claimed, the controller records a narrow prerequisite integration/scope transfer; it must not wait until a milestone whose prerequisites need those tests. This is an explicitly authorized coordination adjustment, not an excuse to accept zero tests.

Only actual accepted prerequisite work releases the next milestone. Contract receipt can release design and interface work; linked/GPU consumer claims require the later capability receipts.
