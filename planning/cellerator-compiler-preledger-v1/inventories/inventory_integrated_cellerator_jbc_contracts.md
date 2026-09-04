# Integrated Cellerator JBC contract inventory

## Receipt identity

- Part One Todo: `CE-CCP1-A02-001`
- Observed Cellerator commit: `31e491ed29de0fcde70259cbeab8c5c7ad353485`
- Embedded CellShard gitlink at that commit: `b9749ad3e5146a04f847533d8c6f1a54146aed20`
- JBC planning authority: `planning/jbc-preledger-v1/proposed_todos.csv`
- Inventory rule: enumerate every regular file under the four declared Cellerator contract roots and every current file under `tests/jbc`, hash file bytes with SHA-256, and map each entry to its implementing `CE-JBC-*` Todo.
- Counts: 37 integrated contract headers and 94 current JBC test artifacts.

This is an evidence receipt, not a new compiler abstraction. It records the integrated Cellerator-side JBC surface before rehoming. The empty `components/CellShard` directory in this isolated worktree represents an uninitialized gitlink; CellShard branch/worktree enumeration belongs to `CE-CCP1-A02-002`.

## Coverage summary

| Capability | Implementing JBC Todos | Integrated contract roots |
|---|---|---|
| Persistent identity and logical coverage | `CE-JBC-I01` through `CE-JBC-I03` | `include/Cellerator/execution/joint_compiler` |
| Decomposition and partial algebra | `CE-JBC-I06`, `CE-JBC-I07`, `CE-JBC-D01` through `CE-JBC-D18` | `include/Cellerator/compute/decomposition` |
| Atom requirements, affordances, fragments, and binding | `CE-JBC-I04`, `CE-JBC-I05`, `CE-JBC-I08` through `CE-JBC-I10`, `CE-JBC-F01` through `CE-JBC-F14`, `CE-JBC-M01` through `CE-JBC-M09` | execution contracts plus `tests/jbc/fragment` and `tests/jbc/multi_extent` |
| Lowering resumption | `CE-JBC-I11`, `CE-JBC-R01` through `CE-JBC-R10` | execution contract plus `tests/jbc/resumption` |
| External costs and compiler exchange | `CE-JBC-C01` through `CE-JBC-C06` | `include/Cellerator/planner/external_cost` |
| Execution export and verification | `CE-JBC-I12`, `CE-JBC-V01` through `CE-JBC-V05` | `include/Cellerator/profiling/joint_compiler` and `tests/jbc/verification` |
| Atom planes and cross-operation views | `CE-JBC-P01` through `CE-JBC-P10`, `CE-JBC-X01` through `CE-JBC-X08` | current tests under `tests/jbc/atom_plane` and `tests/jbc/cross_operation` |

No intentional behavior difference is introduced. The inventory preserves ordinary C++ pointers and native control flow where those contracts use them. It makes no Part Two JIT or deep CellShard-runtime claim. `CE-JBC-M10` and `CE-JBC-V06` are integration/benchmark outcomes without distinct files in this bounded contract/test inventory.

## Integrated contract manifest

| Path | SHA-256 | Implementing JBC Todo | Current validation |
|---|---|---|---|
| `include/Cellerator/compute/decomposition/decomposition_v1.hh` | `ee5c569d8a785e14a0aeaea3f3de1c0baf11425d9c0bbe839555fbdca9ae3812` | `CE-JBC-I07` | `tests/jbc/interfaces/decomposition_v1_test.cc` |
| `include/Cellerator/compute/decomposition/dense_width_v1.hh` | `bed08fc89eb96f281bf6024e1dc24ca0cad540bba29e8b16ec521c62a9db2d8f` | `CE-JBC-D04` | `tests/jbc/decomposition/test_dense_width_v1.cc` |
| `include/Cellerator/compute/decomposition/destination_disjoint_v1.hh` | `ccfacfcfbb18c993c792289327eddef94522034c9e22dcc8a7ec01b9b1786f17` | `CE-JBC-D02` | `tests/jbc/decomposition/test_destination_disjoint_v1.cc` |
| `include/Cellerator/compute/decomposition/edge_component_v1.hh` | `cc0a242181360fd3d8607f08a578c92ccfb579e733c7a64a75cb4d98d5591ccb` | `CE-JBC-D05` | `tests/jbc/decomposition/test_edge_component_v1.cc` |
| `include/Cellerator/compute/decomposition/gate_input_v1.hh` | `091649e477a228945894cdd0ca38b2474d2bf6dad3f1c7f18e0666cc3c652e69` | `CE-JBC-D15` | `tests/jbc/decomposition/test_gate_input_v1.cc` |
| `include/Cellerator/compute/decomposition/log_sum_exp_state_v1.hh` | `5c69f564f960ed02d1651d4ed991c4b8f10e60619b175e9583d71d7c9ba8095d` | `CE-JBC-D13` | `tests/jbc/decomposition/test_log_sum_exp_state_v1.cc` |
| `include/Cellerator/compute/decomposition/moments_state_v1.hh` | `f0c3dced2ed973a0e86ea24ec7afbf3c59e78e649b9a7b175babd53aa8619e85` | `CE-JBC-D14` | `tests/jbc/decomposition/test_moments_state_v1.cc` |
| `include/Cellerator/compute/decomposition/partial_result_algebra_v1.hh` | `4a61f975519743519f00d25129ed963c216946f64b6eaae7e69d0f7594f2037d` | `CE-JBC-I06` | `tests/jbc/interfaces/partial_result_algebra_v1_test.cc` |
| `include/Cellerator/compute/decomposition/provider_registry_v1.hh` | `bd45cd8d37121c10f95ba4fdafbf7263a6695ccb59ea16b24b21f9f94404004f` | `CE-JBC-D18` | `tests/jbc/decomposition/test_provider_registry_v1.cc` |
| `include/Cellerator/compute/decomposition/relation_bundle_v1.hh` | `ba340874dcc133eaced2ea8186db148a457eb43d773f0682ac0aaac7738c53f5` | `CE-JBC-D06` | `tests/jbc/decomposition/test_relation_bundle_v1.cc` |
| `include/Cellerator/compute/decomposition/segment_disjoint_v1.hh` | `07fc8d6295793bb6b6d3e415208e3146abf1483b9ccde0db788c206c83ac14bc` | `CE-JBC-D11` | `tests/jbc/decomposition/test_segment_disjoint_v1.cc` |
| `include/Cellerator/compute/decomposition/source_k_v1.hh` | `78b732d7420371ece0b35c74ab425ccb7f82512d96e8e94aecc02c9a34c36bc3` | `CE-JBC-D03` | `tests/jbc/decomposition/test_source_k_v1.cc` |
| `include/Cellerator/compute/decomposition/sparse_update_conflict_v1.hh` | `ed1e7636a76609b189bfb4ea26ea21f10ece7f9a8d48698889348de5d3e47751` | `CE-JBC-D16` | `tests/jbc/decomposition/test_sparse_update_conflict_v1.cc` |
| `include/Cellerator/compute/decomposition/split_segment_reduce_v1.hh` | `b8fe26a4187c7c7656847eb0666b1d00fab1d5fbb25737678afac121ee052b5d` | `CE-JBC-D12` | `tests/jbc/decomposition/test_split_segment_reduce_v1.cc` |
| `include/Cellerator/compute/decomposition/support_contraction_v1.hh` | `eced852d65292d88615252f3aa56022d9f174348b37a40938ac6630aa07304b3` | `CE-JBC-D08` | `tests/jbc/decomposition/test_support_contraction_v1.cc` |
| `include/Cellerator/compute/decomposition/support_edge_rectangle_v1.hh` | `4575c6f051b3e51cf8ddeb9d913617081868f7725b1c55c63cdb1a814f741d0e` | `CE-JBC-D09` | `tests/jbc/decomposition/test_support_edge_rectangle_v1.cc` |
| `include/Cellerator/compute/decomposition/support_embedding_v1.hh` | `8466aa9cd0e74e0efb9a8a1a477b266b4a81ee9efa5347959a23a23a77f34fac` | `CE-JBC-D10` | `tests/jbc/decomposition/test_support_embedding_v1.cc` |
| `include/Cellerator/compute/decomposition/training_gradient_v1.hh` | `f3cf9ab2554bade2333a0541e35b577a37caac8619635985f2b268356fafaa76` | `CE-JBC-D17` | `tests/jbc/decomposition/test_training_gradient_v1.cc` |
| `include/Cellerator/compute/decomposition/transpose_source_partials_v1.hh` | `950f9a3f3c65c37ce180df33908576f15e60b6e6d9c77bc9f28ed5aa3f39ada6` | `CE-JBC-D07` | `tests/jbc/decomposition/test_transpose_source_partials_v1.cc` |
| `include/Cellerator/compute/decomposition/vocabulary_v1.hh` | `61ce15d33a59cacc770f4d6a352bef25f1cc90c5048dbaee03040af67e9df252` | `CE-JBC-D01` | `tests/jbc/decomposition/test_vocabulary_v1.cc` |
| `include/Cellerator/execution/joint_compiler/atom_affordance_v1.hh` | `a694beac285acb9a63ae0c34e05104897a72eb45903f44327220a55776cac7ac` | `CE-JBC-I05` | `tests/jbc/interfaces/atom_affordance_v1_test.cc` |
| `include/Cellerator/execution/joint_compiler/atom_fragment_request_v1.hh` | `6158744a3d308798cfbf102876a651fa00f902e2b0ba5d9fb69c8c2d1b13a740` | `CE-JBC-I08` | `tests/jbc/interfaces/atom_fragment_request_v1_test.cc` |
| `include/Cellerator/execution/joint_compiler/atom_fragment_result_v1.hh` | `cbd14728d58f003b460b68c3f80ac939632935b418892e43182ea93c9d42ce55` | `CE-JBC-I09` | `tests/jbc/interfaces/atom_fragment_result_v1_test.cc` |
| `include/Cellerator/execution/joint_compiler/atom_requirement_v1.hh` | `525c3fe39559152863b38ef83c112250e100894092c4c21b553b8dd622756cb9` | `CE-JBC-I04` | `tests/jbc/interfaces/atom_requirement_v1_test.cc` |
| `include/Cellerator/execution/joint_compiler/coverage_roles_v1.hh` | `a458ece8259a7bc921aadaeb1addd65e1b782c42d1262e9e553338f787415c3a` | `CE-JBC-I03` | `tests/jbc/interfaces/coverage_roles_v1_test.cc` |
| `include/Cellerator/execution/joint_compiler/external_binding_v1.hh` | `7955045d65f3d2a4c1fc4f016704ff57fa9785a110f6eb2626391d2924e7bccf` | `CE-JBC-I10` | `tests/jbc/interfaces/external_binding_v1_test.cc` |
| `include/Cellerator/execution/joint_compiler/logical_coverage_v1.hh` | `0e70a73be17360ca3bbc8af18bd049cfdb384ae421803014685029acc030833b` | `CE-JBC-I02` | `tests/jbc/interfaces/logical_coverage_v1_test.cc` |
| `include/Cellerator/execution/joint_compiler/lowering_resumption_v1.hh` | `d7b2107ff528c9eb179e7574428dd01526694026d6c58f6a8b8fd7537cd5f678` | `CE-JBC-I11` | `tests/jbc/interfaces/lowering_resumption_v1_test.cc` |
| `include/Cellerator/execution/joint_compiler/persistent_identity_v1.hh` | `e36c6556be687d39961380f6ac03b4b1a340f0bb15d7b8f2b197454488f0bbbe` | `CE-JBC-I01` | `tests/jbc/interfaces/persistent_identity_v1_test.cc` |
| `include/Cellerator/planner/external_cost/compiler_exchange_v1.hh` | `667edb4b452bfa427ac93329942a2eea07cdac05fbf08e76658ff9492157e3ce` | `CE-JBC-C05` | `tests/jbc/external_cost/compiler_exchange_v1_test.cc` |
| `include/Cellerator/planner/external_cost/complete_cost_v1.hh` | `62c917d88e715773bfd5e5e775fa6423f6e5dd53ba101638af21d316442428cd` | `CE-JBC-C02` | `tests/jbc/external_cost/complete_cost_v1_test.cc` |
| `include/Cellerator/planner/external_cost/frontier_v1.hh` | `88e5a33065284e70aa850bd3dbebd0cf76d9453726146f268940f7180ea222b2` | `CE-JBC-C04` | `tests/jbc/external_cost/frontier_v1_test.cc` |
| `include/Cellerator/planner/external_cost/geometry_objective_v1.hh` | `62dd815816fb9c4005e1b92a29de6310a6f015cff2dda2918a3a5620abba31be` | `CE-JBC-C03` | `tests/jbc/external_cost/geometry_objective_v1_test.cc` |
| `include/Cellerator/planner/external_cost/pricing_oracle_v1.hh` | `e41b68cca12b1848edaf76b998a4ce3344868c2e4c8be73b924d12b352cfbf6b` | `CE-JBC-C06` | `tests/jbc/external_cost/pricing_oracle_v1_test.cc` |
| `include/Cellerator/planner/external_cost/vector_v1.hh` | `ead57132a8129e635136f103e77d4638461db47c2c1f8c0db18f407daf2188e9` | `CE-JBC-C01` | `tests/jbc/external_cost/vector_v1_test.cc` |
| `include/Cellerator/profiling/joint_compiler/execution_export_v2.hh` | `83726ba14944c916d79d1f50ba34b8cef0900cbf63fe19501a97e8cd555e20fd` | `CE-JBC-I12` | `tests/jbc/interfaces/execution_export_v2_test.cc` |
| `include/Cellerator/profiling/joint_compiler/manifest_v1.hh` | `f659d5ae84d80dd646f3a571d423df5e19d87c84af0258081d350e63995d98e0` | `CE-JBC-V03` | `tests/jbc/verification/profiler_manifest_v1_test.cc` |

## Current JBC test manifest

| Path | SHA-256 | Implementing JBC Todo |
|---|---|---|
| `tests/jbc/atom_plane/active_support_overlay_v1_test.cc` | `2f149acd1bdf353c6ea7cb9a4252946bcf6b36a56261d657a64b86ae7cc4c4d6` | `CE-JBC-P04` |
| `tests/jbc/atom_plane/dense_result_atom_v1_test.cc` | `7471c2c3fe7897a8ef62d7162834d7a7681361527567313089df5c250f678b36` | `CE-JBC-P08` |
| `tests/jbc/atom_plane/external_plane_mapping_v1_test.cc` | `9e9dcff176817452458534c94963ae1b64d91a38a952c0355e812d604fa2b8e3` | `CE-JBC-P01` |
| `tests/jbc/atom_plane/generation_publication_binding_v1_test.cc` | `a1dbd742d2e45b959a141c537c5dd8fda4e95172ef99f1bd8885af39a0c31213` | `CE-JBC-P07` |
| `tests/jbc/atom_plane/gradient_plane_v1_test.cc` | `e369576072977a05cf8987b737c71f55888c5620e9c147c1f6a1a7fc39846768` | `CE-JBC-P06` |
| `tests/jbc/atom_plane/mutable_state_plane_v1_test.cc` | `ad87992988eef40fb846877b0d101e540d41ca7bd7428d27664382a189ce3171` | `CE-JBC-P05` |
| `tests/jbc/atom_plane/partial_result_atom_v1_test.cc` | `737f3780e0eba973b6238a5ce8f4b3b24c316ca94bc72bd16600ed212a3cec2e` | `CE-JBC-P09` |
| `tests/jbc/atom_plane/ready_lease_binding_v1_test.cc` | `a97fa34c6f5cbad162ebe3d84be97feffd2d49d5fefe5c46fbd9abee4c90677e` | `CE-JBC-P10` |
| `tests/jbc/atom_plane/relation_value_plane_v1_test.cc` | `d4544509c04e090904b9081bdfb72791388b3e1ad64cac643ffbb739fcdd42c0` | `CE-JBC-P03` |
| `tests/jbc/atom_plane/structural_plane_binding_v1_test.cc` | `d4181ed51d24bb67763734eeda8e3660d2cbcdd0dbee0d0a7a1cf9da1d7c7d0b` | `CE-JBC-P02` |
| `tests/jbc/cross_operation/cross_operation_pareto_v1_test.cc` | `4caade3494bb9c19bc72af614fe765f3af697fa3ed9d36d842bc2b3321f6d08b` | `CE-JBC-X08` |
| `tests/jbc/cross_operation/forward_relation_apply_v1_test.cc` | `171e137cf61dfa8c1e63179635893e46026ad14116f03c63c5d043e491118018` | `CE-JBC-X02` |
| `tests/jbc/cross_operation/segment_and_gate_v1_test.cc` | `66fcf4cd2bb00dd7404ab1ef6eb953ec5943164b33f924ae24a4a45b70c83878` | `CE-JBC-X05` |
| `tests/jbc/cross_operation/support_contraction_v1_test.cc` | `746f2202921aa30fd5be26e7a4946f6f498f720981074f30794b0731c412a0f2` | `CE-JBC-X04` |
| `tests/jbc/cross_operation/support_family_identity_v1_test.cc` | `35c9531be13fd532e821c79b0e66d96153b50ebe5ce472e9b3076fe5514a59cf` | `CE-JBC-X01` |
| `tests/jbc/cross_operation/transpose_relation_apply_v1_test.cc` | `dfbbd8b56f18cbe55e96fa937d81f9811a8f6ddb36ab79a35efd779faabc945a` | `CE-JBC-X03` |
| `tests/jbc/cross_operation/value_gradient_identity_v1_test.cc` | `c46d24bb668c1ff1832b4c1ed8417dd6197ac06c66bf563ce7a649f8d4fb5b36` | `CE-JBC-X06` |
| `tests/jbc/cross_operation/view_family_comparison_v1_test.cc` | `c367ad8a981a891d4a63f6765546ec8b2f575a7ae432d3df339b83802651da30` | `CE-JBC-X07` |
| `tests/jbc/decomposition/test_dense_width_v1.cc` | `26f7b35634c23ca69009773adab8028db9cd5b241dfde4c26984aea7c9185d35` | `CE-JBC-D04` |
| `tests/jbc/decomposition/test_destination_disjoint_v1.cc` | `213d6e1bcc8fcf99d7fba62ad50c3f8d26a74837cc3bb4eb37f5aa63583f3b59` | `CE-JBC-D02` |
| `tests/jbc/decomposition/test_edge_component_v1.cc` | `492c1c36d4b4d01d8e4540e86772202033e48de29ebf31b2a0bf7dc1c397e7d2` | `CE-JBC-D05` |
| `tests/jbc/decomposition/test_gate_input_v1.cc` | `5c99517d755868ce7cb9ab99532065fd3f2d82bd93f6dceb10e04f9407553540` | `CE-JBC-D15` |
| `tests/jbc/decomposition/test_log_sum_exp_state_v1.cc` | `53211ee3aadf478ea9e66c9abd4cf35886fc2e2bf80529ae72076658d3eaa76a` | `CE-JBC-D13` |
| `tests/jbc/decomposition/test_moments_state_v1.cc` | `16848ae3a3f0bbf19dbb2da4197b48161801d18866a98b8dcd5531e281c561cd` | `CE-JBC-D14` |
| `tests/jbc/decomposition/test_provider_registry_v1.cc` | `6377908b6d9d4ba1bf4bf838060c439d6a9b3b01a7743b2ed897d9d1210ef96d` | `CE-JBC-D18` |
| `tests/jbc/decomposition/test_relation_bundle_v1.cc` | `211bf20ce50a8bad6f8c37231e7a25ea3a242068923346be06cf78f1a3e78c77` | `CE-JBC-D06` |
| `tests/jbc/decomposition/test_segment_disjoint_v1.cc` | `d50d5b3dd050c7b08101198949d1a250c7d75d71cb0666354ca66ad9da1ab473` | `CE-JBC-D11` |
| `tests/jbc/decomposition/test_source_k_v1.cc` | `14b0fbb868228f51442c5dbd99293bc48535b56eacc8fe61f2ff1ab8eb45135a` | `CE-JBC-D03` |
| `tests/jbc/decomposition/test_sparse_update_conflict_v1.cc` | `504ed4a1776abdf2d4643e3f60c9cd2b6a60916028a376cba13044951ce14070` | `CE-JBC-D16` |
| `tests/jbc/decomposition/test_split_segment_reduce_v1.cc` | `a084835bf40d5ec8495e96b4b65f615bd7c2a0d0c3446dcdb11105e007488bd4` | `CE-JBC-D12` |
| `tests/jbc/decomposition/test_support_contraction_v1.cc` | `8c30f7421ca5122838c58e8c3785111f6740a71fc64f5fb19495468d9fb092ef` | `CE-JBC-D08` |
| `tests/jbc/decomposition/test_support_edge_rectangle_v1.cc` | `04c8372f315c7a725b12fd0aefd01546881a867013c0a6878f12afd69845ce21` | `CE-JBC-D09` |
| `tests/jbc/decomposition/test_support_embedding_v1.cc` | `9a6098ba34d06ab60a67e8dc0bc94b92f114ab67536cc7d0beb88e7aa9195197` | `CE-JBC-D10` |
| `tests/jbc/decomposition/test_training_gradient_v1.cc` | `f078a58844b6feafff9f4787b2d6e9aeb5fe3bf7c83001ca032198ff61474dee` | `CE-JBC-D17` |
| `tests/jbc/decomposition/test_transpose_source_partials_v1.cc` | `cb740e0014b8ed8c2c41236df47ca46a73d0190aaf3ef72ee0f36c4205a06b8c` | `CE-JBC-D07` |
| `tests/jbc/decomposition/test_vocabulary_v1.cc` | `72725eba3fa103d497f22c51454190b808b27c783098d59be89c873d05c03743` | `CE-JBC-D01` |
| `tests/jbc/external_cost/compiler_exchange_v1_test.cc` | `28baa49ba7f7900fbf5a2596cfd86d34c2897f069384e75d209adc97510261ad` | `CE-JBC-C05` |
| `tests/jbc/external_cost/complete_cost_v1_test.cc` | `2f725b3385d017fff763fd7fc3e2a6d8ba60b0e858d5cdfa1b63b7694c8b6a40` | `CE-JBC-C02` |
| `tests/jbc/external_cost/frontier_v1_test.cc` | `d18f880038442048523d4c3b443dd48a5dd1274d06e27e22eb432642e78db455` | `CE-JBC-C04` |
| `tests/jbc/external_cost/geometry_objective_v1_test.cc` | `f54bc1494d43685f70f790464676e5d42948d94df3a1269123d4e49d39831b8c` | `CE-JBC-C03` |
| `tests/jbc/external_cost/pricing_oracle_v1_test.cc` | `15f616bbc3e0e1df700b7603ce5ce6ed35023c7afe2fc7ff0b4349fc3301dd37` | `CE-JBC-C06` |
| `tests/jbc/external_cost/vector_v1_test.cc` | `12835b7e6f1bb071a75141b2f55a9afebec4ea32f077af81567c58930417ec22` | `CE-JBC-C01` |
| `tests/jbc/fragment/atom_bound_candidate_v1_test.cc` | `b3f3b14a33596794beb7347fe8674ccff24d7b8ee5196f6667546cecaaa645a5` | `CE-JBC-F07` |
| `tests/jbc/fragment/canonical_fallback_v1_test.cc` | `2375da171c889bf4966a173ea5bb99ba6e33e57607d40dffa93b1a4e738b70c7` | `CE-JBC-F12` |
| `tests/jbc/fragment/canonical_relation_apply_smoke.cc` | `bd9ea97412ebe62aa6b257e3c7bb89c864f694ef6c8bf2e7337186685fabfc9c` | `CE-JBC-F14` |
| `tests/jbc/fragment/compiler_registry_v1_test.cc` | `080b493eaaf30bd38b60b2f8eef78c0ca79fc9d55ec9bcc7b425d0ab2bbe5584` | `CE-JBC-F13` |
| `tests/jbc/fragment/external_decomposition_v1_test.cc` | `9ea5feb5d335f51d31c286bb6125c545d4ba013216ddebf12c267eb823c4ba2a` | `CE-JBC-F04` |
| `tests/jbc/fragment/external_persistent_order_v1_test.cc` | `544e937038c126313bd1b6c5325d359591d911ffeefc9bc5fa065209054d82c9` | `CE-JBC-F05` |
| `tests/jbc/fragment/external_plane_binding_v1_test.cc` | `5bd0d78d916380e7911208b04265f33811aa1ec7e859e386e7a1d6c10307c762` | `CE-JBC-F10` |
| `tests/jbc/fragment/local_candidate_requirements_v1_test.cc` | `0fd3c3cd4a981691db9210387bdc95ffddd6735c8532340771cb97a3d9825b1e` | `CE-JBC-F06` |
| `tests/jbc/fragment/local_index_builder_v1_test.cc` | `0673aa4bff450afd8591b0e88a341146e3fdb90dc48970a44e30b13383b098e4` | `CE-JBC-F03` |
| `tests/jbc/fragment/local_pareto_frontier_v1_test.cc` | `beeb304df93669cdd122b3f3f25c9c0f233dc18919661c6ef7ef1aab6377376e` | `CE-JBC-F08` |
| `tests/jbc/fragment/operation_adapter_v1_test.cc` | `6a806e75cdf9c60a9f7d55aa0d6080a5f952bac2ddc08978b32a03a022fbd99f` | `CE-JBC-F02` |
| `tests/jbc/fragment/output_affordance_v1_test.cc` | `3e450e78a1f519ce00d1160afb9287f9016d76812c1c3ee840c881e53d9e3123` | `CE-JBC-F11` |
| `tests/jbc/fragment/prepared_atom_fragment_v1_test.cc` | `6d73ef252a93868bce9c9bffb76471482ef90b3380509806cddc44daf505bdc8` | `CE-JBC-F09` |
| `tests/jbc/fragment/requirements_v1_test.cc` | `374c95a7953d3a6b1d8b68a23f1c4ec4808a5de2351535b4c9559a83e3c18f9e` | `CE-JBC-F01` |
| `tests/jbc/interfaces/atom_affordance_v1_test.cc` | `3f7f38eb691e014f5ab1b80e2c12bb9ef38fb4cedb47dcfebeb6aa68ed695363` | `CE-JBC-I05` |
| `tests/jbc/interfaces/atom_fragment_request_v1_test.cc` | `1b7a70573f207f793d51450d766fda0d34097d590204cf4f975e8e6031ddfc91` | `CE-JBC-I08` |
| `tests/jbc/interfaces/atom_fragment_result_v1_test.cc` | `8be0d6bfdb7f2f6ed220052853b7ff3ec75c18f7c22b7aecb8644bb2e51a0a97` | `CE-JBC-I09` |
| `tests/jbc/interfaces/atom_requirement_v1_test.cc` | `ede67ec2dfa4582dfefd392dcbc9694a32cdf2d004b94522526b6ca6e4566bb7` | `CE-JBC-I04` |
| `tests/jbc/interfaces/coverage_roles_v1_test.cc` | `1c3cf8dc1472e46d844ebcfb2af70e49b559d20608908b2d4ca60b332f9f64e9` | `CE-JBC-I03` |
| `tests/jbc/interfaces/decomposition_v1_test.cc` | `859f43b00e9528b3077f178bcc9a90e1c323273c2ca3df676647b3cbd6c64ace` | `CE-JBC-I07` |
| `tests/jbc/interfaces/execution_export_v2_test.cc` | `daa757174a1849de406acc88bcaffbe505b44a58e297bd79304d6aada5ae5c10` | `CE-JBC-I12` |
| `tests/jbc/interfaces/external_binding_v1_test.cc` | `dd6c4afcb2da25f74ab728d45b6bbbaca7bbe7ecae349bff43ee1df479685298` | `CE-JBC-I10` |
| `tests/jbc/interfaces/logical_coverage_v1_test.cc` | `d8a648b5ce148bf8a6b104063d8c77d5386868d882345da92a32bc923adfe9a2` | `CE-JBC-I02` |
| `tests/jbc/interfaces/lowering_resumption_v1_test.cc` | `02db03ece8bc2425b2d1fd8a3a6fbe462c551abe74ca4ce26b67f751c8d376c1` | `CE-JBC-I11` |
| `tests/jbc/interfaces/partial_result_algebra_v1_test.cc` | `ad91997ec20fd2262c7e2d645ea81766a1cf98bcdd2479cf357f0f23a135624d` | `CE-JBC-I06` |
| `tests/jbc/interfaces/persistent_identity_v1_test.cc` | `08aa4eecb7f928ad49295d6db37ef381faba5278c7843baf70101bfeec8cdabd` | `CE-JBC-I01` |
| `tests/jbc/multi_extent/chunk_native_acquisition_v1_test.cc` | `6eaf3e174186b7dbe137eb3cc128d0340fe20eaf574bd0002c4ddbad6358e653` | `CE-JBC-M09` |
| `tests/jbc/multi_extent/contiguous_assembly_v1_test.cc` | `f1b5d3d082dd3a180aa43db8fbbdf6d90e1fe28e92749d5d8c6640e370012c91` | `CE-JBC-M04` |
| `tests/jbc/multi_extent/direct_candidate_contract_v1_test.cc` | `80fb093c160a3da3c5096bee207754cc530a944d991f43657bb07e1925cda4b0` | `CE-JBC-M06` |
| `tests/jbc/multi_extent/extent_requirements_v1_test.cc` | `70ad5b1730ca7eb35f91784cdbe1796851579828eb6850a6b94bbd9eff508d53` | `CE-JBC-M03` |
| `tests/jbc/multi_extent/multi_atom_port_binding_v1_test.cc` | `e7b8dcc2b4225eb2ed9125b64edab3444890d0e3523eb71ecbac098c0b4ffc47` | `CE-JBC-M01` |
| `tests/jbc/multi_extent/permutation_operations_v1_test.cc` | `ac09221980f15ebb37845375b3b75cb518bccf70634ffbe1bdc355b1d471e4c7` | `CE-JBC-M05` |
| `tests/jbc/multi_extent/physical_binding_v1_test.cc` | `a3b6ce1c06e044715bb5a4527cbe71806b96ac2f2404ae2ddfd0afdb82a94ba0` | `CE-JBC-M02` |
| `tests/jbc/multi_extent/relation_apply_candidate_v1_test.cc` | `5ef0e5eefb80100e40df7722858b2f513cb0dfab0e380acf8f7d19142cbb82c6` | `CE-JBC-M07` |
| `tests/jbc/multi_extent/structural_overlay_v1_test.cc` | `44b390151b46a7b82f977cd8d01eba1f22d5b4ab46a05e44947d2fd71b25e3d7` | `CE-JBC-M08` |
| `tests/jbc/resumption/atom_evidence_v1_test.cc` | `7ac21bfe0453f39a96eeea97810e9beca417a347153e92912603aa3a4cb78b11` | `CE-JBC-R03` |
| `tests/jbc/resumption/canonical_source_v1_test.cc` | `dc068028ab643c0bcf06ac147d1d383c19f02c1eda69093457b91b4b24fe7f71` | `CE-JBC-R02` |
| `tests/jbc/resumption/executable_recipe_v1_test.cc` | `624e400d7aaceec7a34ca7465290154f912c372a252cafb6fe66d9007af57983` | `CE-JBC-R08` |
| `tests/jbc/resumption/instrumentation_v1_test.cc` | `b95780315a506c76e8f52fa515332d5ccf772febc3adcd6e6d1cba6f3f7c26d9` | `CE-JBC-R10` |
| `tests/jbc/resumption/local_realization_v1_test.cc` | `0d11d316f741fcef19bb4ed1b326b415dcd36b21027c50f49cebd6b49f1c9e62` | `CE-JBC-R09` |
| `tests/jbc/resumption/packed_operand_v1_test.cc` | `7d188a701bf07df9ab45605c5adb89b14d0fe269ad9c34e6b3a065b6cb808dec` | `CE-JBC-R07` |
| `tests/jbc/resumption/physical_projection_v1_test.cc` | `425b71ff6ac45a517d1d4147e70e7eca428760d6c017bc64dbe9471d6d7cd05d` | `CE-JBC-R06` |
| `tests/jbc/resumption/semantic_atom_v1_test.cc` | `fd522cbded93ee632ba6f3e9e159edb607c401baf4deb834ae4ca9c6a7f040c9` | `CE-JBC-R04` |
| `tests/jbc/resumption/status_taxonomy_v1_test.cc` | `25cd6a11533727da11ebee70b645846547a2f4b58fed1cd10caa807d84d641f7` | `CE-JBC-R01` |
| `tests/jbc/resumption/target_cover_v1_test.cc` | `a25a76006e40d66726edde0269235a8b43c3faa84dfbcd926da879e248b7ffa8` | `CE-JBC-R05` |
| `tests/jbc/verification/atom_fragment_verifier_v1.hh` | `1eb2d5adf74781e368917bd637fd283cae6e578417b64aa5891d7596678ad540` | `CE-JBC-V01` |
| `tests/jbc/verification/atom_fragment_verifier_v1_test.cc` | `4200b851e4b74b4d59ff4e5d169a04a279c8fb7e08e26a45b0a118e08a223141` | `CE-JBC-V01` |
| `tests/jbc/verification/cellshard_bridge_gate_v1_test.cc` | `b127066c65ff1cd43732aa7c63262e2c47a755f1d2ec60e4d3fe9cd0f88e551e` | `CE-JBC-V05` |
| `tests/jbc/verification/numerical_verifier_v1.hh` | `a9e044663cf98e2afeee196b216c588dcb0c30f5e2f0fb11d60ea2d857cd31f3` | `CE-JBC-V02` |
| `tests/jbc/verification/numerical_verifier_v1_test.cc` | `a057d4218dbb3dab01c5d502ab4d531add42f075603898112f86c875b1bfe4f9` | `CE-JBC-V02` |
| `tests/jbc/verification/profiler_manifest_v1_test.cc` | `7f33eda370f55429e0e6413743a35a3cce2d8ca3f9c0d1afb287153491f56aa2` | `CE-JBC-V03` |
| `tests/jbc/verification/standalone_abi_gate_v1_test.cc` | `27967a50d8b001516154dcffca4f238b651677bf23a5b14c1df1ea94d6f8f20e` | `CE-JBC-V04` |

## Completeness and preservation disposition

The source manifest is the exact sorted regular-file set at the recorded commit for:

- `include/Cellerator/compute/decomposition`
- `include/Cellerator/execution/joint_compiler`
- `include/Cellerator/planner/external_cost`
- `include/Cellerator/profiling/joint_compiler`

The test manifest is the exact sorted regular-file set below `tests/jbc`. Every manifest entry has one owning historical JBC Todo and every integrated contract header has at least one current test mapping. Later migration work must preserve, move, adapt, split, or explicitly replace these rows with proof; omission is not a valid rehoming disposition.
