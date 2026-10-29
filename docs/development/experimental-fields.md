# Experimental prepared execution fields

`cellerator.experimental.ExecutionField` composes existing
`cellerator.torch.Product2Module` instances in an explicit order. It is a CPU
witness for prepared-operation composition, not a compiler or a new numerical
backend. Each stage continues to own its native prepared Product2 handle,
coefficient parameter and Torch autograd integration. Torch modules and stock
optimizers retain their normal roles.

```python
import torch
from cellerator.experimental import ExecutionField
from cellerator.torch import Product2Module

first = Product2Module(
    torch.tensor([0.5, 1.25], dtype=torch.float32),
    torch.tensor([0, 1]), torch.tensor([1, 2]),
)
second = Product2Module(
    torch.tensor([2.0], dtype=torch.float32),
    torch.tensor([0]), torch.tensor([1]),
)
field = ExecutionField((("pairwise", first), ("combine", second)), input_extent=3)
bound = field.bind(torch.tensor([1.0, 2.0, 3.0], dtype=torch.float32))
result = field(bound)
```

## Contract

- **Logical structure:** immutable ordered stage names, Product2 input indices,
  packet extents, and a structure epoch. Stage order is execution order. Each
  stage emits one value per packet; the next stage must accept that extent.
  Changing indices is rejected at field use until `rebind_structure()` explicitly
  accepts a new structure epoch and invokes the existing module preparation
  boundary. Bindings from earlier epochs are invalid.
- **Physical layout and bindings:** the current adapter accepts rank-one,
  strided CPU FP32 Torch tensors. Logical indices and output values use identity
  mapping. Existing Product2 Torch code materializes its NumPy inputs as needed;
  this layer does not claim a zero-copy or device-resident layout. `bind(tensor)`
  creates a checked input binding. Rebinding another tensor of the same extent
  advances the binding generation while reusing the prepared handles.
- **Mutable values and generations:** each stage's coefficients remain its
  ordinary Torch `Parameter`. The field's value generation advances when a
  parameter version change is observed, including after a stock optimizer step.
  Updating values does not rebuild structure or prepared topology. Input values
  are supplied by the binding and are not copied into an independent state
  owner.
- **Effects and lifetime:** field calls execute native Product2 forward and its
  existing custom backward in stage order. The modules keep prepared handles
  alive. This host CPU path introduces no CUDA stream work, implicit GPU
  synchronization, or field-exit publication. It does not provide workset
  discovery, adaptive execution, live compilation, code replacement, or a
  general graph/IR contract.

The first witness is intentionally narrow: native Product2 stages composed by
Torch autograd, with an independent Torch expression as the correctness oracle.
Other existing native operations can be admitted only after they expose a
compatible prepared-operation contract; no missing operation is synthesized
here.
