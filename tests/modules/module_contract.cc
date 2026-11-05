import cellerator.core;
import anndatapp.h5ad;

struct Model {
    int operator()(int value) const { return value + 1; }
};

int main() {
    Model model;
    auto cell = cellerator::make_cell(model, 41);
    anndatapp::BackedOptions options;
    return cell.forward() == 42 && options.scratch_bytes > 0 ? 0 : 1;
}
