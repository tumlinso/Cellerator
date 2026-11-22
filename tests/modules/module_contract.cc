import cellerator.core;

struct Model {
    int operator()(int value) const { return value + 1; }
};

int main() {
    Model model;
    auto cell = cellerator::make_cell(model, 41);
    return cell.forward() == 42 ? 0 : 1;
}
