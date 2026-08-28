import cellerator.module_probe;

int main() {
    static_assert(cellerator::module_probe::contract_version == 1u);
    return 0;
}
