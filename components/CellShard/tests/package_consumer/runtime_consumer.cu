#include <CellShard/CellShard.hh>

int main() {
    ::cellshard::sharded<::cellerator::tensor::base::compressed> matrix;
    init(&matrix);
    return 0;
}
