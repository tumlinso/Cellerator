#include <CellShard/CellShard.hh>

int main() {
    ::cellshard::sharded<::cellshard::compressed> matrix;
    init(&matrix);
    return 0;
}
