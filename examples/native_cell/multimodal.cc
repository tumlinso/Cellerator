#include <Cellerator/memory/view.hh>

#include <stdint.h>
#include <stdio.h>

int main() {
    using namespace cellerator;
    using Token = uint32_t;
    using FeatureId = const char*;
    constexpr char cell_id = 'M';
    constexpr int evidence_code = 8;

    const Token event_storage[] = {4, 7, 7, 9};
    const memory::const_array_view<Token> events{
        event_storage, 4, {memory::domain::host}};

    // Two rows, three features, and one padding value per row.
    double expression_storage[] = {
        1.0, 0.5, 2.0, -1.0,
        3.0, 4.5, 6.0, -2.0};
    const memory::matrix_view<double> matrix{
        expression_storage, 2, 3, 4, {memory::domain::host}};
    const FeatureId feature_ids[] = {"gene-A", "gene-B", "gene-C"};
    const memory::array_view<double> row{
        matrix.data + matrix.row_stride, matrix.columns, matrix.where};

    printf("cell-%c events=%zu first=%u expression[%s]=%g evidence=%d\n",
           cell_id, events.count, unsigned(events.data[0]),
           feature_ids[1], row.data[1], evidence_code);

    row.data[1] = 5.25;
    row.data[2] = 6.5;
    if (expression_storage[5] != 5.25 || expression_storage[6] != 6.5 ||
        expression_storage[1] != 0.5 || expression_storage[7] != -2.0) {
        fputs("expression edit did not reach the selected backing matrix row\n", stderr);
        return 1;
    }
    printf("updated %s=%g %s=%g\n",
           feature_ids[1], row.data[1], feature_ids[2], row.data[2]);
}
