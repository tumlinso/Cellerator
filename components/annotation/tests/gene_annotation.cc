#include <filesystem>
#include <fstream>
#include <string_view>
#include <unistd.h>

#ifdef CELLERATOR_ANNOTATION_TEST_MODULE
import cellerator.annotation;
#else
#include <Cellerator/biology/gene_annotation.hh>
#endif

namespace bio = cellerator::biology;

static bool equal(bio::AnnotationText a, std::string_view b) {
    return a.size == b.size() && (!a.size || std::string_view(a.data, a.size) == b);
}

int main() {
    const auto path = std::filesystem::temp_directory_path() /
        ("cellerator-annotation-" + std::to_string(getpid()));
    {
        std::ofstream file(path);
        file << "# gtf\nchr1\tsource\tgene\t1\t4\t.\t+\t.\tgene_alias \"other\"; "
                "gene_name \"G;L\"; gene_id \"GENE.3\"; gene_synonym \"ALT|ALT2\"; tag \"a\"; tag \"b\";\n";
    }
    bio::GeneAnnotation catalog;
    auto error = bio::read_gene_annotation(path.c_str(), bio::GeneAnnotationFormat::gtf, catalog);
    if (error.message || catalog.count() != 1) return 1;
    auto row = catalog.records()[0];
    if (!equal(row.id, "GENE.3") || !equal(row.symbol, "G;L")
        || row.alias_count != 3 || !equal(row.aliases[2], "other")) return 2;
    {
        std::ofstream file(path);
        file << "# gff3\r\nchr1\tsource\tmRNA\t1\t4\t.\t+\t.\t"
                "Alias=one%2Ctwo,three;ID=GENE%3A1;Name=gene%20one\r\n"
                "##FASTA\r\n>chr1\r\nACGT\r\n";
    }
    error = bio::read_gene_annotation(path.c_str(), bio::GeneAnnotationFormat::gff3, catalog, "mRNA");
    if (error.message || catalog.count() != 1) return 3;
    row = catalog.records()[0];
    if (!equal(row.id, "GENE:1") || !equal(row.symbol, "gene one")
        || row.alias_count != 2 || !equal(row.aliases[0], "one,two")) return 4;
    const auto *prior = catalog.records();
    {
        std::ofstream file(path);
        file << "# header\nchr1\tsource\tgene\t1\t4\t.\t+\t.\tID=%GG\n";
    }
    error = bio::read_gene_annotation(path.c_str(), bio::GeneAnnotationFormat::gff3, catalog);
    std::filesystem::remove(path);
    if (!error.message || error.line != 2 || catalog.records() != prior) return 5;
    return 0;
}
