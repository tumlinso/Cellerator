#ifndef TRIPLEXGENEMAP_BIOTYPES_H
#define TRIPLEXGENEMAP_BIOTYPES_H

#include <string>
#include <unordered_map>

typedef enum BioType {
    UNASSIGNED,
    IG_C_gene,
    IG_D_gene,
    IG_J_gene,
    IG_LV_gene,
    IG_V_gene,
    TR_C_gene,
    TR_J_gene,
    TR_V_gene,
    TR_D_gene,
    IG_pseudogene,
    IG_C_pseudogene,
    IG_J_pseudogene,
    IG_V_pseudogene,
    TR_V_pseudogene,
    TR_J_pseudogene,
    Mt_rRNA,
    Mt_tRNA,
    miRNA,
    misc_RNA,
    rRNA,
    scRNA,
    snRNA,
    snoRNA,
    ribozyme,
    sRNA,
    scaRNA,
    lncRNA,
    Mt_tRNA_pseudogene,
    tRNA_pseudogene,
    snoRNA_pseudogene,
    snRNA_pseudogene,
    scRNA_pseudogene,
    rRNA_pseudogene,
    misc_RNA_pseudogene,
    miRNA_pseudogene,
    TEC,
    nonsense_mediated_decay,
    non_stop_decay,
    retained_intron,
    protein_coding,
    protein_coding_LoF,
    protein_coding_CDS_not_defined,
    processed_transcript,
    non_coding,
    ambiguous_orf,
    sense_intronic,
    sense_overlapping,
    antisense,
    known_ncrna,
    pseudogene,
    processed_pseudogene,
    polymorphic_pseudogene,
    retrotransposed,
    transcribed_processed_pseudogene,
    transcribed_unprocessed_pseudogene,
    transcribed_unitary_pseudogene,
    translated_processed_pseudogene,
    translated_unprocessed_pseudogene,
    unitary_pseudogene,
    unprocessed_pseudogene,
    artifact,
    lincRNA,
    macro_lncRNA,
    threeprime_overlapping_ncRNA,
    disrupted_domain,
    vaultRNA,
    bidirectional_promoter_lncRNA
} BioType;

const char* getBioTypeName(BioType biotype) {
    switch (biotype) {
        case IG_C_gene: return "IG_C_gene";
        case IG_D_gene: return "IG_D_gene";
        case IG_J_gene: return "IG_J_gene";
        case IG_LV_gene: return "IG_LV_gene";
        case IG_V_gene: return "IG_V_gene";
        case TR_C_gene: return "TR_C_gene";
        case TR_J_gene: return "TR_J_gene";
        case TR_V_gene: return "TR_V_gene";
        case TR_D_gene: return "TR_D_gene";
        case IG_pseudogene: return "IG_pseudogene";
        case IG_C_pseudogene: return "IG_C_pseudogene";
        case IG_J_pseudogene: return "IG_J_pseudogene";
        case IG_V_pseudogene: return "IG_V_pseudogene";
        case TR_V_pseudogene: return "TR_V_pseudogene";
        case TR_J_pseudogene: return "TR_J_pseudogene";
        case Mt_rRNA: return "Mt_rRNA";
        case Mt_tRNA: return "Mt_tRNA";
        case miRNA: return "miRNA";
        case misc_RNA: return "misc_RNA";
        case rRNA: return "rRNA";
        case scRNA: return "scRNA";
        case snRNA: return "snRNA";
        case snoRNA: return "snoRNA";
        case ribozyme: return "ribozyme";
        case sRNA: return "sRNA";
        case scaRNA: return "scaRNA";
        case lncRNA: return "lncRNA";
        case Mt_tRNA_pseudogene: return "Mt_tRNA_pseudogene";
        case tRNA_pseudogene: return "tRNA_pseudogene";
        case snoRNA_pseudogene: return "snoRNA_pseudogene";
        case snRNA_pseudogene: return "snRNA_pseudogene";
        case scRNA_pseudogene: return "scRNA_pseudogene";
        case rRNA_pseudogene: return "rRNA_pseudogene";
        case misc_RNA_pseudogene: return "misc_RNA_pseudogene";
        case miRNA_pseudogene: return "miRNA_pseudogene";
        case TEC: return "TEC";
        case nonsense_mediated_decay: return "nonsense_mediated_decay";
        case non_stop_decay: return "non_stop_decay";
        case retained_intron: return "retained_intron";
        case protein_coding: return "protein_coding";
        case protein_coding_LoF: return "protein_coding_LoF";
        case protein_coding_CDS_not_defined: return "protein_coding_CDS_not_defined";
        case processed_transcript: return "processed_transcript";
        case non_coding: return "non_coding";
        case ambiguous_orf: return "ambiguous_orf";
        case sense_intronic: return "sense_intronic";
        case sense_overlapping: return "sense_overlapping";
        case antisense: return "antisense";
        case known_ncrna: return "known_ncrna";
        case pseudogene: return "pseudogene";
        case processed_pseudogene: return "processed_pseudogene";
        case polymorphic_pseudogene: return "polymorphic_pseudogene";
        case retrotransposed: return "retrotransposed";
        case transcribed_processed_pseudogene: return "transcribed_processed_pseudogene";
        case transcribed_unprocessed_pseudogene: return "transcribed_unprocessed_pseudogene";
        case transcribed_unitary_pseudogene: return "transcribed_unitary_pseudogene";
        case translated_processed_pseudogene: return "translated_processed_pseudogene";
        case translated_unprocessed_pseudogene: return "translated_unprocessed_pseudogene";
        case unitary_pseudogene: return "unitary_pseudogene";
        case unprocessed_pseudogene: return "unprocessed_pseudogene";
        case artifact: return "artifact";
        case lincRNA: return "lincRNA";
        case macro_lncRNA: return "macro_lncRNA";
        case threeprime_overlapping_ncRNA: return "3prime_overlapping_ncRNA";
        case disrupted_domain: return "disrupted_domain";
        case vaultRNA: return "vaultRNA";
        case bidirectional_promoter_lncRNA: return "bidirectional_promoter_lncRNA";
        default: return "Unassigned";
    }
}

BioType getBioTypeEnum(const std::string& name) {
    static const std::unordered_map<std::string, BioType> biotype_map = {
            {"IG_C_gene", BioType::IG_C_gene},
            {"IG_D_gene", BioType::IG_D_gene},
            {"IG_J_gene", BioType::IG_J_gene},
            {"IG_LV_gene", BioType::IG_LV_gene},
            {"IG_V_gene", BioType::IG_V_gene},
            {"TR_C_gene", BioType::TR_C_gene},
            {"TR_J_gene", BioType::TR_J_gene},
            {"TR_V_gene", BioType::TR_V_gene},
            {"TR_D_gene", BioType::TR_D_gene},
            {"IG_pseudogene", BioType::IG_pseudogene},
            {"IG_C_pseudogene", BioType::IG_C_pseudogene},
            {"IG_J_pseudogene", BioType::IG_J_pseudogene},
            {"IG_V_pseudogene", BioType::IG_V_pseudogene},
            {"TR_V_pseudogene", BioType::TR_V_pseudogene},
            {"TR_J_pseudogene", BioType::TR_J_pseudogene},
            {"Mt_rRNA", BioType::Mt_rRNA},
            {"Mt_tRNA", BioType::Mt_tRNA},
            {"miRNA", BioType::miRNA},
            {"misc_RNA", BioType::misc_RNA},
            {"rRNA", BioType::rRNA},
            {"scRNA", BioType::scRNA},
            {"snRNA", BioType::snRNA},
            {"snoRNA", BioType::snoRNA},
            {"ribozyme", BioType::ribozyme},
            {"sRNA", BioType::sRNA},
            {"scaRNA", BioType::scaRNA},
            {"lncRNA", BioType::lncRNA},
            {"Mt_tRNA_pseudogene", BioType::Mt_tRNA_pseudogene},
            {"tRNA_pseudogene", BioType::tRNA_pseudogene},
            {"snoRNA_pseudogene", BioType::snoRNA_pseudogene},
            {"snRNA_pseudogene", BioType::snRNA_pseudogene},
            {"scRNA_pseudogene", BioType::scRNA_pseudogene},
            {"rRNA_pseudogene", BioType::rRNA_pseudogene},
            {"misc_RNA_pseudogene", BioType::misc_RNA_pseudogene},
            {"miRNA_pseudogene", BioType::miRNA_pseudogene},
            {"TEC", BioType::TEC},
            {"nonsense_mediated_decay", BioType::nonsense_mediated_decay},
            {"non_stop_decay", BioType::non_stop_decay},
            {"retained_intron", BioType::retained_intron},
            {"protein_coding", BioType::protein_coding},
            {"protein_coding_LoF", BioType::protein_coding_LoF},
            {"protein_coding_CDS_not_defined", BioType::protein_coding_CDS_not_defined},
            {"processed_transcript", BioType::processed_transcript},
            {"non_coding", BioType::non_coding},
            {"ambiguous_orf", BioType::ambiguous_orf},
            {"sense_intronic", BioType::sense_intronic},
            {"sense_overlapping", BioType::sense_overlapping},
            {"antisense", BioType::antisense},
            {"known_ncrna", BioType::known_ncrna},
            {"pseudogene", BioType::pseudogene},
            {"processed_pseudogene", BioType::processed_pseudogene},
            {"polymorphic_pseudogene", BioType::polymorphic_pseudogene},
            {"retrotransposed", BioType::retrotransposed},
            {"transcribed_processed_pseudogene", BioType::transcribed_processed_pseudogene},
            {"transcribed_unprocessed_pseudogene", BioType::transcribed_unprocessed_pseudogene},
            {"transcribed_unitary_pseudogene", BioType::transcribed_unitary_pseudogene},
            {"translated_processed_pseudogene", BioType::translated_processed_pseudogene},
            {"translated_unprocessed_pseudogene", BioType::translated_unprocessed_pseudogene},
            {"unitary_pseudogene", BioType::unitary_pseudogene},
            {"unprocessed_pseudogene", BioType::unprocessed_pseudogene},
            {"artifact", BioType::artifact},
            {"lincRNA", BioType::lincRNA},
            {"macro_lncRNA", BioType::macro_lncRNA},
            {"threeprime_overlapping_ncRNA", BioType::threeprime_overlapping_ncRNA},
            {"disrupted_domain", BioType::disrupted_domain},
            {"vaultRNA", BioType::vaultRNA},
            {"bidirectional_promoter_lncRNA", BioType::bidirectional_promoter_lncRNA}
    };
    auto it = biotype_map.find(name);
    if (it != biotype_map.end()) {
        return it->second;
    } else return BioType::UNASSIGNED;
}

#endif //TRIPLEXGENEMAP_BIOTYPES_H
