set(catalog_assembly "GRCm38.p6")
set(catalog_release "GENCODE M25")
set(catalog_filename "gencode.vM25.annotation.gtf.gz")
set(catalog_format "gtf")
set(catalog_source "https://ftp.ebi.ac.uk/pub/databases/gencode/Gencode_mouse/release_M25/${catalog_filename}")
set(catalog_sha256 "e6c9704cb5832b8a1c71431ecac9ea30057377c72cdc9a12dc5d0405ab066d5a")
set(catalog_subsets
    "${CMAKE_CURRENT_LIST_DIR}/mm10_m25/mitochondrial.ids"
    "${CMAKE_CURRENT_LIST_DIR}/mm10_m25/mitochondrial_protein_coding.ids")
