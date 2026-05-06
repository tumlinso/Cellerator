#ifndef GENE_HEADER_CLASS
#define GENE_HEADER_CLASS

#include <utility>
#include <vector>
#include <string>
#include <unordered_map>

#include "Feature.hh"
#include "BioType.hh"

class Gene : public virtual Feature {
public:
    class Transcript : public virtual Feature {
    public:
        class Exon : public virtual Feature {
        public:
            [[maybe_unused]] int order;
            Exon(std::string &chromosome, long start, long end, Feature::Strand strand, int order) :
            Feature(chromosome, start, end, strand) {
                this->order = order;
            }
        };

        std::vector<Exon*>* exons;

        Transcript(std::string &chromosome, int start, int end, Feature::Strand strand, BioType biotype,
                   std::string &name, std::string &id) :Feature(chromosome, start, end, strand, biotype, name, id) {
            exons = new std::vector<Exon*>();
        }

        ~Transcript() {
            for (auto &exon : *exons) {
                delete exon;
            }
            delete exons;
        }
    };

    std::unordered_map<std::string, Transcript*>* MapTranscriptID;
    std::vector<Transcript*>* transcripts;

    Gene(std::string &chromosome, int start, int end, Feature::Strand strand, BioType biotype,
         std::string &name, std::string &id) : Feature(chromosome, start, end, strand, biotype, name, id) {
        MapTranscriptID = new std::unordered_map<std::string, Transcript*>();
        transcripts = new std::vector<Transcript*>();
    }

    ~Gene() {
        for (auto &pair : *MapTranscriptID) {
            delete pair.second;
        }
        delete MapTranscriptID;
        delete transcripts;
    }
};


#endif //GENE_HEADER_CLASS
