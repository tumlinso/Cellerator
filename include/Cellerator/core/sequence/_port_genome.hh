#ifndef GENOME_HEADER_CLASS
#define GENOME_HEADER_CLASS

#include <string>
#include <cstring>
#include <vector>
#include <unordered_map>
#include <fstream>
#include <sstream>
#include <stdexcept>

#include "Sequence.hh"
#include "Gene.hh"
#include "Feature.hh"

class Genome : public SequenceDictionary {
public:
    explicit Genome(const char *filename) : SequenceDictionary(filename){}

   ~Genome() = default;

    std::vector<Sequence*> *getTranscriptSequences(Gene &gene) const {
        auto sequences = new std::vector<Sequence*>();
        for (auto &transcript : *gene.transcripts) {
            char *excerpt;
            try {
                excerpt = getBase(*transcript);
            } catch (std::runtime_error &e) {
                // add later for verbose std::cout << "Warning: " << e.what() << "\n";
                continue;
            }

            unsigned long length = transcript->end - transcript->start + 1;
            if (transcript->strand == Feature::Strand::UNASSIGNED) [[unlikely]] throw std::runtime_error("Transcript strand must be specified to get RNA sequences.");

            // an RNA transcript can only be sense, it can come from the sense or antisense strand of the DNA, but it is always sense
            char *sequenceString = (transcript->strand == Feature::SENSE) ? excerpt : Sequence::reverseComplement(excerpt, length);
            if (transcript->strand == Feature::ANTISENSE) delete[] excerpt;

            for (unsigned long i = 0; i < length; ++i) {
                switch (sequenceString[i]) {
                    [[likely]] case 'T': sequenceString[i] = 'U'; break;
                    [[unlikely]] default: break;
                }
            }

            auto sequence = new Sequence(*transcript, sequenceString);
            sequences->push_back(sequence);
        }
        return sequences;
    }

    Sequence getSequence(Feature &feature) const {
        char *excerpt;
        try {
            excerpt = getBase(feature);
        } catch (std::runtime_error &e) {
            // add back for verbose std::cout << "Warning: " << e.what() << "\n";
            return Sequence(feature, nullptr, nullptr);
        }
        unsigned long length = feature.end - feature.start + 1;

        char *sense = (feature.strand == Feature::SENSE || feature.strand == Feature::Strand::UNASSIGNED) ?
                excerpt : nullptr;
        char *antisense = (feature.strand == Feature::ANTISENSE || feature.strand == Feature::Strand::UNASSIGNED) ?
                Sequence::reverseComplement(excerpt, length) : nullptr;

        if (feature.strand == Feature::ANTISENSE) delete[] excerpt;

        return Sequence(feature, sense, antisense);
    }

private:
    char* getBase(Feature &feature) const {
        auto it = (*this).find(feature.chromosome);
        if (it == (*this).end()) [[unlikely]] throw std::runtime_error("Chromosome not found.");
        if (feature.start < 1 || feature.end > it->second->end || feature.start > feature.end)
            [[unlikely]] throw std::runtime_error("Invalid range in sequence call.");

        unsigned long length = feature.end - feature.start + 1;

        char *excerpt = new char[length + 1];
        std::memcpy(excerpt, it->second->sense + feature.start - 1, length);
        excerpt[length] = '\0';

        return excerpt;
    }
};

#endif // GENOME_HEADER_CLASS