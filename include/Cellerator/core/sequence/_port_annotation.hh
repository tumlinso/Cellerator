#ifndef ANNOTATION_HEADER_CLASS
#define ANNOTATION_HEADER_CLASS

#include <iostream>
#include <fstream>
#include <sstream>
#include <unordered_map>
#include <string>
#include <vector>
#include <memory>

#include "Gene.hh"

class Annotation {
public:
    std::unordered_map<std::string, Gene*>* MapGeneID;
    std::unordered_map<std::string, Gene*>* MapGeneNames;

    explicit Annotation(const std::string &filename) {
        MapGeneID = new std::unordered_map<std::string, Gene*>();
        MapGeneNames = new std::unordered_map<std::string, Gene*>();

        std::ifstream file(filename);
        if (!file.is_open()) {
            throw std::runtime_error("Could not open file " + filename);
        }

        std::string line;
        while (std::getline(file, line)) {
            if (line.empty() || line[0] == '#') {
                continue;
            }
            std::istringstream lineStream(line);
            std::string field;
            std::vector<std::string> fields;

            while (std::getline(lineStream, field, '\t')) {
                fields.push_back(field);
            }

            if (fields.size() < 9) continue;

            if (fields[2] == "gene") {
                std::istringstream attributesStream(fields[8]);
                std::string attribute;
                std::vector<std::string> attributes;

                while (std::getline(attributesStream, attribute, ';')) {
                    attributes.push_back(attribute);
                }

                std::string chromosome;
                int start, end;
                Feature::Strand strand;
                BioType biotype;
                std::string id, name;

                // chromosome
                chromosome = fields[0];

                // start, end
                start = std::stoi(fields[3]);
                end = std::stoi(fields[4]);

                // strand
                char strandChar = fields[6][0];
                strand = strandChar == '+' ? Feature::Strand::SENSE : strandChar == '-' ?
                        Feature::Strand::ANTISENSE : Feature::Strand::UNASSIGNED;

                // biotype
                std::string stringBioType = attributes[1].substr(12);
                stringBioType = stringBioType.substr(0, stringBioType.find('"'));
                biotype = getBioTypeEnum(stringBioType);

                // name, id
                name = attributes[2].substr(12);
                name = name.substr(0, name.find('"'));
                id = attributes[0].substr(9);
                id = id.substr(0, id.find('.'));

                // alloc gene
                auto gene = new Gene(chromosome, start, end, strand, biotype, name, id);

                // emplace gene in maps
                MapGeneNames->emplace(gene->name, gene);
                MapGeneID->emplace(gene->id, gene);
            }
        }

        file.clear();
        file.seekg(0, std::ios::beg);
        while (std::getline(file, line)) {
            if (line.empty() || line[0] == '#') {
                continue;
            }
            std::istringstream lineStream(line);
            std::string field;
            std::vector<std::string> fields;

            while (std::getline(lineStream, field, '\t')) {
                fields.push_back(field);
            }

            if (fields.size() < 9) continue;

            if (fields[2] == "transcript") {
                std::istringstream attributesStream(fields[8]);
                std::string attribute;
                std::vector<std::string> attributes;

                while (std::getline(attributesStream, attribute, ';')) {
                    attributes.push_back(attribute);
                }

                std::string parentGeneID;

                std::string chromosome;
                int start, end;
                Feature::Strand strand;
                BioType biotype;
                std::string id, name;

                // parent gene id -- to assign transcript to gene
                parentGeneID = attributes[0].substr(9);
                parentGeneID = parentGeneID.substr(0, parentGeneID.find('.'));

                // chromosome
                chromosome = fields[0];

                // start, end
                start = std::stoi(fields[3]);
                end = std::stoi(fields[4]);

                // strand
                char strandChar = fields[6][0];
                strand = strandChar == '+' ? Feature::Strand::SENSE : strandChar == '-' ?
                        Feature::Strand::ANTISENSE : Feature::Strand::UNASSIGNED;

                // biotype
                std::string stringBioType = attributes[4].substr(18);
                stringBioType = stringBioType.substr(0, stringBioType.find('"'));
                biotype = getBioTypeEnum(stringBioType);

                // name, id
                name = attributes[5].substr(18);
                name = name.substr(0, name.find('"'));
                id = attributes[1].substr(16);
                id = id.substr(0, id.find('.'));

                auto it = MapGeneID->find(parentGeneID);
                if (it != MapGeneID->end()) {
                    auto transcript = new Gene::Transcript(chromosome, start, end, strand, biotype,
                                                           name, id);
                    it->second->MapTranscriptID->emplace(transcript->name, transcript);
                    it->second->transcripts->push_back(transcript);
                }
            }
        }

        file.clear();
        file.seekg(0, std::ios::beg);
        while (std::getline(file, line)) {
            if (line.empty() || line[0] == '#') {
                continue;
            }

            std::istringstream lineStream(line);
            std::string field;
            std::vector<std::string> fields;

            while (std::getline(lineStream, field, '\t')) {
                fields.push_back(field);
            }

            if (fields.size() < 9) continue;

            if (fields[2] == "exon") {
                std::istringstream attributesStream(fields[8]);
                std::string attribute;
                std::vector<std::string> attributes;

                while (std::getline(attributesStream, attribute, ';')) {
                    attributes.push_back(attribute);
                }

                std::string parentGeneID, parentTranscriptID;

                std::string chromosome;
                int start, end;
                Feature::Strand strand;
                int order;

                // parent gene id -- seek to gene seeking transcript
                parentGeneID = attributes[0].substr(9);
                parentGeneID = parentGeneID.substr(0, parentGeneID.find('.'));

                // parent transcript id -- seek to transcript assign exon
                parentTranscriptID = attributes[1].substr(16);
                parentTranscriptID = parentTranscriptID.substr(0, parentTranscriptID.find('.'));

                // chromosome
                chromosome = fields[0];

                // start, end
                start = std::stoi(fields[3]);
                end = std::stoi(fields[4]);

                // strand
                char strandChar = fields[6][0];
                strand = strandChar == '+' ? Feature::Strand::SENSE : strandChar == '-' ?
                        Feature::Strand::ANTISENSE : Feature::Strand::UNASSIGNED;

                // order
                order = std::stoi(attributes[6].substr(12));

                auto it = MapGeneID->find(parentGeneID);
                if (it != MapGeneID->end()) {
                    auto ittr = it->second->MapTranscriptID->find(parentTranscriptID);
                    if (ittr != it->second->MapTranscriptID->end()) {
                        auto exon = new Gene::Transcript::Exon(chromosome, start, end, strand, order);
                        ittr->second->exons->push_back(exon);
                    }
                }
            }
        }
    }

    ~Annotation() {
        for (auto &pair : *MapGeneID) {
            delete pair.second;
        }
        delete MapGeneID;
        delete MapGeneNames;
    }
};


#endif // ANNOTATION_HEADER_CLASS