#ifndef SEQUENCE_HEADER_CLASS
#define SEQUENCE_HEADER_CLASS

#include <cstddef>
#include <cassert>
#include <array>
#include <unordered_map>
#include <vector>
#include <string>
#include <fstream>
#include <sstream>
#include <cstring>
#include <bitset>
#include "Feature.hh"

typedef enum {
    A = 0b00,
    C = 0b01,
    G = 0b10,
    T = 0b11,
} Base;

class Sequence {
private:
    std::vector<uint64_t> data;
protected:
    inline void *encodeString(const std::string& SequenceString) {
        assert(SequenceString.length() == length);
        for (size_t i = 0; i < SequenceString.length(); ++i) {
            size_t index = i / 32;
            size_t shift = (i % 32) * 2;
            this->data[index] |= packNucleotide(charToBase(SequenceString[i])) << shift;
        }
    }

    static inline constexpr std::array<bool, 256> definedBases = [] {
        std::array<bool, 256> table = {};
        for (size_t i = 0; i < 256; ++i) {
            table[i] = false;
        }
        table['A'] = true;
        table['C'] = true;
        table['G'] = true;
        table['T'] = true;
        table['U'] = true;
        table['a'] = true;
        table['c'] = true;
        table['g'] = true;
        table['t'] = true;
        table['u'] = true;
        return table;
    }();

    static inline constexpr std::array<Base, 256> baseCharTable = [] {
        std::array<Base, 256> table = {};
        table['A'] = A;
        table['C'] = C;
        table['G'] = G;
        table['T'] = T;
        table['U'] = T;
        table['a'] = A;
        table['c'] = C;
        table['g'] = G;
        table['t'] = T;
        table['u'] = T;
        return table;
    }();

    static inline Base charToBase(char nucleotide) {
        assert (definedBases[nucleotide]);
        return baseCharTable[static_cast<unsigned char>(nucleotide)];
    }

    static inline char baseToChar(Base base) {
        switch(base) {
            case A: return 'A';
            case C: return 'C';
            case G: return 'G';
            case T: return 'T';
            [[unlikely]] default: throw std::invalid_argument("Invalid base");
        }
    }

    inline static unsigned long packNucleotide(Base nucleotide) {
        return static_cast<uint64_t>(nucleotide);
    }

    inline static Base unpackNucleotide(unsigned long packed) {
        return static_cast<Base>(packed);
    }

public:
    size_t length;

    explicit Sequence(size_t length) : length(length) {
        this->data.resize((length * 2 + 63) / 64, 0);
    }

    explicit Sequence(const std::string& sequenceString) : Sequence(sequenceString.length()) {
        encodeString(sequenceString);
    }

    ~Sequence() = default;

    [[nodiscard]] inline Sequence* reverseComplement() const {
        auto *reverseComplement = new Sequence(*this);
        size_t size = this->data.size();
        for (size_t i = 0; i < size; ++i) {
            unsigned long current = (*this).data[i], complement = current ^ 0xffffffffffffffff, reversed = 0;
            for (int j = 0; j < 32; ++j) {
                reversed <<= 2;
                reversed |= (complement & 0b11);
                complement >>= 2;
            }
            (*reverseComplement).data[size - 1 - i] = reversed;
        }

        size_t overhang = ((size * 64) / 2) - length;
        for (size_t i = 0; i < size - 1; ++i) {
            uint64_t buffer = 0 | ((*reverseComplement).data[i] << overhang);
            for (size_t j = 0; j < overhang; ++j) {
                buffer |= ((*reverseComplement).data[i + 1] >> (64 - j * 2)) & 0b11;
            }
            (*reverseComplement).data[i] = buffer;
        }

        (*reverseComplement).data[size - 1] >>= overhang;

        return reverseComplement;
    }

    inline Base operator[] (size_t pos) const {
        [[unlikely]] assert(pos < length);
        size_t index = pos / 32;
        size_t shift = (pos % 32) * 2;
        return unpackNucleotide((data[index] >> shift) & 0b11);
    }

    inline void operator[] (size_t pos, Base base) {
        [[unlikely]] assert(pos < length);
        size_t index = pos / 32;
        size_t shift = (pos % 32) * 2;
        data[index] &= ~(0b11ULL << shift);
        data[index] |= packNucleotide(base) << shift;
    }

    [[nodiscard]] char getBase(unsigned long pos) const {
        return baseToChar((*this)[pos]);
    }

    [[nodiscard]] char setBase(size_t pos, Base base) {
        char oldBase = getBase(pos);
        (*this)[pos, base];
        return oldBase;
    }

    [[nodiscard]] std::string toString() const {
        std::string sequenceString;
        sequenceString.reserve(length);
        for (size_t i = 0; i < length; ++i) {
            sequenceString.push_back(getBase(i));
        }
        return sequenceString;
    }
};

class SequenceArray : public std::vector<Sequence*> {
public:
    explicit SequenceArray (const char *fastaFilePath) {
        std::ifstream file(fastaFilePath);
        if (!file.is_open()) {
            throw std::runtime_error("Could not open the FASTA file.");
        }

        Feature *current = nullptr;
        std::string line;
        std::stringstream runningBuffer;

        while (std::getline(file, line)) {
            if (line.empty()) {
                continue;
            }
            if (line[0] == '>') {
                if (current != nullptr) {
                    std::string sequenceString = runningBuffer.str();
                    current->end = sequenceString.length();
                    (*this).push_back(new Sequence(*current, sequenceString));
                    runningBuffer.str("");
                    delete current;
                    current = nullptr;
                }
                std::string identifier = line.substr(1);
                current = new Feature("null", 1, 0, Feature::Strand::UNASSIGNED,
                                      BioType::UNASSIGNED, "null", identifier);
            } else {
                runningBuffer << line;
            }
        }
        if (current != nullptr) {
            std::string sequenceString = runningBuffer.str();
            current->end = sequenceString.length();
            (*this).push_back(new Sequence(*current, sequenceString));
            delete current;
        }
    }

    explicit SequenceArray(SequenceArray *sequences) {
        for (auto &sequence : *sequences) {
            char *sequenceString = new char[sequence->end + 1];
            std::strcpy(sequenceString, sequence->toString().c_str());
            (*this).push_back(new Sequence(*sequence, sequenceString));
        }
    }

    ~SequenceArray() {
        for (auto &sequence : *this) {
            delete sequence;
        }
    }
};


class SequenceDictionary : public std::unordered_map<std::string, Sequence*> {
public:
    explicit SequenceDictionary(const char* fastaFilePath) {
        vectorized = new SequenceArray(fastaFilePath);
        for (auto &sequence : *vectorized) {
            (*this)[sequence->id] = sequence;
        }
    }

    explicit SequenceDictionary(SequenceArray *vectorized) {
        this->vectorized = vectorized;
        for (auto &sequence : *vectorized) {
            (*this)[sequence->id] = sequence;
        }
    }

    ~SequenceDictionary() {
        delete vectorized;
    }

private:
    SequenceArray *vectorized;
};


#endif //SEQUENCE_HEADER_CLASS
