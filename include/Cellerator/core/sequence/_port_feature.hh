#ifndef FEATURE_HEADER_CLASS
#define FEATURE_HEADER_CLASS

#include <string>
#include <utility>

#include "Sequence.hh"

class Feature {
public:
    typedef enum {UNDEFINED, SENSE, ANTISENSE} Strand;

    Sequence *sequence;
    size_t start, end;
    Strand strand;

    Feature(Sequence *sequence, size_t start, size_t end, Strand strand = Strand::UNDEFINED)
    : sequence(sequence), start(start), end(end), strand(strand) {};

    ~Feature() = default;

    [[nodiscard]] unsigned long length() const {
        return (end - start) + 1;
    }
};

#endif //FEATURE_HEADER_CLASS
