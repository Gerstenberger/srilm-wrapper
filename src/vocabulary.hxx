#ifndef __VOCABULARY_HXX__
#define __VOCABULARY_HXX__

#include <algorithm>
#include <string>
#include <vector>

#include "Prob.h"
#include "Vocab.h"
#include "File.h"
#include "Ngram.h"

#include <pybind11/pybind11.h>

namespace py = pybind11;

class Vocabulary {
public:
    Vocabulary(bool unk_as_word = true, bool to_lower = false) {
        vocab.unkIsWord() = unk_as_word;
        vocab.toLower() = to_lower; }

public:
    const char * word(VocabIndex index) {
        return vocab.getWord(index); }
    
    VocabIndex index(const char* key) {
        return vocab.getIndex(key); }
    
    size_t size() {
        return vocab.numWords(); }
    
    VocabIndex high_index() { 
        return vocab.highIndex(); }
    
    VocabIndex unk_index() {
        return vocab.unkIndex(); }

    VocabIndex ss_index() { 
        return vocab.ssIndex(); }

    VocabIndex se_index() { 
        return vocab.seIndex(); }
    
    bool non_event(VocabIndex index) {
        return vocab.isNonEvent(index); }
    
    size_t read(std::string const& filename) {
        File file(filename.c_str(), "r");
        return vocab.read(file);
    }
    
    Vocab& get_vocab() {
        return vocab; }

private:
    Vocab vocab; 
};

#endif
