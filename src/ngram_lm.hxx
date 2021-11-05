#ifndef __NGRAM_LM_HXX__
#define __NGRAM_LM_HXX__

#include <vector>
#include <cstdint>
#include <stdexcept>

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>

#include "vocabulary.hxx"
#include "Ngram.h"

namespace py = pybind11;

class NgramLM {
public:
    NgramLM(Vocabulary& vocabulary, unsigned order = 4) 
        : vocab(vocabulary), lm(vocab.get_vocab(), order), order(order - 1) {}
    // we save the context length, not the n-gram length here

public:
    float word_logprob(VocabIndex word, py::list context) {
        auto length = context.size();
        size_t i = 0;
        VocabIndex c_context[order + 1];
        
        auto end = context.begin() + order;
        if(length < order) {
            end = context.end(); }

        for(auto it = context.begin(); it != end; ++it, i++) {
            c_context[i] = it->cast<VocabIndex>(); }
        c_context[i] = Vocab_None;

        return lm.wordProb(word, c_context);
    }

    float word_prob(VocabIndex word, py::list context) {
        auto logp = word_logprob(word, context);
        return static_cast<float>(LogPtoProb(logp));
    }

    py::array_t<float> word_prob(py::list context) {
        auto result = py::array_t<float>(words.size());
        auto r = result.mutable_unchecked<1>();
            
        auto length = context.size();
        auto end = context.begin() + order;
        if(length < order) {
            end = context.end(); }

        size_t i = 0;
        VocabIndex c_context[order + 1];
        for(auto it = context.begin(); it != end; ++it, i++) {
            c_context[i] = it->cast<VocabIndex>(); }
        c_context[i] = Vocab_None;

        py::ssize_t j = 0;
        for(auto it = words.begin(); it != words.end(); ++it, j++) {
            auto logp = lm.wordProb(*it, c_context);
            r(j) = static_cast<float>(LogPtoProb(logp));
        }
        
        return result;
    }
    
    py::array_t<float> word_prob_step(py::list indices, py::list states) {
        if(indices.size() != states.size()) {
            throw std::runtime_error("expected indices and states to be same length");
        }

        VocabIndex c_context[order + 1];
        auto result = py::array_t<float>({static_cast<py::ssize_t>(indices.size()), static_cast<py::ssize_t>(words.size())});
        auto r = result.mutable_unchecked<2>();

        // we expect the states to be in reverse already, we need to copy and Vocab_None
        for(py::ssize_t i = 0; i < indices.size(); i++) {
            py::ssize_t j = 0;
            
            py::list c = states[i].cast<py::list>();
            if(c.size() > order) {
                throw std::runtime_error("state cannot be larger than order");
            }

            for(py::ssize_t k = 0; k < c.size(); k++) {
                c_context[k] = c[k].cast<VocabIndex>();
            }
            c_context[c.size()] = Vocab_None;

            for(auto it = words.begin(); it != words.end(); ++it, j++) {
                auto logp = lm.wordProb(*it, c_context);
                r(i, j) = static_cast<float>(LogPtoProb(logp));
            }
        }

        return result;
    }
    
    py::array_t<float> word_prob_batch(py::array_t<VocabIndex, py::array::c_style | py::array::forcecast> batch,
                                       py::array_t<py::ssize_t, py::array::c_style | py::array::forcecast> lengths) {
        if(batch.ndim() != 2) {
            throw std::runtime_error("batch expexted to be shape [B, T]");
        }

        if(batch.shape(0) != lengths.shape(0)) {
            throw std::runtime_error("batch dimension need to match");
        }

        auto indices = batch.unchecked<2>();
        auto ulengths = lengths.unchecked<1>();

        auto result = py::array_t<float>({indices.shape(0), indices.shape(1), static_cast<py::ssize_t>(words.size())});
        auto r = result.mutable_unchecked<3>();

        VocabIndex c_context[order + 1];
        
        for(py::ssize_t b = 0; b < indices.shape(0); b++) {
            for(py::ssize_t t = 0; t < ulengths(b); t++) {
                //construct context here; in reverse
                py::ssize_t i = 0;
                for(; i < (t >= order ? order : t + 1); i++) {
                    c_context[i] = indices(b, t - i); }
                c_context[i] = Vocab_None;
                
                i = 0;
                for(auto it = words.begin(); it != words.end(); ++it, i++) {
                    auto logp = lm.wordProb(*it, c_context);
                    r(b, t, i) = static_cast<float>(LogPtoProb(logp));
                }
            }
        }
        
        return result;
    }

    bool read(std::string const& filename, bool limit_vocab = false) {
        File file(filename.c_str(), "r");
        bool r = lm.read(file, limit_vocab);

        for(VocabIndex k = 0; k < vocab.size(); k++) {
            if(vocab.non_event(k)) { continue; }
            words.push_back(k);
        }
        
        return r;
    }

private:
    Vocabulary& vocab;
    Ngram lm;
    unsigned order;
    std::vector<VocabIndex> words;
};

#endif
