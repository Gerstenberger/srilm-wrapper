
#include <pybind11/pybind11.h>

#include "vocabulary.hxx"
#include "ngram_lm.hxx"

namespace py = pybind11;

PYBIND11_MODULE(srilm, m) {
    py::class_<Vocabulary> vocab(m, "Vocabulary");
    vocab.def(py::init<bool, bool>())
         .def("word", &Vocabulary::word)
         .def("index", &Vocabulary::index)
         .def("size", &Vocabulary::size)
         .def("high_index", &Vocabulary::high_index)
         .def("unk_index", &Vocabulary::unk_index)
         .def("ss_index", &Vocabulary::ss_index)
         .def("se_index", &Vocabulary::se_index)
         .def("non_event", &Vocabulary::non_event)
         .def("read", &Vocabulary::read);
    
    py::class_<NgramLM> lm(m, "NgramLM");
    lm.def(py::init<Vocabulary&, unsigned>())
      .def("word_logprob", &NgramLM::word_logprob)
      .def("word_prob", py::overload_cast<VocabIndex, py::list>(&NgramLM::word_prob))
      .def("word_prob", py::overload_cast<py::list>(&NgramLM::word_prob))
      .def("word_prob_step", &NgramLM::word_prob_step)
      .def("word_prob_batch", &NgramLM::word_prob_batch)
      .def("read", &NgramLM::read);
}