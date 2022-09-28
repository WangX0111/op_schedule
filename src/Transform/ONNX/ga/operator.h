#ifndef OPERATOR_H
#define OPERATOR_H

#include "chromo.h"

#include <algorithm>
#include <cassert>
#include <cstdint>
#include <random>
#include <string>
#include <type_traits>
#include <vector>

namespace cc_opt {

template <typename FloatTy>
inline std::vector<FloatTy> GenRandomFloatVec(int64_t size, double lower_bound,
                                              double upper_bound) {
  // engine
  std::random_device seed;
  std::ranlux48 rand_engine(seed());
  // distribution
  std::uniform_real_distribution<FloatTy> dist(lower_bound, upper_bound);

  auto gen = [&dist, &rand_engine]() { return dist(rand_engine); };

  std::vector<FloatTy> vec(size);
  std::generate(vec.begin(), vec.end(), gen);

  return vec;
}

template <typename IntTy>
inline std::vector<IntTy> GenRandomIntVec(int64_t size, int64_t lower_bound,
                                          int64_t upper_bound) {
  // engine
  std::random_device seed;
  std::ranlux48 rand_engine(seed());
  // distribution
  std::uniform_int_distribution<IntTy> dist(lower_bound, upper_bound);

  auto gen = [&dist, &rand_engine]() { return dist(rand_engine); };

  std::vector<IntTy> vec(size);
  std::generate(vec.begin(), vec.end(), gen);

  return vec;
}

template <typename FloatTy>
inline FloatTy GetRandomFloat(double lower_bound, double upper_bound) {
  // engine
  std::random_device seed;
  std::ranlux48 rand_engine(seed());
  // distribution
  std::uniform_real_distribution<FloatTy> dist(lower_bound, upper_bound);

  return dist(rand_engine);
}

template <typename IntTy>
inline IntTy GetRandomInt(int64_t lower_bound, int64_t upper_bound) {
  // engine
  std::random_device seed;
  std::ranlux48 rand_engine(seed());
  // distribution
  std::uniform_int_distribution<IntTy> dist(lower_bound, upper_bound);

  return dist(rand_engine);
}

inline void BinaryMutate(BinaryChromo &chromo, double mut_rate) {
  int64_t n_genes = chromo.gene.size();
  std::vector<double> mut_mask = GenRandomFloatVec<double>(n_genes, 0, 1);
  for (int64_t i = 0; i < n_genes; ++i) {
    if (mut_mask[i] < mut_rate) {
      chromo.gene[i] = (chromo.gene[i] == '0' ? '1' : '0');
    }
  }
}

inline void FloatMutate(FloatChromo &f_chromo, double mut_rate) {
  int64_t n_genes = f_chromo.gene.size();
  std::vector<double> mut_mask = GenRandomFloatVec<double>(n_genes, 0, 1);
  for (int64_t i = 0; i < n_genes; ++i) {
    if (mut_mask[i] < mut_rate) {
      f_chromo.gene[i] = GetRandomFloat<double>(0, 1);
    }
  }
}

template <typename ChromoTy>
inline std::vector<ChromoTy> Crossover2Point(const ChromoTy &parent_a,
                                             const ChromoTy &parent_b) {
  int64_t n_genes = parent_a.gene.size();
  ChromoTy child_a{parent_a.gene}, child_b{parent_b.gene};
  int64_t begin_idx = GetRandomInt<int64_t>(0, n_genes - 1);
  int64_t end_idx = GetRandomInt<int64_t>(0, n_genes - 1);
  if (begin_idx > end_idx) {
    std::swap(begin_idx, end_idx);
  }
  for (int64_t i = begin_idx; i < end_idx; ++i) {
    child_a.gene[i] = parent_b.gene.at(i);
    child_b.gene[i] = parent_a.gene.at(i);
  }
  return {child_a, child_b};
}

template <typename ChromoTy>
inline void SelectTournament(std::vector<ChromoTy> &population,
                             double remain_rate, int64_t candidate_num) {
  assert(candidate_num >= 2);

  int64_t pop_size = population.size();
  int64_t remain_num = pop_size * remain_rate;
  if (remain_num % 2 == 1) {
    remain_num -= 1;
  }
  assert(remain_num % 2 == 0);

  std::vector<ChromoTy> next_population;
  while (next_population.size() < remain_num) {
    auto candidate_idx =
        GenRandomIntVec<int64_t>(candidate_num, 0, pop_size - 1);
    ChromoTy *winner = &population[candidate_idx[0]];
    for (int64_t i = 1; i < candidate_idx.size(); ++i) {
      int64_t cand_idx = candidate_idx[i];
      if (population[cand_idx].fitness > winner->fitness) {
        winner = &population[cand_idx];
      }
    }
    next_population.push_back(*winner);
  }

  population = next_population;
}

} // namespace cc_opt

#endif /* OPERATOR_H */
