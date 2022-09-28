#ifndef CCOPT_CHROMO_H
#define CCOPT_CHROMO_H
#include <cassert>
#include <cmath>
#include <cstdint>
#include <string>
#include <vector>
namespace cc_opt {

struct BinaryChromo {
  BinaryChromo() = default;
  BinaryChromo(const std::string &gene) : gene(gene), fitness(0) {}
  std::string gene; // binary string
  double fitness;
};

struct FloatChromo {
  FloatChromo() = default;
  FloatChromo(const std::vector<double> gene) : gene(gene) {}
  std::vector<double> gene; // [0, 1]^n
  double fitness;
};

std::vector<double> Decode(const BinaryChromo &bin_chromo, int64_t seg_len,
                           const std::vector<double> &lower_bound,
                           const std::vector<double> &upper_bound) {

  assert(upper_bound > lower_bound);

  int64_t n_gene = bin_chromo.gene.size();
  assert(n_gene % seg_len == 0);
  double max_ = std::pow(2, seg_len) - 1;
  std::vector<double> features;
  for (int64_t i = 0; i < n_gene / seg_len; ++i) {

    assert(upper_bound[i] > lower_bound[i]);

    int64_t gene_begin = i * seg_len;
    const std::string &segment = bin_chromo.gene.substr(gene_begin, seg_len);

    double feature_val =
        lower_bound[i] +
        std::stoi(segment, 0, 2) * (upper_bound[i] - lower_bound[i]) / max_;
    features.push_back(feature_val);
  }
  return features;
}

std::vector<double> Decode(const FloatChromo &f_chromo,
                           const std::vector<double> &lower_bound,
                           const std::vector<double> &upper_bound) {

  int64_t n_gene = f_chromo.gene.size();

  std::vector<double> features;
  for (int64_t i = 0; i < n_gene; ++i) {
    assert(upper_bound[i] > lower_bound[i]);

    double feature_val =
        lower_bound[i] + f_chromo.gene[i] * (upper_bound[i] - lower_bound[i]);

    features.push_back(feature_val);
  }

  return features;
}

} // namespace cc_opt
#endif /* CCOPT_CHROMO_H */
