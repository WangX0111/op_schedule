#ifndef GA_HPP
#define GA_HPP

#include <cstdint>
#include <functional>
#include <vector>

template <typename CostFuncTy>
class GeneticAlgorithmBase {
public:
  GeneticAlgorithmBase(CostFuncTy func, int64_t n_dim, int64_t size_pop,
      int64_t max_iter, double prob_mut, bool early_stop)
      : func_(func), n_dim_(n_dim), size_pop_(size_pop), max_iter_(max_iter),
        prob_mut_(prob_mut), early_stop_(early_stop) {}

  void XToY() {
    for (int i = 0; i < size_pop_; ++i) {
      Y_raw_[i] = func_(X_[i]);
    }
  }

  virtual void ChromoToX() = 0;
  virtual void Ranking() = 0;
  virtual void Selection() = 0;
  virtual void Crossover() = 0;
  virtual void Mutation() = 0;

  void Run(){
    for(int i = 0; i < max_iter_; ++i){
      ChromoToX();
      XToY();

      Ranking();
      Selection();
      Crossover();
      Mutation();
    }
  }

private:
  int64_t size_pop_;
  int64_t max_iter_;
  int64_t n_dim_;

  std::vector<std::vector<double>> Chromo_;
  std::vector<std::vector<double>> X_;
  std::vector<double> Y_raw_;

  double prob_mut_;
  bool early_stop_;
  CostFuncTy func_;
};

#endif /* GA_HPP */
