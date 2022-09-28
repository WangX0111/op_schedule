#ifndef CCOPT_GA_H
#define CCOPT_GA_H

#include "chromo.h"
#include "operator.h"

#include <cassert>
#include <cmath>
#include <cstdint>
#include <functional>
#include <iostream>
#include <vector>

constexpr int64_t kEarlyStopIter = 100;

namespace cc_opt {
template <typename CostFuncTy, typename ChromoType>
class GABase {
public:
  GABase(CostFuncTy cost_func, int64_t n_features, int64_t size_pop,
      int64_t max_iter, double prob_mut, const std::vector<double> &lower_bound,
      const std::vector<double> &upper_bound, bool early_stop)
      : cost_func_(cost_func), n_features_(n_features), size_pop_(size_pop),
        max_iter_(max_iter), prob_mut_(prob_mut), lower_bound_(lower_bound),
        upper_bound_(upper_bound), early_stop_(early_stop) {

    assert(size_pop_ % 2 == 0);
  }

  virtual double GetChromoFit(const ChromoType &chromo) = 0;
  virtual std::vector<double> DecodeChromo(const ChromoType &chromo) = 0;
  virtual void InitPopulation() = 0;
  virtual void Selection() = 0;
  virtual void CrossOver() = 0;
  virtual void Mutation() = 0;

  std::vector<double> GetBestFeature() const { return best_feature_; }

  double GetBestCost() const { return best_cost_; }

  ChromoType GetBestChromo() {
    ChromoType best = this->population_.front();
    for (const auto &chromo : this->population_) {
      if (chromo.fitness > best.fitness) {
        best = chromo;
      }
    }
    return best;
  }

  void Run() {
    this->InitPopulation();
    ChromoType best_chromo = this->GetBestChromo();
    int64_t best_iter = 0;
    for (int64_t i = 0; i < this->max_iter_; ++i) {
      this->Selection();
      this->CrossOver();
      this->Mutation();

      auto gen_best_chromo = this->GetBestChromo();
      if (gen_best_chromo.fitness > best_chromo.fitness) {
        best_chromo = gen_best_chromo;
        best_iter = i;
      }
      auto features = this->DecodeChromo(best_chromo);
      auto cost = -best_chromo.fitness;
      std::cout << "iter: " << i << " cost: " << cost << " \n";

      if (this->early_stop_ && i - best_iter >= kEarlyStopIter) {
        std::cout << "early stop at: " << i << "\n";
        break;
      }
    }
    this->best_feature_ = this->DecodeChromo(best_chromo);
    this->best_cost_ = -best_chromo.fitness;
    std::cout << "best cost: " << this->best_cost_
              << " at iteration: " << best_iter << "\n";
  }

protected:
  int64_t size_pop_;
  int64_t max_iter_;
  int64_t n_features_;

  std::vector<double> lower_bound_;
  std::vector<double> upper_bound_;

  double prob_mut_;
  bool early_stop_;

  std::vector<ChromoType> population_;
  std::vector<std::vector<double>> features_;

  std::vector<double> best_feature_;
  double best_cost_;

  CostFuncTy cost_func_;
};

template <typename CostFuncTy>
class GA : public GABase<CostFuncTy, BinaryChromo> {
public:
  GA(CostFuncTy cost_func, int64_t n_features, int64_t size_pop,
      int64_t max_iter, double prob_mut, double remain_rate,
      const std::vector<double> &lower_bound,
      const std::vector<double> &upper_bound, bool early_stop, int64_t seg_len)
      : GABase<CostFuncTy, BinaryChromo>(cost_func, n_features, size_pop,
            max_iter, prob_mut, lower_bound, upper_bound, early_stop),
        remain_rate_(remain_rate), seg_len_(seg_len) {}

  double GetChromoFit(const BinaryChromo &chromo) override final {

    auto features = Decode(
        chromo, seg_len_, this->lower_bound_, this->upper_bound_);

    auto cost = this->cost_func_(features);
    // mini cost
    auto fitness = -cost;
    return fitness;
  }

  void InitPopulation() override final {
    int64_t n_genes = seg_len_ * this->n_features_;
    for (int i = 0; i < this->size_pop_; ++i) {

      std::string gene;
      for (int j = 0; j < n_genes; ++j) {
        auto rand_01 = GetRandomInt<int64_t>(0, 1);
        gene.push_back(rand_01 == 0 ? '0' : '1');
      }
      BinaryChromo chromo{gene};

      auto fitness = this->GetChromoFit(chromo);
      chromo.fitness = fitness;

      this->population_.push_back(chromo);
    }
  }
  void Selection() override final {
    SelectTournament<BinaryChromo>(this->population_, remain_rate_, 3);
  }

  void CrossOver() override final {
    while (this->population_.size() < this->size_pop_) {
      auto parent_a_idx =
          GetRandomInt<int64_t>(0, this->population_.size() - 1);
      auto parent_b_idx =
          GetRandomInt<int64_t>(0, this->population_.size() - 1);

      auto childs = Crossover2Point(
          this->population_[parent_a_idx], this->population_[parent_b_idx]);
      for (auto &child : childs) {
        child.fitness = this->GetChromoFit(child);
        this->population_.push_back(child);
      }
    }
  }

  void Mutation() override final {
    for (auto &chromo : this->population_) {
      BinaryMutate(chromo, this->prob_mut_);
      chromo.fitness = GetChromoFit(chromo);
    }
  }

  std::vector<double> DecodeChromo(const BinaryChromo &chromo) override final {
    return Decode(chromo, seg_len_, this->lower_bound_, this->upper_bound_);
  }

private:
  int64_t seg_len_; // binary gene number each feature
  double remain_rate_;
};

template <typename CostFuncTy>
class FloatGA : public GABase<CostFuncTy, FloatChromo> {
public:
  FloatGA(CostFuncTy cost_func, int64_t n_features, int64_t size_pop,
      int64_t max_iter, double prob_mut, double remain_rate,
      const std::vector<double> &lower_bound,
      const std::vector<double> &upper_bound, bool early_stop)
      : GABase<CostFuncTy, FloatChromo>(cost_func, n_features, size_pop,
            max_iter, prob_mut, lower_bound, upper_bound, early_stop),
        remain_rate_(remain_rate) {}

  double GetChromoFit(const FloatChromo &chromo) override final {

    auto features =
        cc_opt::Decode(chromo, this->lower_bound_, this->upper_bound_);

    auto cost = this->cost_func_(features);

    // mini cost
    auto fitness = -cost;
    return fitness;
  }

  void InitPopulation() override final {
    int64_t n_genes = this->n_features_;
    for (int64_t i = 0; i < this->size_pop_; ++i) {

      auto gene = GenRandomFloatVec<double>(n_genes, 0, 1);
      FloatChromo chromo{gene};

      auto fitness = this->GetChromoFit(chromo);
      chromo.fitness = fitness;

      this->population_.push_back(chromo);
    }
  }

  void Selection() override final {
    SelectTournament<FloatChromo>(this->population_, remain_rate_, 3);
  }

  void CrossOver() override final {
    while (this->population_.size() < this->size_pop_) {
      auto parent_a_idx =
          GetRandomInt<int64_t>(0, this->population_.size() - 1);
      auto parent_b_idx =
          GetRandomInt<int64_t>(0, this->population_.size() - 1);

      auto childs = Crossover2Point(
          this->population_[parent_a_idx], this->population_[parent_b_idx]);
      for (auto &child : childs) {
        child.fitness = GetChromoFit(child);
        this->population_.push_back(child);
      }
    }
  }

  void Mutation() override final {
    for (auto &chromo : this->population_) {
      FloatMutate(chromo, this->prob_mut_);
      chromo.fitness = this->GetChromoFit(chromo);
    }
  }

  std::vector<double> DecodeChromo(const FloatChromo &chromo) override final{
    return Decode(chromo, this->lower_bound_, this->upper_bound_);
  }

private:
  double remain_rate_;
};

} // namespace cc_opt

#endif /* CCOPT_GA_H */
