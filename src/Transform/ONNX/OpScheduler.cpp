#include "src/Transform/ONNX/OpScheduler.hpp"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/Operation.h"
#include "src/CostModel/Device.hpp"
#include "src/CostModel/GraphCostNC.hpp"

#include "src/Dialect/ONNX/ONNXOps.hpp"
#include "src/Utils/Utils.hpp"

#include "llvm/Support/raw_ostream.h"

#define PARALLEL
#include "third_party/cc_opt/ga/brkga.h"
#undef PARALLEL
#include "third_party/cc_opt/ga/ga.h"
#include "third_party/cc_opt/pso/pso.h"

#include <algorithm>
#include <cassert>
#include <cstddef>
#include <cstdint>
#include <iterator>
#include <random>
#include <unordered_map>
#include <vector>

namespace onnx_mlir {

std::vector<double> GenRandomVec(int64_t size) {

  // engine
  std::random_device seed;
  std::ranlux48 rand_engine(seed());
  // distribution
  std::uniform_real_distribution<double> dist(0, 1);

  auto gen = [&dist, &rand_engine]() { return dist(rand_engine); };

  std::vector<double> vec(size);
  std::generate(vec.begin(), vec.end(), gen);

  return vec;
}

template <typename ArrayTy>
size_t ArgMaxRange(const ArrayTy &array, int64_t first, int64_t last) {
  // [first, last]
  int64_t n_array = array.size();
  assert(first >= 0 && first < n_array);
  assert(last >= 0 && last < n_array);

  int max_idx = first;
  for (int64_t idx = first; idx <= last; ++idx) {
    if (array.at(idx) > array.at(max_idx)) {
      max_idx = idx;
    }
  }

  return max_idx;
}

void SetDefaultPlacementAndPriority(mlir::OpBuilder &builder,
    std::vector<Operation *> &ops, const DeviceSetT &device_set) {
  auto default_device = *device_set.begin();
  auto default_priority = 0;

  for (auto &op : ops) {

    op->setAttr(DEVICE_ATTR, builder.getStringAttr(default_device.id()));
    op->setAttr(PRIORITY_ATTR, builder.getF32FloatAttr(default_priority));
  }
}

void HackSchedule(mlir::OpBuilder &builder, std::vector<Operation *> &ops,
    const DeviceSetT &device_set) {
  for (auto &op : ops) {
    if (isa<ONNXNoneOp>(op) || isa<ONNXConstantOp>(op) ||
        isa<ONNXReturnOp>(op) || isa<func::ReturnOp>(op)) {
      continue;
    }
    std::string node_name =
        op->getAttr("onnx_node_name").cast<mlir::StringAttr>().str();
    if (node_name == "c1") {
      op->setAttr(DEVICE_ATTR, builder.getStringAttr("dtu0"));
    } else if (node_name == "c2") {
      op->setAttr(DEVICE_ATTR, builder.getStringAttr("dtu0"));
    } else if (node_name == "c3") {
      op->setAttr(DEVICE_ATTR, builder.getStringAttr("dtu1"));

    } else if (node_name == "c4") {
      op->setAttr(DEVICE_ATTR, builder.getStringAttr("dtu0"));

    } else if (node_name == "c5") {
      op->setAttr(DEVICE_ATTR, builder.getStringAttr("dtu2"));
    }
  }
}

void RandomSchedule(mlir::OpBuilder &builder, std::vector<Operation *> &ops,
    const DeviceSetT &device_set) {

  auto n_device = device_set.size();

  std::random_device seed;
  std::ranlux48 rand_engine(seed());

  std::uniform_real_distribution<float> priority_distrib(0, 1);
  std::uniform_int_distribution<int> device_id_distrib(0, n_device - 1);

  for (auto &op : ops) {

    // random choose from device set
    int rand_device_index = device_id_distrib(rand_engine);
    auto it = device_set.begin();
    std::advance(it, rand_device_index);
    auto device = *it;
    auto priority = priority_distrib(rand_engine);

    op->setAttr(DEVICE_ATTR, builder.getStringAttr(device.id()));
    op->setAttr(PRIORITY_ATTR, builder.getF32FloatAttr(priority));
  }
}

void SingleDTUSchedule(mlir::OpBuilder &builder, std::vector<Operation *> &ops,
    const DeviceSetT &device_set) {

  Device dtu;
  for (const auto &device : device_set) {
    if (device.type() == DeviceType::DTU) {
      dtu = device;
      break;
    }
  }

  double n_ops = ops.size();
  double priority = n_ops;
  for (auto &op : ops) {
    op->setAttr(DEVICE_ATTR, builder.getStringAttr(dtu.id()));
    op->setAttr(PRIORITY_ATTR, builder.getF32FloatAttr(priority));
    priority -= 1 / n_ops;
  }
}

void SingleCPUSchedule(mlir::OpBuilder &builder, std::vector<Operation *> &ops,
    const DeviceSetT &device_set) {

  Device cpu;
  for (const auto &device : device_set) {
    if (device.type() == DeviceType::CPU) {
      cpu = device;
      break;
    }
  }

  double n_ops = ops.size();
  double priority = n_ops;
  for (auto &op : ops) {
    op->setAttr(DEVICE_ATTR, builder.getStringAttr(cpu.id()));
    op->setAttr(PRIORITY_ATTR, builder.getF32FloatAttr(priority));
    priority -= 1 / n_ops;
  }
}

void GetOpScheduleAttr(const std::vector<mlir::Operation *> &ops,
    std::unordered_map<mlir::Operation *, std::string> &op_device_table,
    std::unordered_map<mlir::Operation *, double> &op_priority_table) {
  for (auto op : ops) {
    op_device_table[op] = op->getAttr("device").cast<mlir::StringAttr>().str();
    op_priority_table[op] = op->getAttr("priority")
                                .cast<mlir::FloatAttr>()
                                .getValue()
                                .convertToDouble();
  }
}

void RandomSearchSchedule(mlir::OpBuilder &builder,
    std::vector<Operation *> &ops, const DeviceSetT &device_set,
    int32_t iter_num) {

  GraphCostNC graph_cost(ops, device_set);
  std::vector<Operation *> best_schedule_ops;

  std::unordered_map<mlir::Operation *, std::string> op_device_table;
  std::unordered_map<mlir::Operation *, double> op_priority_table;
  GetOpScheduleAttr(ops, op_device_table, op_priority_table);

  double min_latency = graph_cost.GetGraphCost().time_cost;
  for (int32_t i = 0; i < iter_num; ++i) {
    RandomSchedule(builder, ops, device_set);
    graph_cost.UpdateOps(ops);
    double latency = graph_cost.GetGraphCost().time_cost;
    if (latency < min_latency) {
      GetOpScheduleAttr(ops, op_device_table, op_priority_table);
      min_latency = latency;
    }
    std::cout << "iter num: " << i << " latency: " << latency
              << " min_latency: " << min_latency << "\n";
  }
  for (auto op : ops) {
    op->setAttr(DEVICE_ATTR, builder.getStringAttr(op_device_table[op]));
    op->setAttr(PRIORITY_ATTR, builder.getF32FloatAttr(op_priority_table[op]));
  }
}

void SetScheduleAttrByFeatures(mlir::OpBuilder &builder,
    std::vector<mlir::Operation *> &ops, const std::vector<double> &features,
    const std::vector<Device> &device_list) {
  int64_t n_device = device_list.size();
  int64_t n_ops = ops.size();

  for (int64_t op_idx = 0; op_idx < n_ops; ++op_idx) {
    int64_t device_affine_range_begin = op_idx * n_device;
    int64_t max_affine_device_idx = ArgMaxRange(features,
        device_affine_range_begin, device_affine_range_begin + n_device - 1);
    Device device =
        device_list[max_affine_device_idx - device_affine_range_begin];
    mlir::Operation *op = ops[op_idx];
    op->setAttr(DEVICE_ATTR, builder.getStringAttr(device.id()));

    int64_t op_priority_idx = n_ops * n_device + op_idx;
    auto priority = features[op_priority_idx];
    op->setAttr(PRIORITY_ATTR, builder.getF32FloatAttr(priority));
  }
}

void GASchedule(mlir::OpBuilder &builder, std::vector<Operation *> &ops,
    const DeviceSetT &device_set, int32_t iter_num) {
  std::cout << "Schedule Strategy : GASchedule\n";

  std::vector<Device> device_list;
  for (const auto &device : device_set) {
    device_list.push_back(device);
  }

  GraphCostNC graph_cost(ops, device_set);
  auto cost_func = [&](const std::vector<double> &features) {
    SetScheduleAttrByFeatures(builder, ops, features, device_list);
    graph_cost.UpdateOps(ops);
    auto latency = graph_cost.GetGraphCost().time_cost;
    return latency;
  };

  int64_t n_ops = ops.size();
  int64_t n_device = device_list.size();

  cc_opt::ga::FloatGAParam param;
  param.n_features = n_ops * n_device + n_ops;
  param.size_pop = 50;
  param.max_iter = iter_num;
  param.prob_mut = 0.01;
  param.remain_rate = 0.5;
  param.lower_bound = std::vector<double>(param.n_features, 0.0);
  param.upper_bound = std::vector<double>(param.n_features, 1.0);
  param.early_stop = true;
  cc_opt::ga::FloatGA<decltype(cost_func)> float_ga(cost_func, param);

  float_ga.Run();

  auto best_feature = float_ga.GetBestFeature();
  SetScheduleAttrByFeatures(builder, ops, best_feature, device_list);

  std::cout << "best :" << float_ga.GetBestCost() << "\n";
}

void BRKGASchedule(mlir::OpBuilder &builder, std::vector<Operation *> &ops,
    const DeviceSetT &device_set, int32_t iter_num) {
  std::cout << "Schedule Strategy : BRKGASchedule\n";

  std::vector<Device> device_list;
  for (const auto &device : device_set) {
    device_list.push_back(device);
  }

  GraphCostNC graph_cost(ops, device_set);
  auto cost_func = [&](const std::vector<double> &features) {
    SetScheduleAttrByFeatures(builder, ops, features, device_list);
    graph_cost.UpdateOps(ops);
    auto latency = graph_cost.GetGraphCost().time_cost;
    return latency;
  };

  int64_t n_ops = ops.size();
  int64_t n_device = device_list.size();

  cc_opt::ga::BRKGAParam param;
  param.n_features = n_ops * n_device + n_ops;
  param.size_pop = param.n_features;
  param.elite_rate = 0.20;
  param.mutant_rate = 0.20;
  param.inherit_elite_prob = 0.70;

  param.max_iter = iter_num;
  param.lower_bound = std::vector<double>(param.n_features, 0);
  param.upper_bound = std::vector<double>(param.n_features, 1.0);

  cc_opt::ga::BRKGA<decltype(cost_func)> brkga(cost_func, param);

  brkga.Run();

  auto best_feature = brkga.GetBestFeature();
  SetScheduleAttrByFeatures(builder, ops, best_feature, device_list);

  std::cout << "best :" << brkga.GetBestCost() << "\n";
}

void PSOSchedule(mlir::OpBuilder &builder, std::vector<Operation *> &ops,
    const DeviceSetT &device_set, int32_t iter_num) {

  std::cout << "Schedule Strategy : PSOSchedule\n";
  std::vector<Device> device_list;
  for (const auto &device : device_set) {
    device_list.push_back(device);
  }

  GraphCostNC graph_cost(ops, device_set);
  auto cost_func = [&](const std::vector<double> &features) {
    SetScheduleAttrByFeatures(builder, ops, features, device_list);
    graph_cost.UpdateOps(ops);
    auto latency = graph_cost.GetGraphCost().time_cost;
    return latency;
  };

  int64_t n_ops = ops.size();
  int64_t n_device = device_list.size();

  int64_t n_features = n_ops * n_device + n_ops;

  cc_opt::pso::PSOParam param;
  param.n_features = n_features;
  param.size_pop = 500;
  param.max_iter = iter_num;
  param.person_learning_factor = 2;
  param.group_learning_factor = 2;
  param.inertia_weight = 0.8;
  param.time_step = 1;
  param.lower_bound = std::vector<double>(n_features, 0);
  param.upper_bound = std::vector<double>(n_features, 1.0);
  param.early_stop = true;

  cc_opt::pso::PSO<decltype(cost_func)> pso(cost_func, param);
  pso.Run();

  auto best_feature = pso.GetBestFeature();
  SetScheduleAttrByFeatures(builder, ops, best_feature, device_list);

  std::cout << "best :" << pso.GetBestCost() << "\n";
}

} // namespace onnx_mlir