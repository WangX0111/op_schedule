#include "src/CostModel/GraphCostNC.hpp"
#include "src/CostModel/Utils.hpp"

#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Types.h"
#include "src/CostModel/Device.hpp"
#include "src/CostModel/Task.hpp"
#include "src/Dialect/ONNX/ONNXOps.hpp"
#include "src/Log/Log.hpp"
#include "src/Utils/Utils.hpp"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/Value.h"

#include "llvm/ADT/DenseSet.h"
#include "llvm/Support/raw_ostream.h"

#include <algorithm>
#include <cassert>
#include <deque>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace onnx_mlir {

std::vector<Operation *> TopoSortByPriority(std::vector<Operation *> ops) {

  // each node indgree
  std::unordered_map<Operation *, int> indegree;
  std::deque<mlir::Operation *> worklist;
  std::unordered_map<mlir::Operation *, bool> visited;
  for (auto op : ops) {
    for (const auto &operand : op->getOperands()) {
      auto operand_op = operand.getDefiningOp();
      if (!operand_op || ShouldIgnoreOperandOp(operand_op)) {
        continue;
      }
      indegree[op]++;
    }

    if (indegree[op] == 0) {
      worklist.push_back(op);
      visited[op] = true;
    }
  }

  auto cmp_op_priority = [](mlir::Operation *lhs, mlir::Operation *rhs) {
    auto lhs_priority_attr = lhs->getAttr("priority");
    auto rhs_priority_attr = rhs->getAttr("priority");
    assert(lhs_priority_attr);
    assert(rhs_priority_attr);

    auto lhs_priority =
        lhs_priority_attr.cast<mlir::FloatAttr>().getValueAsDouble();
    auto rhs_priority =
        rhs_priority_attr.cast<mlir::FloatAttr>().getValueAsDouble();

    return lhs_priority > rhs_priority;
  };

  std::sort(worklist.begin(), worklist.end(), cmp_op_priority);
  // BFS
  std::vector<mlir::Operation *> sorted_ops;
  while (!worklist.empty()) {
    mlir::Operation *front_op = worklist.front();
    worklist.pop_front();
    sorted_ops.push_back(front_op);

    std::vector<mlir::Operation *> ready_ops;
    for (auto user_op : front_op->getUsers()) {
      if (visited[user_op] || isa<ONNXConstantOp>(user_op) ||
          isa<ONNXNoneOp>(user_op) || isa<ONNXReturnOp>(user_op) ||
          isa<func::ReturnOp>(user_op)) {
        continue;
      }
      --indegree[user_op];
      if (indegree[user_op] == 0) {
        ready_ops.push_back(user_op);
        visited[user_op] = true;
      }
    }
    std::sort(ready_ops.begin(), ready_ops.end(), cmp_op_priority);
    for (auto ready_op : ready_ops) {
      worklist.push_back(ready_op);
    }
  }

  return sorted_ops;
}

void GraphCostNC::GenDeviceTaskList(
    std::unordered_map<std::string, std::deque<ComputeTask>> &compute_task,
    std::unordered_map<std::string, std::deque<TransferTask>> &transfer_task,
    llvm::DenseMap<mlir::Value, int> &transfer_cnt,
    const std::unordered_map<std::string, Device> &device_table) {
  // each device memory
  std::unordered_map<std::string, llvm::DenseSet<mlir::Value>> device_memory;

  // topological sort by priority
  auto sorted_ops = TopoSortByPriority(ops_);

  for (auto op : sorted_ops) {

    // ignore op
    // if (isa<ONNXConstantOp>(op) || isa<ONNXNoneOp>(op) ||
    //     isa<ONNXReturnOp>(op) || isa<func::ReturnOp>(op)) {
    //   continue;
    // }

    std::string compute_device_id =
        op->getAttr("device").cast<mlir::StringAttr>().data();

    for (const auto &operand : op->getOperands()) {
      auto produce_op = operand.getDefiningOp();
      if (!produce_op || isa<ONNXConstantOp>(produce_op) ||
          isa<ONNXNoneOp>(produce_op) || isa<ONNXSliceOp>(produce_op)) {
        continue;
      }
      if (!device_memory[compute_device_id].contains(operand)) {
        // add TransferTask
        std::string produce_op_name =
            produce_op->getName().getStringRef().str();
        auto task = TransferTask(
            operand, device_table.at(compute_device_id), device_table);
        transfer_task[compute_device_id].push_back(task);
        device_memory[compute_device_id].insert(operand);
        // add cnt
        transfer_cnt[operand]++;
      }
    }

    for (const auto &result : op->getResults()) {
      device_memory[compute_device_id].insert(result);
    }

    compute_task[compute_device_id].push_back(ComputeTask(op));
  }
}

void AddReadyTasks(std::deque<ComputeTask> &worklist,
    std::unordered_map<std::string, std::deque<ComputeTask>> &device_tasks,
    std::unordered_map<std::string, bool> &device_compute_busy,
    const std::unordered_map<std::string, llvm::DenseSet<mlir::Value>>
        &device_memory) {
  for (auto &kv : device_tasks) {
    // each device can only do one Compute task at a time
    std::string device_id = kv.first;
    if (device_compute_busy[device_id]) {
      continue;
    }
    if (kv.second.empty()) {
      continue;
    }
    auto task = kv.second.front();
    if (task.IsReady(device_memory)) {
      // add to working list
      auto insert_it = std::upper_bound(worklist.begin(), worklist.end(), task);
      if (insert_it == worklist.cend()) {
        worklist.push_back(task);
      } else {
        worklist.insert(insert_it, task);
      }
      device_compute_busy[device_id] = true;
      kv.second.pop_front();
    }
  }
}

void AddReadyTasks(std::deque<TransferTask> &worklist,
    std::unordered_map<std::string, std::deque<TransferTask>> &device_tasks,
    std::unordered_map<std::string, std::unordered_set<std::string>>
        &device_transfer_from_busy,
    const std::unordered_map<std::string, llvm::DenseSet<mlir::Value>>
        &device_memory) {

  for (auto &kv : device_tasks) {
    std::string target_device_id = kv.first;
    auto &tasks = kv.second;
    int num_device_task = tasks.size();
    while (num_device_task--) {
      auto task = tasks.front();
      tasks.pop_front();
      auto from_device_id = task.GetProducedDeviceId();
      // transfer bandwidth from from device_id is busy
      if (SET_CONTAINS(
              device_transfer_from_busy[target_device_id], from_device_id) ||
          !task.IsReady(device_memory)) {
        tasks.push_back(task);
        continue;
      }

      // add Transfer Task to worklist when bandwidth is not busy
      auto insert_it = std::upper_bound(worklist.begin(), worklist.end(), task);
      if (insert_it == worklist.cend()) {
        worklist.push_back(task);
      } else {
        worklist.insert(insert_it, task);
      }
      device_transfer_from_busy[target_device_id].insert(from_device_id);
    }
  }
}

template <typename TaskT>
void ContinueWork(std::deque<TaskT> &worklist, double past_time) {
  for (auto &task : worklist) {
    task.Continue(past_time);
  }
}

double FinishWork(std::deque<ComputeTask> &worklist, double &cur_time,
    std::unordered_map<std::string, llvm::DenseSet<mlir::Value>> &device_memory,
    std::unordered_map<std::string, bool> &device_compute_busy) {

  // finish compute task
  auto finish_task = worklist.front();
  worklist.pop_front();
  finish_task.Finish(cur_time, device_memory);

  // release compute resource
  auto compute_device_id = finish_task.GetComputeOp()
                               ->getAttr("device")
                               .cast<mlir::StringAttr>()
                               .data();
  device_compute_busy[compute_device_id] = false;

  return finish_task.RemainTime();
}

double FinishWork(std::deque<TransferTask> &worklist, double &cur_time,
    std::unordered_map<std::string, llvm::DenseSet<mlir::Value>> &device_memory,
    std::unordered_map<std::string, std::unordered_set<std::string>>
        &device_transfer_from_busy,
    llvm::DenseMap<mlir::Value, int> &transfer_cnt) {

  // finish transfer task
  auto finish_task = worklist.front();
  worklist.pop_front();
  finish_task.Finish(cur_time, device_memory);

  // release bandwidth resource
  auto target_device_id = finish_task.GetTargetDeviceId();
  auto transfer_from_device_id = finish_task.GetProducedDeviceId();
  assert(SET_CONTAINS(
      device_transfer_from_busy[target_device_id], transfer_from_device_id));
  device_transfer_from_busy[target_device_id].erase(transfer_from_device_id);
  --transfer_cnt[finish_task.GetTransferValue()];

  return finish_task.RemainTime();
}

template <typename ValueT>
void InitDeviceTable(const DeviceSetT &device_set,
    std::unordered_map<std::string, ValueT> &device_table) {
  for (const auto &device : device_set) {
    device_table[device.id()] = ValueT();
  }
}

bool SafeRelease(const mlir::Value value, const std::string &device_id,
    const std::unordered_map<std::string, std::deque<ComputeTask>>
        &device_compute_task,
    const std::deque<ComputeTask> &compute_worklist,
    const llvm::DenseMap<mlir::Value, int> &transfer_cnt) {

  // check transfer consumer
  auto produce_op = value.getDefiningOp();
  assert(produce_op);
  auto produce_device_id =
      produce_op->getAttr("device").cast<mlir::StringAttr>().str();
  if (device_id == produce_device_id && transfer_cnt.lookup(value) > 0) {
    return false;
  }

  // check local consumer
  for (const auto &task : compute_worklist) {
    mlir::Operation *op = task.GetComputeOp();
    std::string compute_device_id =
        op->getAttr("device").cast<mlir::StringAttr>().str();
    if (compute_device_id == device_id) {
      for (const auto &operand : op->getOperands()) {
        if (operand == value) {
          return false;
        }
      }
    }
  }
  for (const auto &task : device_compute_task.at(device_id)) {
    mlir::Operation *op = task.GetComputeOp();
    for (const auto &operand : op->getOperands()) {
      if (operand == value) {
        return false;
      }
    }
  }

  return true;
}

void CleanDeviceMemory(
    std::unordered_map<std::string, llvm::DenseSet<mlir::Value>> &device_memory,
    const std::unordered_map<std::string, std::deque<ComputeTask>>
        &device_compute_task,
    const std::deque<ComputeTask> &compute_worklist,
    const llvm::DenseMap<mlir::Value, int> &transfer_cnt) {

  for (auto &kv : device_memory) {
    std::string device_id = kv.first;
    llvm::DenseSet<mlir::Value> release_value;
    for (const auto &value : kv.second) {
      if (SafeRelease(value, device_id, device_compute_task, compute_worklist,
              transfer_cnt)) {
        release_value.insert(value);
      }
    }
    for (const auto &value : release_value) {
      kv.second.erase(value);
    }
  }
}

double GetMemoryUsage(const mlir::Value &value) {
  auto value_type = value.getType().cast<mlir::RankedTensorType>();
  auto n_element = value_type.getNumElements();
  auto byte_per_elem = value_type.getElementTypeBitWidth() / 8;
  return n_element * byte_per_elem;
}

void UpdateMemoryPeak(double &memory_peak,
    const std::unordered_map<std::string, llvm::DenseSet<mlir::Value>>
        &device_memory) {

  double max_device_usage = 0;
  for (const auto &kv : device_memory) {
    double device_usage = 0;
    for (const auto &value : kv.second) {
      device_usage += GetMemoryUsage(value);
    }
    max_device_usage = std::max(device_usage, max_device_usage);
  }

  if (max_device_usage > memory_peak) {
    memory_peak = max_device_usage;
  }
}

Cost GraphCostNC::GetGraphCost() {
  // device table
  std::unordered_map<std::string, Device> device_table;
  for (const auto &device : device_set_) {
    device_table[device.id()] = device;
  }

  // for each device, generate compute and transfer task list
  std::unordered_map<std::string, std::deque<ComputeTask>> device_compute_task;
  InitDeviceTable(device_set_, device_compute_task);

  std::unordered_map<std::string, std::deque<TransferTask>>
      device_transfer_task;
  InitDeviceTable(device_set_, device_transfer_task);

  // each device memory
  std::unordered_map<std::string, llvm::DenseSet<mlir::Value>> device_memory;
  InitDeviceTable(device_set_, device_memory);

  // each device is transfering a value to other device
  std::unordered_map<std::string, std::unordered_set<std::string>>
      device_transfer_from_busy;
  InitDeviceTable(device_set_, device_transfer_from_busy);

  // each device can only run one op at a time
  std::unordered_map<std::string, bool> device_compute_busy;
  InitDeviceTable(device_set_, device_compute_busy);

  // transfer count for each Value need to transfer
  llvm::DenseMap<mlir::Value, int> transfer_cnt;

  GenDeviceTaskList(
      device_compute_task, device_transfer_task, transfer_cnt, device_table);

  // for debug
  // std::cout << "======device_compute_task==========\n";
  // PrintMapContainer(device_compute_task);
  // std::cout << "===================================\n";
  // std::cout << "=======device transfer_task===========\n";
  // PrintMapContainer(device_transfer_task);
  // std::cout << "===================================\n";
  // for debug end

  // Task in working list are running simultaneously
  // keep worklist in order
  std::deque<ComputeTask> compute_worklist;
  std::deque<TransferTask> transfer_worklist;

  double cur_time = 0;
  double memory_peak = 0;

  // init working list

  AddReadyTasks(compute_worklist, device_compute_task, device_compute_busy,
      device_memory);
  AddReadyTasks(transfer_worklist, device_transfer_task,
      device_transfer_from_busy, device_memory);

  // for debug begin
  // std::cout << "======init compute_task===========\n";
  // PrintContainer(compute_worklist);
  // std::cout << "==================================\n";
  // std::cout << "======init transfer_task==========\n";
  // PrintContainer(transfer_worklist);
  // std::cout << "==================================\n";
  // for debug end

  while (!compute_worklist.empty() || !transfer_worklist.empty()) {

    // find min remain time task and Finish
    double past_time = 0;
    if (!compute_worklist.empty() && !transfer_worklist.empty()) {
      if (compute_worklist.front().RemainTime() <
          transfer_worklist.front().RemainTime()) {
        past_time = FinishWork(
            compute_worklist, cur_time, device_memory, device_compute_busy);
      } else {
        past_time = FinishWork(transfer_worklist, cur_time, device_memory,
            device_transfer_from_busy, transfer_cnt);
      }
    } else if (!compute_worklist.empty()) {
      past_time = FinishWork(
          compute_worklist, cur_time, device_memory, device_compute_busy);
    } else {
      past_time = FinishWork(transfer_worklist, cur_time, device_memory,
          device_transfer_from_busy, transfer_cnt);
    }
    ContinueWork(compute_worklist, past_time);
    ContinueWork(transfer_worklist, past_time);

    // Update Memory Peak
    UpdateMemoryPeak(memory_peak, device_memory);
    // release device memory
    CleanDeviceMemory(
        device_memory, device_compute_task, compute_worklist, transfer_cnt);

    // add new ready task to workList
    AddReadyTasks(compute_worklist, device_compute_task, device_compute_busy,
        device_memory);
    AddReadyTasks(transfer_worklist, device_transfer_task,
        device_transfer_from_busy, device_memory);

    // for debug begin
    // std::cout << "***************************************************\n";
    // std::cout << "cur_time: " << cur_time << "\n";
    // std::cout << "====== compute_task ===========\n";
    // PrintContainer(compute_worklist);
    // std::cout << "===============================\n";
    // std::cout << "====== transfer_task===========\n";
    // PrintContainer(transfer_worklist);
    // std::cout << "===============================\n";
    // std::cout << "======= device_memory =========\n";
    // PrintMapContainer(device_memory);
    // std::cout << "===============================\n";
    // std::cout << "======device_compute_task==========\n";
    // PrintMapContainer(device_compute_task);
    // std::cout << "===================================\n";
    // std::cout << "=======device transfer_task===========\n";
    // PrintMapContainer(device_transfer_task);
    // std::cout << "===================================\n";
    // std::cout << "======== Transfer cnt =========\n";
    // PrintMapValue(transfer_cnt);
    // std::cout << "===============================\n";
    // std::cout << "***************************************************\n";
    // for debug end
  }

  // std::cout << "Not Support Op Level CostModel: \n";
  // for (const auto &str : NotSupportCostOp) {
  //   std::cout << str << " ";
  // }
  // std::cout << "\n";

  Cost cost;
  cost.time_cost = cur_time;
  cost.memory_peak = memory_peak;
  return cost;
}
} // namespace onnx_mlir