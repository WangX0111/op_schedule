#ifndef ONNX_MLIR_DEVICE_HPP
#define ONNX_MLIR_DEVICE_HPP

#include "llvm/ADT/SmallSet.h"
#include <string>
#include <unordered_map>

namespace onnx_mlir {
enum class DeviceType { CPU = 0, DTU = 1, GPU = 2, UNDEFINED };

class Device {
public:
  explicit Device() : type_(DeviceType::UNDEFINED), id_("") {}
  explicit Device(DeviceType type) : type_(type), id_("") {}
  explicit Device(DeviceType type, std::string id) : type_(type), id_(id) {}

  std::string id() const { return id_; }
  DeviceType type() const { return type_; }

  bool operator==(const Device &device) {
    return type_ == device.type_ && id_ == device.id_;
  }

  bool operator!=(const Device &device) { return !(*this == device); }

private:
  DeviceType type_ = DeviceType::UNDEFINED;
  std::string id_ = "";

  friend bool operator<(const Device &lhs, const Device &rhs) {
    return lhs.id_ < rhs.id_;
  }
  friend bool operator==(const Device &lhs, const Device &rhs) {
    return lhs.type_ == rhs.type_ && lhs.id_ == rhs.id_;
  }
};

struct DeviceInfo {

  DeviceInfo()
      : compute_power_1d(0), compute_power_2d(0),
        bandwidth(__builtin_inff32()) {}

  DeviceInfo(double compute_power_id, double compute_power_2d, double bandwidth)
      : compute_power_1d(compute_power_id), compute_power_2d(compute_power_2d),
        bandwidth(bandwidth) {}

  // MFLOPS / ns
  double compute_power_1d;
  double compute_power_2d;


  // MB / ns
  double bandwidth;
};

using DeviceSetT = llvm::SmallSet<Device, 5>;

DeviceSetT GetDeviceSet();

std::unordered_map<std::string, Device> GetDeviceTable();

DeviceInfo GetDeviceInfo(const Device &device);

} // namespace onnx_mlir

#endif /* ONNX_MLIR_DEVICE_HPP */
