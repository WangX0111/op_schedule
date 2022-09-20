#include "src/CostModel/Device.hpp"
#include <string>
#include <unordered_map>

namespace onnx_mlir {

DeviceSetT GetDeviceSet() {
  // hack code

  // one cpu and 4C
  DeviceSetT device_set;
  device_set.insert(Device(DeviceType::CPU, "cpu0"));
  device_set.insert(Device(DeviceType::DTU, "dtu0"));
  device_set.insert(Device(DeviceType::DTU, "dtu1"));
  device_set.insert(Device(DeviceType::DTU, "dtu2"));
  device_set.insert(Device(DeviceType::DTU, "dtu3"));

  return device_set;
}

std::unordered_map<std::string, Device> GetDeviceTable() {
  DeviceSetT device_set = GetDeviceSet();
  std::unordered_map<std::string, Device> device_table;
  for (const auto &device : device_set) {
    device_table[device.id()] = device;
  }
  return device_table;
};

// http://wiki.enflame.cn/pages/viewpage.action?pageId=70331886
// http://blog.sysu.tech/Benchmark/%E5%A6%82%E4%BD%95%E8%AE%A1%E7%AE%97CPU%E7%AE%97%E5%8A%9B%E7%90%86%E8%AE%BA%E5%B3%B0%E5%80%BC/
DeviceInfo GetDeviceInfo(const Device &device) {
  auto device_type = device.type();
  DeviceInfo device_info;
  if (device_type == DeviceType::CPU) {
    // Intel Xeon Gold 6150
    // 1.5 TFLOPS/s
    // 64 GB/s = 0.064 MB/s PCIE
    
    // MFLOP/ns
    device_info.compute_power_1d = 1.5;
    device_info.compute_power_2d = 1.5;

    // MB/ns
    device_info.bandwidth = 0.064;

  } else if (device_type == DeviceType::GPU) {
    // whole NV A10
    // 31.2 TFLOP/s = 31.2 MFLOP/ns
    // 600 GB/s = 0.6 MB/s

    // MFLOP/ns
    device_info.compute_power_1d = 31.2;
    device_info.compute_power_2d = 31.2;
    // MB/ns
    device_info.bandwidth = 0.600;

  } else if (device_type == DeviceType::DTU) {
    // Dorado 1C
    // 1D: 0.9375 TFLOP/s = 0.9375 MFLOP/ns
    // 2D: 6 * 1.25 * 512 * 2 / 1000 = 7.68 TFLOP/s = 7.68 MFLOP/ns
    // 512 GB/s = 0.512 MB/ns

    // MFLOP/ns
    device_info.compute_power_1d = 0.9375;
    device_info.compute_power_2d = 7.68;

    // MB/ns
    device_info.bandwidth = 0.512;
  }

  return device_info;
}

} // namespace onnx_mlir
