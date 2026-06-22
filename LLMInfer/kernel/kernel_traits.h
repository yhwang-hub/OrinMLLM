#pragma once

#include <cstdint>
#include <sstream>
#include <string>
#include <vector>

namespace orinmllm::kernel {

enum class DeviceType {
	kCpu = 0,
	kCuda = 1,
};

enum class DataType {
	kUnknown = 0,
	kFloat16 = 1,
	kFloat32 = 2,
	kInt8 = 3,
	kFp8 = 4,
};

struct KernelTraits {
	std::string op_name;
	DeviceType device = DeviceType::kCpu;
	DataType dtype = DataType::kUnknown;
	int32_t sm = 0;
	int32_t head_dim = 0;
	int32_t tile_m = 0;
	int32_t tile_n = 0;
	bool is_fused = false;

	std::string signature() const {
		std::ostringstream oss;
		oss << op_name << "|dev=" << static_cast<int32_t>(device)
				<< "|dtype=" << static_cast<int32_t>(dtype) << "|sm=" << sm
				<< "|head=" << head_dim << "|m=" << tile_m << "|n=" << tile_n
				<< "|fused=" << (is_fused ? 1 : 0);
		return oss.str();
	}
};

inline std::string DeviceTypeToString(const DeviceType device) {
	switch (device) {
		case DeviceType::kCpu:
			return "cpu";
		case DeviceType::kCuda:
			return "cuda";
		default:
			return "unknown";
	}
}

inline std::string DataTypeToString(const DataType dtype) {
	switch (dtype) {
		case DataType::kFloat16:
			return "fp16";
		case DataType::kFloat32:
			return "fp32";
		case DataType::kInt8:
			return "int8";
		case DataType::kFp8:
			return "fp8";
		default:
			return "unknown";
	}
}

inline std::vector<int32_t> DefaultCudaSms() { return {86, 87, 89, 90, 120}; }

}  // namespace orinmllm::kernel
