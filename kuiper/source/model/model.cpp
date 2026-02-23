#include "model/model.h"
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
namespace model {
Model::Model(base::TokenizerType tokenizer_type, base::ModelType model_type, std::string token_path,
             std::string model_path, bool is_quant_model)
    : tokenizer_type_(tokenizer_type),
      model_type_(model_type),
      token_path_(std::move(token_path)),
      model_path_(std::move(model_path)),
      is_quant_model_(is_quant_model) {}

base::ModelType Model::model_type() const { return model_type_; }

const std::string& Model::token_path() const { return token_path_; }

const std::string& Model::model_path() const { return model_path_; }

base::Status Model::insert_buffer(ModelBufferType buffer_idx, const tensor::Tensor& tensor) {
  if (buffers_.count(buffer_idx) > 0) {
    return base::error::KeyHasExits(std::to_string(int(buffer_idx)) + " has exits in the buffers");
  }
  if (tensor.is_empty()) {
    return base::error::InvalidArgument("The tensor is empty for inserting buffer.");
  }
  buffers_.insert({buffer_idx, tensor});
  return base::error::Success();
}

tensor::Tensor& Model::get_buffer(ModelBufferType buffer_idx) {
  CHECK_GT(buffers_.count(buffer_idx), 0) << int(buffer_idx);
  return buffers_.at(buffer_idx);
}

const tensor::Tensor& Model::get_buffer(ModelBufferType buffer_idx) const {
  CHECK_GT(buffers_.count(buffer_idx), 0);
  return buffers_.at(buffer_idx);
}

base::Status Model::read_model_file() {
  using namespace base;
  if (model_path_.empty()) {
    return error::PathNotValid("Failed to open the weight file, the model path is empty!");
  }
  int32_t fd = open(model_path_.data(), O_RDONLY);
  if (fd == -1) {
    return error::PathNotValid("Failed to open the weight file " + model_path_ +
                               " may be the path does not exist!");
  }

  FILE* file = fopen(model_path_.data(), "rb");
  if (!file) {
    return error::PathNotValid("Failed to open the file. The path may be invalid.");
  }

  // Check for FP16 model format (version 3 with magic header)
  // FP16 format: magic(4) + version(4) + config(28) + shared_classifier(1) + padding
  uint32_t magic = 0;
  int32_t version = 0;
  
  if (fread(&magic, sizeof(uint32_t), 1, file) != 1) {
    return error::ModelParseError("Failed to read magic number from model file.");
  }
  
  // Check if it's the new versioned format
  // magic = 0x616b3432 = "ak42" for Qwen2.5
  // magic = 0x616b3437 = "ak47" for Qwen3
  // magic = 0x616b3438 = "ak48" for Qwen3 AWQ
  if (magic == 0x616b3432 || magic == 0x616b3437 || magic == 0x616b3438) {
    bool is_qwen3_format = (magic == 0x616b3437 || magic == 0x616b3438);
    bool is_awq_format = (magic == 0x616b3438);
    
    if (fread(&version, sizeof(int32_t), 1, file) != 1) {
      return error::ModelParseError("Failed to read version from model file.");
    }
    
    LOG(INFO) << "Model file magic: 0x" << std::hex << magic << std::dec;
    LOG(INFO) << "Model file version: " << version;
    
    if (version == 3) {
      // FP16 format for Qwen2.5
      is_fp16_model_ = true;
      LOG(INFO) << "Loading FP16 model format (Qwen2.5)";
    } else if (version == 4) {
      // FP16 format for Qwen3
      is_fp16_model_ = true;
      LOG(INFO) << "Loading FP16 model format (Qwen3)";
    } else if (version == 5) {
      // AWQ INT4 format for Qwen3
      is_awq_model_ = true;
      is_fp16_model_ = true;  // AWQ uses FP16 for non-quantized weights
      LOG(INFO) << "Loading AWQ INT4 model format (Qwen3)";
    } else if (version == 1) {
      // FP32 format with header
      is_fp16_model_ = false;
      LOG(INFO) << "Loading FP32 model format (version 1)";
    } else if (version == 2) {
      // INT8 quantized format
      is_quant_model_ = true;
      LOG(INFO) << "Loading INT8 quantized model format";
    }
    
    // Read config field by field to avoid structure size mismatch
    // File format: dim(4) + hidden_dim(4) + n_layers(4) + n_heads(4) + n_kv_heads(4) + vocab_size(4) + max_seq_len(4) + shared_classifier(1) + [head_dim(4) for Qwen3]
    auto config = ModelConfig{};
    
    if (fread(&config.dim, sizeof(int32_t), 1, file) != 1 ||
        fread(&config.hidden_dim, sizeof(int32_t), 1, file) != 1 ||
        fread(&config.layer_num, sizeof(int32_t), 1, file) != 1 ||
        fread(&config.head_num, sizeof(int32_t), 1, file) != 1 ||
        fread(&config.kv_head_num, sizeof(int32_t), 1, file) != 1 ||
        fread(&config.vocab_size, sizeof(int32_t), 1, file) != 1 ||
        fread(&config.seq_len, sizeof(int32_t), 1, file) != 1) {
      return error::ModelParseError("Failed to read config from versioned model file.");
    }
    
    // Read shared_classifier flag
    uint8_t shared_classifier = 0;
    if (fread(&shared_classifier, sizeof(uint8_t), 1, file) != 1) {
      return error::ModelParseError("Failed to read shared_classifier flag.");
    }
    
    // For Qwen3, read head_dim
    int32_t head_dim = 0;
    if (is_qwen3_format) {
      if (fread(&head_dim, sizeof(int32_t), 1, file) != 1) {
        return error::ModelParseError("Failed to read head_dim from Qwen3 model file.");
      }
      LOG(INFO) << "Qwen3 head_dim: " << head_dim;
#ifdef QWEN3_SUPPORT
      // For Qwen3, hidden_dim in file is actually intermediate_size
      config.immediate_dim_ = config.hidden_dim;
#endif

      // For AWQ format, read group_size
      if (is_awq_format) {
        if (fread(&group_size_, sizeof(int32_t), 1, file) != 1) {
          return error::ModelParseError("Failed to read group_size from AWQ model file.");
        }
        LOG(INFO) << "AWQ group_size: " << group_size_;
      }
    }
    
    LOG(INFO) << "Config: dim=" << config.dim << ", hidden_dim=" << config.hidden_dim
              << ", layer_num=" << config.layer_num << ", head_num=" << config.head_num
              << ", kv_head_num=" << config.kv_head_num << ", vocab_size=" << config.vocab_size
              << ", seq_len=" << config.seq_len;
    
    auto gen_status = generate_model_infos(config);
    if (!gen_status) {
      return gen_status;
    }
    
    // Override shared weight flag from file
    config_->is_shared_weight_ = (shared_classifier != 0);
    LOG(INFO) << "is_shared_weight_: " << config_->is_shared_weight_;
    
  } else {
    // Legacy format: config starts at offset 0
    // Seek back to beginning
    fseek(file, 0, SEEK_SET);
    
    auto config = ModelConfig{};
    if (fread(&config, sizeof(ModelConfig), 1, file) != 1) {
      return error::ModelParseError(
          "Failed to retrieve the configuration information from the model "
          "file.");
    }
    if (is_quant_model_) {
      if (fread(&group_size_, sizeof(int32_t), 1, file) != 1) {
        return error::ModelParseError(
            "Failed to retrieve the group size information from the model "
            "file.");
      }
    }

    auto gen_status = generate_model_infos(config);
    if (!gen_status) {
      return gen_status;
    }
  }

  // Create appropriate raw model data handler
  if (is_fp16_model_) {
    raw_model_data_ = std::make_shared<RawModelDataFp16>();
  } else if (!is_quant_model_) {
    raw_model_data_ = std::make_shared<RawModelDataFp32>();
  } else {
    raw_model_data_ = std::make_shared<RawModelDataInt8>();
  }

  struct stat sb;
  if (fstat(fd, &sb) == -1) {
    close(fd);
    return error::ModelParseError(
        "Failed to retrieve the file size information from the model "
        "file.");
  }
  raw_model_data_->file_size = sb.st_size;

  raw_model_data_->fd = fd;
  raw_model_data_->data =
      mmap(nullptr, raw_model_data_->file_size, PROT_READ, MAP_PRIVATE, raw_model_data_->fd, 0);

  if (raw_model_data_->data == MAP_FAILED || raw_model_data_->data == nullptr) {
    return error::ModelParseError("Failed to map the weight file " + model_path_ + " into memory.");
  }
  
  // Set weight_data pointer based on format
  // Legacy format header size: 7 x int32_t = 28 bytes (without QWEN3_SUPPORT field)
  constexpr size_t kLegacyHeaderSize = 7 * sizeof(int32_t);  // 28 bytes
  
  if (magic == 0x616b3432 || magic == 0x616b3437 || magic == 0x616b3438) {
    // Versioned format: 256 byte header (for ak42, ak47, and ak48/AWQ)
    raw_model_data_->weight_data =
        static_cast<int8_t*>(raw_model_data_->data) + 256;
  } else if (!is_quant_model_) {
    // Legacy FP32 format: header is always 28 bytes (7 int32_t fields)
    // Do NOT use sizeof(ModelConfig) as it may include QWEN3_SUPPORT field
    raw_model_data_->weight_data =
        static_cast<int8_t*>(raw_model_data_->data) + kLegacyHeaderSize;
  } else {
    // Legacy INT8 quantized format: header + group_size
    raw_model_data_->weight_data =
        static_cast<int8_t*>(raw_model_data_->data) + kLegacyHeaderSize + sizeof(group_size_);
  }
  
  if (raw_model_data_ == nullptr) {
    LOG(ERROR);
    return error::ModelParseError("Failed to map the weight file " + model_path_ +
                                  " into memory, the pointer to weight start address is null");
  }
  return error::Success();
}

base::Status Model::generate_model_infos(const ModelConfig& config) const {
  config_->dim_ = config.dim;
  config_->hidden_dim_ = config.hidden_dim;
  config_->layer_num_ = config.layer_num;
  config_->head_num_ = config.head_num;
  config_->kv_head_num_ = config.kv_head_num;
  // Store original seq_len for weight offset calculations
  // Limit runtime seq_len to reduce memory usage on Orin (unified memory)
  config_->original_seq_len_ = config.seq_len;  // Keep original for weight loading
  config_->seq_len_ = std::min(config.seq_len, 8192);  // Increased from 4096 to support longer conversations
  LOG(INFO) << "Using seq_len: " << config_->seq_len_ << " (original: " << config.seq_len << ")";

  config_->kv_dim_ = (config.dim * config.kv_head_num) / config.head_num;
  config_->kv_mul_ = config.head_num / config.kv_head_num;
  config_->head_size_ = config.dim / config.head_num;
#if defined(QWEN3_SUPPORT)
  config_->immediate_dim_ = config.immediate_dim_;
  LOG(INFO) << "generate_model_infos: dim_=" << config_->dim_ 
            << ", hidden_dim_=" << config_->hidden_dim_
            << ", immediate_dim_=" << config_->immediate_dim_
            << ", kv_dim_=" << config_->kv_dim_
            << ", head_size_=" << config_->head_size_;
#endif
  if (config.vocab_size > 0) {
    config_->is_shared_weight_ = true;
  } else {
    config_->is_shared_weight_ = false;
  }

  // Qwen tokenizer size and embedding size is mismatched
  // refer: https://github.com/QwenLM/Qwen2.5/issues/29
  // if (std::abs(config.vocab_size) != config_->vocab_size_) {
  //   return base::error::ModelParseError(
  //       "Vocabulary size mismatch between the model file and the token list.");
  // }
  config_->vocab_size_ = std::abs(config.vocab_size);
  return base::error::Success();
}

base::Status Model::create_encode_layer() {
  using namespace base;

  // create token encode decode layer
  if (tokenizer_type_ == TokenizerType::kEncodeSpe) {
    encode_layer_ = std::make_unique<op::SpeEncodeLayer>(this->token_path_, true, false);
  } else {
#ifdef LLAMA3_SUPPORT
    encode_layer_ = std::make_unique<op::BpeEncodeLayer>(this->token_path_, true, false);
#endif

#if defined(QWEN2_SUPPORT) || defined(QWEN3_SUPPORT)
    encode_layer_ = std::make_unique<op::QwenEncodeLayer>(this->token_path_, false, false);
#endif
  }
  if (!encode_layer_) {
    return error::InternalError("Create the encode layer failed.");
  }

  config_->vocab_size_ = encode_layer_->vocab_size();
  if (config_->vocab_size_ <= 0) {
    return error::InternalError("The vocab size param read error from the model file!");
  }
  return error::Success();
}

base::Status Model::gen_model_from_file() {
  using namespace base;
  config_ = std::make_unique<TransformerConfig>();

  // init sentence piece processor
  // google sentence piece
  auto create_encode_status = create_encode_layer();
  if (!create_encode_status) {
    LOG(ERROR) << "Create the encode layer failed!";
    return create_encode_status;
  }
  // mmap
  auto mmap_status = read_model_file();
  if (!mmap_status) {
    LOG(ERROR) << "Handle model file " << model_path_ << " failed!";
    return mmap_status;
  }
  auto layer_create_status = create_layers();
  if (!layer_create_status) {
    LOG(ERROR) << "Create layers for the model file " << model_path_ << " failed!";
    return layer_create_status;
  }

  return error::Success();
}

std::vector<int32_t> Model::encode(const std::string& sentence) const {
  CHECK(encode_layer_ != nullptr);
  return encode_layer_->encode(sentence);
}

bool Model::is_sentence_ending(int32_t token_idx) const {
  CHECK(this->encode_layer_ != nullptr);
  return this->encode_layer_->is_sentence_ending(token_idx);
}

std::string Model::decode(int32_t token_idx) const {
  CHECK(this->encode_layer_ != nullptr);
  return this->encode_layer_->decode(token_idx);
}

std::string Model::decode(std::vector<int32_t> token_idxs) const {
  CHECK(this->encode_layer_ != nullptr);
  return this->encode_layer_->decode(token_idxs);
}

std::pair<tensor::Tensor, tensor::Tensor> Model::slice_kv_cache(int32_t layer_idx,
                                                                int32_t token_pos) const {
  // ===================== Paged KV Cache Path =====================
  if (use_paged_attention_ && paged_kv_cache_manager_) {
    auto* mgr = paged_kv_cache_manager_.get();
    size_t byte_offset = mgr->get_kv_byte_offset(layer_idx, token_pos);

    if (mgr->dtype() == base::DataType::kDataTypeFp16) {
      uint16_t* key_ptr = reinterpret_cast<uint16_t*>(
          static_cast<char*>(mgr->key_pool_gpu()) + byte_offset);
      uint16_t* val_ptr = reinterpret_cast<uint16_t*>(
          static_cast<char*>(mgr->value_pool_gpu()) + byte_offset);

      tensor::Tensor key(base::DataType::kDataTypeFp16, config_->kv_dim_, false, nullptr, key_ptr);
      tensor::Tensor val(base::DataType::kDataTypeFp16, config_->kv_dim_, false, nullptr, val_ptr);
      key.set_device_type(device_type_);
      val.set_device_type(device_type_);
      return {key, val};
    } else {
      float* key_ptr = reinterpret_cast<float*>(
          static_cast<char*>(mgr->key_pool_gpu()) + byte_offset);
      float* val_ptr = reinterpret_cast<float*>(
          static_cast<char*>(mgr->value_pool_gpu()) + byte_offset);

      tensor::Tensor key(base::DataType::kDataTypeFp32, config_->kv_dim_, false, nullptr, key_ptr);
      tensor::Tensor val(base::DataType::kDataTypeFp32, config_->kv_dim_, false, nullptr, val_ptr);
      key.set_device_type(device_type_);
      val.set_device_type(device_type_);
      return {key, val};
    }
  }

  // ===================== Contiguous KV Cache Path =====================
  int32_t layer_offset = layer_idx * config_->seq_len_ * config_->kv_dim_;
  int32_t cache_offset = layer_offset + token_pos * config_->kv_dim_;

  const auto& key_cache_buffer = get_buffer(ModelBufferType::kKeyCache);
  const auto& val_cache_buffer = get_buffer(ModelBufferType::kValueCache);
  
  // Check if using FP16 KV cache
  if (key_cache_buffer.data_type() == base::DataType::kDataTypeFp16) {
    uint16_t* key_cache_ptr =
        const_cast<uint16_t*>(key_cache_buffer.ptr<uint16_t>(cache_offset));
    uint16_t* val_cache_ptr =
        const_cast<uint16_t*>(val_cache_buffer.ptr<uint16_t>(cache_offset));

    tensor::Tensor key(base::DataType::kDataTypeFp16, config_->kv_dim_, false, nullptr,
                       key_cache_ptr);
    tensor::Tensor val(base::DataType::kDataTypeFp16, config_->kv_dim_, false, nullptr,
                       val_cache_ptr);
    key.set_device_type(device_type_);
    val.set_device_type(device_type_);
    return {key, val};
  }
  
  // FP32 path
  float* key_cache_ptr =
      const_cast<float*>(key_cache_buffer.ptr<float>(cache_offset));
  float* val_cache_ptr =
      const_cast<float*>(val_cache_buffer.ptr<float>(cache_offset));

  tensor::Tensor key(base::DataType::kDataTypeFp32, config_->kv_dim_, false, nullptr,
                     key_cache_ptr);
  tensor::Tensor val(base::DataType::kDataTypeFp32, config_->kv_dim_, false, nullptr,
                     val_cache_ptr);
  key.set_device_type(device_type_);
  val.set_device_type(device_type_);
  return {key, val};
}

tensor::Tensor Model::fill_input(const tensor::Tensor& pos_tensor,
                                 const op::EmbeddingOutput& embedding_output,
                                 bool is_prompt) const {
  const int32_t pos = pos_tensor.index<int32_t>(0);
  auto [input_tokens, input_embeddings, input_token_num] = embedding_output;

  int32_t index = 0;
  if (is_prompt) {
    index = pos;
  }
  
  // Check data type of embeddings
  base::DataType dtype = input_embeddings.data_type();
  
  // For all models, use config_->dim_ as the embedding dimension
  const int32_t dim = config_->dim_;

  tensor::Tensor input(dtype, dim);
  
  if (dtype == base::DataType::kDataTypeFp16) {
    std::shared_ptr<base::Buffer> input_emb_buffer = std::make_shared<base::Buffer>(
        dim * sizeof(uint16_t), nullptr,
        input_embeddings.ptr<uint16_t>(index * dim), true);
    input.assign(input_emb_buffer);
  } else {
    std::shared_ptr<base::Buffer> input_emb_buffer = std::make_shared<base::Buffer>(
        dim * sizeof(float), nullptr,
        input_embeddings.ptr<float>(index * dim), true);
    input.assign(input_emb_buffer);
  }
  
  input.set_device_type(device_type_);
  return input;
}

}  // namespace model