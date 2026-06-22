#pragma once

#include <string>

namespace orinmllm::graph {

class DescInfo {
 public:
  DescInfo() = default;
  virtual ~DescInfo() = default;

  const std::string& name() const {
    if (!name_.empty()) {
      return name_;
    }
    return session_;
  }

  const std::string& session() const { return session_; }
  const std::string& description() const { return description_; }

  bool set_name(const std::string& name) {
    name_ = name;
    return true;
  }

  bool set_session(const std::string& session) {
    session_ = session;
    return true;
  }

  bool set_description(const std::string& description) {
    description_ = description;
    return true;
  }

 protected:
  std::string name_;
  std::string session_;
  std::string description_;
};

}  // namespace orinmllm::graph
