#include "common.h"

namespace common {

Status::Status(StatusCode code, std::string err_message)
    : code_(code), err_message_(std::move(err_message)) {}

Status& Status::operator=(StatusCode code) {
    code_ = code;
    return *this;
}

bool Status::operator==(StatusCode code) const {
    return code_ == code;
}

bool Status::operator!=(StatusCode code) const {
    return code_ != code;
}

int32_t Status::get_err_code() const {
    return static_cast<int32_t>(code_);
}

const std::string& Status::get_err_msg() const {
    return err_message_;
}

void Status::set_err_msg(const std::string& err_msg) {
    err_message_ = err_msg;
}

Status Success(const std::string& err_msg) { return Status{StatusCode::kSuccess, err_msg}; }

Status FunctionNotImplement(const std::string& err_msg) {
  return Status{StatusCode::kFunctionUnImplement, err_msg};
}

Status PathNotValid(const std::string& err_msg) {
  return Status{StatusCode::kPathNotValid, err_msg};
}

Status ModelParseError(const std::string& err_msg) {
  return Status{StatusCode::kModelParseError, err_msg};
}

Status InternalError(const std::string& err_msg) {
  return Status{StatusCode::kInternalError, err_msg};
}

Status InvalidArgument(const std::string& err_msg) {
  return Status{StatusCode::kInvalidArgument, err_msg};
}

Status KeyHasExits(const std::string& err_msg) {
  return Status{StatusCode::kKeyValueHasExist, err_msg};
}

}