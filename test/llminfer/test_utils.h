#pragma once

#include <cstdlib>
#include <iostream>

inline int ExpectTrueOrExit(const bool condition, const char* const expr,
                            const char* const file, const int line) {
  if (condition) {
    return EXIT_SUCCESS;
  }
  std::cerr << "EXPECT_TRUE failed: " << expr << " at " << file << ":" << line << std::endl;
  return EXIT_FAILURE;
}

template <typename Lhs, typename Rhs>
int ExpectEqOrExit(const Lhs& lhs, const Rhs& rhs, const char* const lhs_expr,
                   const char* const rhs_expr, const char* const file, const int line) {
  if (lhs == rhs) {
    return EXIT_SUCCESS;
  }
  std::cerr << "EXPECT_EQ failed: " << lhs_expr << " == " << rhs_expr << " at " << file << ":"
            << line << ", lhs=" << lhs << ", rhs=" << rhs << std::endl;
  return EXIT_FAILURE;
}

#define EXPECT_TRUE_OR_EXIT(condition) if (ExpectTrueOrExit((condition), #condition, __FILE__, __LINE__) != EXIT_SUCCESS) return EXIT_FAILURE

#define EXPECT_EQ_OR_EXIT(lhs, rhs) if (ExpectEqOrExit((lhs), (rhs), #lhs, #rhs, __FILE__, __LINE__) != EXIT_SUCCESS) return EXIT_FAILURE