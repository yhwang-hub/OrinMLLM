/**
 * @file test_radix_tree.cpp
 * @brief RadixTree PrefixCache 单元测试
 */

#include <iostream>
#include <vector>
#include <cassert>
#include "../kuiper/include/base/radix_tree.h"
#include "../kuiper/include/base/prefix_cache.h"

using namespace base;

void test_basic_operations() {
    std::cout << "=== Test 1: Basic RadixTree Operations ===" << std::endl;
    
    RadixTree tree;
    
    // 测试插入和查找
    std::vector<int32_t> seq1 = {1, 2, 3, 4, 5};
    
    tree.insert(seq1, 0, 5);  // kv_start_pos=0, kv_length=5
    std::cout << "Inserted sequence [1,2,3,4,5] with cache info (0, 5)" << std::endl;
    
    // 查找完全匹配
    auto result = tree.find_longest_prefix(seq1);
    assert(result.matched_length == 5);
    std::cout << "Found exact match: " << result.matched_length << " tokens" << std::endl;
    
    // 查找部分匹配
    std::vector<int32_t> seq2 = {1, 2, 3, 4, 5, 6, 7};
    result = tree.find_longest_prefix(seq2);
    assert(result.matched_length == 5);
    std::cout << "Found partial match for [1,2,3,4,5,6,7]: " << result.matched_length << " tokens" << std::endl;
    
    // 查找不匹配
    std::vector<int32_t> seq3 = {10, 20, 30};
    result = tree.find_longest_prefix(seq3);
    assert(result.matched_length == 0);
    std::cout << "No match for [10,20,30]: " << result.matched_length << " tokens" << std::endl;
    
    std::cout << "Test 1 PASSED!" << std::endl << std::endl;
}

void test_prefix_sharing() {
    std::cout << "=== Test 2: Prefix Sharing ===" << std::endl;
    
    RadixTree tree;
    
    // 插入多个共享前缀的序列
    std::vector<int32_t> seq1 = {1, 2, 3, 4, 5};
    std::vector<int32_t> seq2 = {1, 2, 3, 6, 7};
    std::vector<int32_t> seq3 = {1, 2, 8, 9, 10};
    
    tree.insert(seq1, 0, 5);
    tree.insert(seq2, 0, 5);
    tree.insert(seq3, 0, 5);
    
    std::cout << "Inserted 3 sequences with shared prefixes" << std::endl;
    
    // 查找应该利用共享前缀
    auto r1 = tree.find_longest_prefix({1, 2, 3, 4, 5, 100});
    auto r2 = tree.find_longest_prefix({1, 2, 3, 6, 7, 100});
    auto r3 = tree.find_longest_prefix({1, 2, 8, 9, 10, 100});
    
    assert(r1.matched_length == 5);
    assert(r2.matched_length == 5);
    assert(r3.matched_length == 5);
    
    std::cout << "All 3 sequences correctly matched!" << std::endl;
    std::cout << "Test 2 PASSED!" << std::endl << std::endl;
}

void test_prefix_cache() {
    std::cout << "=== Test 3: PrefixCache Wrapper ===" << std::endl;
    
    PrefixCacheConfig config;
    config.max_cached_tokens = 1000;
    config.eviction_threshold = 0.9f;
    config.min_prefix_length = 2;
    
    PrefixCache cache(config);
    
    // 模拟多轮对话
    // Turn 1: 系统prompt + 用户问题
    std::vector<int32_t> turn1 = {1, 2, 3, 4, 5, 10, 11, 12};  // system + user1
    cache.insert(turn1, 0, 8);
    std::cout << "Turn 1: Inserted " << turn1.size() << " tokens" << std::endl;
    
    // Turn 2: 系统prompt + 用户问题 + 助手回复 + 新用户问题
    std::vector<int32_t> turn2 = {1, 2, 3, 4, 5, 10, 11, 12, 20, 21, 22, 30, 31, 32};
    auto match = cache.match(turn2);
    
    std::cout << "Turn 2: Matching " << turn2.size() << " tokens" << std::endl;
    std::cout << "  Cache hit: " << (match.cache_hit ? "yes" : "no") << std::endl;
    std::cout << "  Matched: " << match.matched_tokens << " tokens" << std::endl;
    std::cout << "  Reuse ratio: " << (int)(match.reuse_ratio * 100) << "%" << std::endl;
    
    assert(match.cache_hit);
    assert(match.matched_tokens == 8);  // 应该匹配 turn1 的所有 tokens
    
    // 注册 turn2
    cache.insert(turn2, 0, 14);
    cache.release(match.matched_prefix);
    
    // 检查统计
    const auto& stats = cache.get_stats();
    std::cout << "Stats:" << std::endl;
    std::cout << "  Requests: " << stats.total_requests.load() << std::endl;
    std::cout << "  Hits: " << stats.cache_hits.load() << std::endl;
    std::cout << "  Hit rate: " << (int)(stats.hit_rate() * 100) << "%" << std::endl;
    std::cout << "  Tokens reused: " << stats.tokens_reused.load() << std::endl;
    
    std::cout << "Test 3 PASSED!" << std::endl << std::endl;
}

void test_lru_eviction() {
    std::cout << "=== Test 4: LRU Eviction ===" << std::endl;
    
    PrefixCacheConfig config;
    config.max_cached_tokens = 20;  // 很小的缓存
    config.eviction_threshold = 0.9f;
    config.min_prefix_length = 1;
    
    PrefixCache cache(config);
    
    // 插入多个序列
    cache.insert({1, 2, 3, 4, 5}, 0, 5);
    cache.insert({10, 11, 12, 13, 14}, 5, 5);
    cache.insert({20, 21, 22, 23, 24}, 10, 5);
    
    std::cout << "Inserted 15 tokens into cache with max 20" << std::endl;
    std::cout << "Current cached tokens: " << cache.get_cached_tokens() << std::endl;
    
    // 插入超过阈值的序列，触发淘汰
    cache.insert({30, 31, 32, 33, 34, 35, 36, 37}, 15, 8);
    
    std::cout << "After inserting 8 more tokens:" << std::endl;
    std::cout << "Current cached tokens: " << cache.get_cached_tokens() << std::endl;
    
    std::cout << "Test 4 PASSED!" << std::endl << std::endl;
}

void test_multi_turn_simulation() {
    std::cout << "=== Test 5: Multi-Turn Conversation Simulation ===" << std::endl;
    
    PrefixCacheConfig config;
    config.max_cached_tokens = 10000;
    config.min_prefix_length = 4;
    
    PrefixCache cache(config);
    
    // 模拟 Qwen chat 模板的 token 结构
    // <|im_start|>system\n你是...<|im_end|>\n
    std::vector<int32_t> system_tokens = {151644, 8948, 198, 100, 101, 102, 151645, 198};  // 8 tokens
    
    // Turn 1: system + user1
    std::vector<int32_t> turn1_prompt;
    turn1_prompt.insert(turn1_prompt.end(), system_tokens.begin(), system_tokens.end());
    turn1_prompt.insert(turn1_prompt.end(), {151644, 872, 198, 1, 2, 3, 151645, 198});  // user1: 8 tokens
    turn1_prompt.insert(turn1_prompt.end(), {151644, 77091, 198});  // <|im_start|>assistant\n: 3 tokens
    // Total: 19 tokens
    
    cache.insert(turn1_prompt, 0, 19);
    std::cout << "Turn 1: Inserted " << turn1_prompt.size() << " prompt tokens" << std::endl;
    
    // 模拟生成了 10 个 token 的回复
    std::vector<int32_t> turn1_response = {200, 201, 202, 203, 204, 205, 206, 207, 208, 209};
    std::vector<int32_t> turn1_full = turn1_prompt;
    turn1_full.insert(turn1_full.end(), turn1_response.begin(), turn1_response.end());
    turn1_full.push_back(151645);  // <|im_end|>
    turn1_full.push_back(198);     // \n
    // Total: 31 tokens
    
    cache.insert(turn1_full, 0, 31);
    std::cout << "Turn 1: Total " << turn1_full.size() << " tokens after response" << std::endl;
    
    // Turn 2: 继续对话
    std::vector<int32_t> turn2_prompt = turn1_full;
    turn2_prompt.insert(turn2_prompt.end(), {151644, 872, 198, 4, 5, 6, 151645, 198});  // user2
    turn2_prompt.insert(turn2_prompt.end(), {151644, 77091, 198});  // assistant
    
    auto match2 = cache.match(turn2_prompt);
    std::cout << "Turn 2:" << std::endl;
    std::cout << "  Prompt tokens: " << turn2_prompt.size() << std::endl;
    std::cout << "  Cache hit: " << (match2.cache_hit ? "yes" : "no") << std::endl;
    std::cout << "  Matched: " << match2.matched_tokens << " tokens" << std::endl;
    std::cout << "  Need prefill: " << match2.prefill_count << " tokens" << std::endl;
    std::cout << "  Reuse ratio: " << (int)(match2.reuse_ratio * 100) << "%" << std::endl;
    
    assert(match2.cache_hit);
    assert(match2.matched_tokens == 31);  // 应该复用 turn1 的全部
    
    cache.release(match2.matched_prefix);
    
    // 最终统计
    std::cout << "\nFinal stats:" << std::endl;
    std::cout << cache.get_stats().to_string() << std::endl;
    
    std::cout << "Test 5 PASSED!" << std::endl << std::endl;
}

int main() {
    std::cout << "========================================" << std::endl;
    std::cout << "RadixTree PrefixCache Unit Tests" << std::endl;
    std::cout << "========================================" << std::endl << std::endl;
    
    test_basic_operations();
    test_prefix_sharing();
    test_prefix_cache();
    test_lru_eviction();
    test_multi_turn_simulation();
    
    std::cout << "========================================" << std::endl;
    std::cout << "All tests PASSED!" << std::endl;
    std::cout << "========================================" << std::endl;
    
    return 0;
}
