#ifndef KUIPER_INCLUDE_BASE_PREFIX_CACHE_H_
#define KUIPER_INCLUDE_BASE_PREFIX_CACHE_H_

#include <memory>
#include <vector>
#include <unordered_map>
#include <optional>
#include <mutex>
#include <atomic>
#include <functional>
#include <cstdint>
#include <chrono>
#include "radix_tree.h"

namespace base {

/**
 * @brief PrefixCache - 基于 RadixTree 的 KV Cache 前缀复用系统
 * 
 * 实现 SGLang 风格的 RadixAttention 算法:
 * 1. 使用 RadixTree 存储 token 序列到 KV Cache 的映射
 * 2. 新请求到来时，通过 RadixTree 查找最长公共前缀
 * 3. 复用匹配的 KV Cache，只计算新增部分
 * 
 * 核心概念:
 * - Prefix: 已经计算过的 token 序列及其对应的 KV Cache
 * - Match: 新请求与已缓存前缀的匹配
 * - Reuse: 复用已计算的 KV Cache，避免重复计算
 * 
 * 与简单的单请求 KV Cache 复用的区别:
 * - 支持多个不同请求之间的前缀共享
 * - 使用树结构高效管理多个前缀
 * - 支持 LRU 淘汰策略控制内存使用
 */

/**
 * @brief 前缀匹配结果
 */
struct PrefixMatchResult {
    // 匹配的 token 数量
    int32_t matched_tokens = 0;
    
    // 是否命中缓存
    bool cache_hit = false;
    
    // 需要从哪个位置开始 prefill
    int32_t prefill_start_pos = 0;
    
    // 需要 prefill 的 token 数量
    int32_t prefill_count = 0;
    
    // KV Cache 复用比例 (0.0 - 1.0)
    float reuse_ratio = 0.0f;
    
    // 匹配到的缓存 key（用于引用计数）
    std::vector<int32_t> matched_prefix;
};

/**
 * @brief KV Cache 块信息
 * 
 * 记录一段 KV Cache 的位置和状态
 */
struct KVCacheBlock {
    int32_t start_pos = 0;      // 在 KV Cache 中的起始位置
    int32_t length = 0;         // 块长度
    int32_t ref_count = 0;      // 引用计数
    int64_t last_access = 0;    // 最后访问时间
    bool is_valid = false;      // 是否有效
    
    // 对应的 token 序列（用于验证）
    std::vector<int32_t> tokens;
};

/**
 * @brief PrefixCache 配置
 */
struct PrefixCacheConfig {
    // 最大缓存 token 数（超过后触发 LRU 淘汰）
    int64_t max_cached_tokens = 65536;  // 64K tokens
    
    // 最小前缀匹配长度（太短的前缀不值得复用）
    int32_t min_prefix_length = 4;
    
    // 是否启用自动淘汰
    bool enable_auto_eviction = true;
    
    // 淘汰阈值（当使用率超过此值时触发淘汰）
    float eviction_threshold = 0.9f;
    
    // 是否启用统计
    bool enable_stats = true;
};

/**
 * @brief PrefixCache 统计信息
 */
struct PrefixCacheStats {
    std::atomic<int64_t> total_requests{0};
    std::atomic<int64_t> cache_hits{0};
    std::atomic<int64_t> cache_misses{0};
    std::atomic<int64_t> total_tokens_processed{0};
    std::atomic<int64_t> tokens_reused{0};
    std::atomic<int64_t> tokens_computed{0};
    std::atomic<int64_t> eviction_count{0};
    std::atomic<int64_t> tokens_evicted{0};
    
    void reset() {
        total_requests = 0;
        cache_hits = 0;
        cache_misses = 0;
        total_tokens_processed = 0;
        tokens_reused = 0;
        tokens_computed = 0;
        eviction_count = 0;
        tokens_evicted = 0;
    }
    
    float hit_rate() const {
        int64_t total = total_requests.load();
        if (total == 0) return 0.0f;
        return static_cast<float>(cache_hits.load()) / total;
    }
    
    float reuse_rate() const {
        int64_t total = total_tokens_processed.load();
        if (total == 0) return 0.0f;
        return static_cast<float>(tokens_reused.load()) / total;
    }
    
    std::string to_string() const {
        std::stringstream ss;
        ss << "PrefixCache Stats:\n";
        ss << "  Total requests: " << total_requests.load() << "\n";
        ss << "  Cache hits: " << cache_hits.load() << " (" 
           << (hit_rate() * 100) << "%)\n";
        ss << "  Cache misses: " << cache_misses.load() << "\n";
        ss << "  Total tokens: " << total_tokens_processed.load() << "\n";
        ss << "  Tokens reused: " << tokens_reused.load() << " ("
           << (reuse_rate() * 100) << "%)\n";
        ss << "  Tokens computed: " << tokens_computed.load() << "\n";
        ss << "  Evictions: " << eviction_count.load() << "\n";
        ss << "  Tokens evicted: " << tokens_evicted.load() << "\n";
        return ss.str();
    }
};

/**
 * @brief PrefixCache 主类
 */
class PrefixCache {
 public:
    explicit PrefixCache(const PrefixCacheConfig& config = PrefixCacheConfig())
        : config_(config), radix_tree_(std::make_unique<RadixTree>()) {}
    
    /**
     * @brief 查找最长前缀匹配
     * 
     * @param tokens 新请求的 token 序列
     * @return 匹配结果
     */
    PrefixMatchResult match(const std::vector<int32_t>& tokens) {
        PrefixMatchResult result;
        result.prefill_count = static_cast<int32_t>(tokens.size());
        result.prefill_start_pos = 0;
        
        if (tokens.empty()) {
            return result;
        }
        
        // 更新统计
        if (config_.enable_stats) {
            stats_.total_requests++;
            stats_.total_tokens_processed += tokens.size();
        }
        
        // 在 RadixTree 中查找最长前缀
        auto radix_result = radix_tree_->find_longest_prefix(tokens);
        
        if (radix_result.has_kv_cache() && 
            radix_result.matched_length >= config_.min_prefix_length) {
            // 找到有效的前缀匹配
            result.matched_tokens = radix_result.matched_length;
            result.cache_hit = true;
            result.prefill_start_pos = radix_result.matched_length;
            result.prefill_count = static_cast<int32_t>(tokens.size()) - radix_result.matched_length;
            result.reuse_ratio = static_cast<float>(radix_result.matched_length) / tokens.size();
            result.matched_prefix.assign(tokens.begin(), tokens.begin() + radix_result.matched_length);
            
            // 增加引用计数
            radix_tree_->add_reference(result.matched_prefix);
            
            // 更新统计
            if (config_.enable_stats) {
                stats_.cache_hits++;
                stats_.tokens_reused += radix_result.matched_length;
                stats_.tokens_computed += result.prefill_count;
            }
        } else {
            // 没有找到有效匹配
            result.cache_hit = false;
            
            if (config_.enable_stats) {
                stats_.cache_misses++;
                stats_.tokens_computed += tokens.size();
            }
        }
        
        return result;
    }
    
    /**
     * @brief 注册新的前缀到缓存
     * 
     * @param tokens token 序列
     * @param kv_start_pos KV Cache 起始位置
     * @param kv_length KV Cache 长度
     */
    void insert(const std::vector<int32_t>& tokens, int32_t kv_start_pos, int32_t kv_length) {
        if (tokens.empty() || kv_length <= 0) {
            return;
        }
        
        // 检查是否需要淘汰
        if (config_.enable_auto_eviction) {
            maybe_evict();
        }
        
        // 插入到 RadixTree
        radix_tree_->insert(tokens, kv_start_pos, kv_length);
    }
    
    /**
     * @brief 释放前缀引用
     * 
     * 当请求完成时调用，减少引用计数
     */
    void release(const std::vector<int32_t>& tokens) {
        if (!tokens.empty()) {
            radix_tree_->release_reference(tokens);
        }
    }
    
    /**
     * @brief 删除指定前缀
     */
    bool remove(const std::vector<int32_t>& tokens) {
        return radix_tree_->remove(tokens);
    }
    
    /**
     * @brief 清空所有缓存
     */
    void clear() {
        radix_tree_->clear();
    }
    
    /**
     * @brief 获取当前缓存的 token 数
     */
    int64_t get_cached_tokens() const {
        return radix_tree_->get_total_cached_tokens();
    }
    
    /**
     * @brief 获取缓存使用率
     */
    float get_usage_ratio() const {
        return static_cast<float>(get_cached_tokens()) / config_.max_cached_tokens;
    }
    
    /**
     * @brief 手动触发 LRU 淘汰
     * 
     * @param target_tokens 目标 token 数
     * @return 被淘汰的 token 数
     */
    int64_t evict(int64_t target_tokens) {
        int64_t evicted = radix_tree_->evict_lru(target_tokens);
        if (evicted > 0 && config_.enable_stats) {
            stats_.eviction_count++;
            stats_.tokens_evicted += evicted;
        }
        return evicted;
    }
    
    /**
     * @brief 获取统计信息
     */
    const PrefixCacheStats& get_stats() const {
        return stats_;
    }
    
    /**
     * @brief 重置统计信息
     */
    void reset_stats() {
        stats_.reset();
    }
    
    /**
     * @brief 获取配置
     */
    const PrefixCacheConfig& get_config() const {
        return config_;
    }
    
    /**
     * @brief 更新配置
     */
    void set_config(const PrefixCacheConfig& config) {
        config_ = config;
    }
    
    /**
     * @brief 打印 RadixTree 结构（调试用）
     */
    std::string dump_tree() const {
        return radix_tree_->dump();
    }
    
    /**
     * @brief 获取 RadixTree 统计信息
     */
    RadixTree::Stats get_tree_stats() const {
        return radix_tree_->get_stats();
    }
    
 private:
    void maybe_evict() {
        float usage = get_usage_ratio();
        if (usage > config_.eviction_threshold) {
            // 淘汰到 80% 容量
            int64_t target = static_cast<int64_t>(config_.max_cached_tokens * 0.8);
            evict(target);
        }
    }
    
    PrefixCacheConfig config_;
    std::unique_ptr<RadixTree> radix_tree_;
    PrefixCacheStats stats_;
};

}  // namespace base

#endif  // KUIPER_INCLUDE_BASE_PREFIX_CACHE_H_
