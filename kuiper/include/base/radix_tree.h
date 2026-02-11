#ifndef KUIPER_INCLUDE_BASE_RADIX_TREE_H_
#define KUIPER_INCLUDE_BASE_RADIX_TREE_H_

#include <memory>
#include <vector>
#include <unordered_map>
#include <optional>
#include <algorithm>
#include <functional>
#include <cstdint>
#include <mutex>
#include <atomic>
#include <iostream>
#include <sstream>

namespace base {

/**
 * @brief RadixTree (基数树/压缩前缀树) 实现
 * 
 * RadixTree 是一种优化的前缀树，用于高效地存储和检索具有公共前缀的序列。
 * 与普通 Trie 不同，RadixTree 将只有一个子节点的节点与其子节点合并，
 * 从而减少树的深度和空间复杂度。
 * 
 * 主要用于:
 * - Token 序列到 KV Cache 位置的映射
 * - 高效的前缀匹配（找到最长公共前缀）
 * - 支持多个请求共享相同的 KV Cache 前缀
 * 
 * 时间复杂度:
 * - 插入: O(n) 其中 n 是序列长度
 * - 查找: O(n)
 * - 前缀匹配: O(n)
 * 
 * 空间复杂度: O(m) 其中 m 是所有唯一字符数
 */

/**
 * @brief RadixTree 节点
 * 
 * 每个节点代表一个 token 序列段，可以存储 KV cache 的起始位置和长度
 */
struct RadixNode {
    // 边上的 token 序列（从父节点到此节点）
    std::vector<int32_t> edge_tokens;
    
    // 子节点映射: token[0] -> 子节点
    std::unordered_map<int32_t, std::shared_ptr<RadixNode>> children;
    
    // 是否是完整序列的终点
    bool is_terminal = false;
    
    // KV cache 信息（仅在 is_terminal 为 true 时有效）
    struct KVCacheInfo {
        int32_t kv_start_pos = 0;     // KV cache 起始位置
        int32_t kv_length = 0;        // KV cache 长度
        int64_t last_access_time = 0; // 最后访问时间（用于 LRU 淘汰）
        int32_t ref_count = 0;        // 引用计数（有多少请求正在使用）
    } kv_info;
    
    // 从根到此节点的完整 token 序列长度
    int32_t prefix_length = 0;
    
    // 父节点（用于遍历）
    std::weak_ptr<RadixNode> parent;
    
    RadixNode() = default;
    
    explicit RadixNode(const std::vector<int32_t>& tokens)
        : edge_tokens(tokens) {}
    
    bool is_leaf() const {
        return children.empty();
    }
};

/**
 * @brief RadixTree 前缀匹配结果
 */
struct RadixMatchResult {
    // 匹配的 token 数量
    int32_t matched_length = 0;
    
    // 匹配到的节点（可能是部分匹配）
    std::shared_ptr<RadixNode> matched_node;
    
    // 在 matched_node 的 edge_tokens 中匹配的位置
    int32_t edge_offset = 0;
    
    // 对应的 KV cache 位置（如果存在）
    int32_t kv_cache_pos = -1;
    
    // 是否找到有效的 KV cache
    bool has_kv_cache() const {
        return kv_cache_pos >= 0 && matched_length > 0;
    }
};

/**
 * @brief RadixTree 实现
 * 
 * 线程安全版本，支持并发访问
 */
class RadixTree {
 public:
    RadixTree() {
        root_ = std::make_shared<RadixNode>();
        root_->prefix_length = 0;
    }
    
    /**
     * @brief 插入 token 序列并关联 KV cache 信息
     * 
     * @param tokens token 序列
     * @param kv_start_pos KV cache 起始位置
     * @param kv_length KV cache 长度
     */
    void insert(const std::vector<int32_t>& tokens, int32_t kv_start_pos, int32_t kv_length) {
        std::lock_guard<std::mutex> lock(mutex_);
        insert_impl(tokens, kv_start_pos, kv_length);
    }
    
    /**
     * @brief 查找最长前缀匹配
     * 
     * @param tokens 要查找的 token 序列
     * @return 匹配结果
     */
    RadixMatchResult find_longest_prefix(const std::vector<int32_t>& tokens) const {
        std::lock_guard<std::mutex> lock(mutex_);
        return find_longest_prefix_impl(tokens);
    }
    
    /**
     * @brief 检查序列是否存在
     */
    bool contains(const std::vector<int32_t>& tokens) const {
        std::lock_guard<std::mutex> lock(mutex_);
        auto result = find_longest_prefix_impl(tokens);
        return result.matched_length == static_cast<int32_t>(tokens.size()) && 
               result.matched_node && result.matched_node->is_terminal;
    }
    
    /**
     * @brief 删除 token 序列
     * 
     * @param tokens 要删除的序列
     * @return 是否成功删除
     */
    bool remove(const std::vector<int32_t>& tokens) {
        std::lock_guard<std::mutex> lock(mutex_);
        return remove_impl(tokens);
    }
    
    /**
     * @brief 增加节点引用计数
     */
    void add_reference(const std::vector<int32_t>& tokens) {
        std::lock_guard<std::mutex> lock(mutex_);
        auto result = find_longest_prefix_impl(tokens);
        if (result.matched_node && result.matched_length == static_cast<int32_t>(tokens.size())) {
            result.matched_node->kv_info.ref_count++;
            result.matched_node->kv_info.last_access_time = get_current_time();
        }
    }
    
    /**
     * @brief 减少节点引用计数
     */
    void release_reference(const std::vector<int32_t>& tokens) {
        std::lock_guard<std::mutex> lock(mutex_);
        auto result = find_longest_prefix_impl(tokens);
        if (result.matched_node && result.matched_length == static_cast<int32_t>(tokens.size())) {
            if (result.matched_node->kv_info.ref_count > 0) {
                result.matched_node->kv_info.ref_count--;
            }
        }
    }
    
    /**
     * @brief 清空树
     */
    void clear() {
        std::lock_guard<std::mutex> lock(mutex_);
        root_ = std::make_shared<RadixNode>();
        root_->prefix_length = 0;
        total_cached_tokens_ = 0;
    }
    
    /**
     * @brief 获取缓存的总 token 数
     */
    int64_t get_total_cached_tokens() const {
        return total_cached_tokens_.load();
    }
    
    /**
     * @brief 基于 LRU 策略淘汰节点
     * 
     * @param max_tokens 最大保留 token 数
     * @return 被淘汰的 token 数
     */
    int64_t evict_lru(int64_t max_tokens) {
        std::lock_guard<std::mutex> lock(mutex_);
        
        if (total_cached_tokens_ <= max_tokens) {
            return 0;
        }
        
        // 收集所有可淘汰的叶子节点
        std::vector<std::pair<int64_t, std::shared_ptr<RadixNode>>> evictable;
        collect_evictable_nodes(root_, evictable);
        
        // 按最后访问时间排序（最早的在前）
        std::sort(evictable.begin(), evictable.end(),
                  [](const auto& a, const auto& b) {
                      return a.first < b.first;
                  });
        
        int64_t evicted = 0;
        for (const auto& [time, node] : evictable) {
            if (total_cached_tokens_ <= max_tokens) {
                break;
            }
            
            // 跳过正在使用的节点
            if (node->kv_info.ref_count > 0) {
                continue;
            }
            
            int64_t tokens_freed = node->prefix_length;
            // 标记为非终端（相当于删除 KV cache 关联）
            node->is_terminal = false;
            node->kv_info = RadixNode::KVCacheInfo{};
            total_cached_tokens_ -= tokens_freed;
            evicted += tokens_freed;
        }
        
        return evicted;
    }
    
    /**
     * @brief 打印树结构（调试用）
     */
    std::string dump() const {
        std::lock_guard<std::mutex> lock(mutex_);
        std::stringstream ss;
        dump_impl(root_, "", ss);
        return ss.str();
    }
    
    /**
     * @brief 获取统计信息
     */
    struct Stats {
        int64_t total_nodes = 0;
        int64_t terminal_nodes = 0;
        int64_t total_cached_tokens = 0;
        int32_t max_depth = 0;
    };
    
    Stats get_stats() const {
        std::lock_guard<std::mutex> lock(mutex_);
        Stats stats;
        stats.total_cached_tokens = total_cached_tokens_;
        collect_stats(root_, 0, stats);
        return stats;
    }
    
 private:
    void insert_impl(const std::vector<int32_t>& tokens, int32_t kv_start_pos, int32_t kv_length) {
        if (tokens.empty()) {
            root_->is_terminal = true;
            root_->kv_info.kv_start_pos = kv_start_pos;
            root_->kv_info.kv_length = kv_length;
            root_->kv_info.last_access_time = get_current_time();
            return;
        }
        
        std::shared_ptr<RadixNode> current = root_;
        size_t token_idx = 0;
        
        while (token_idx < tokens.size()) {
            int32_t first_token = tokens[token_idx];
            
            auto it = current->children.find(first_token);
            if (it == current->children.end()) {
                // 没有匹配的子节点，创建新节点
                auto new_node = std::make_shared<RadixNode>();
                new_node->edge_tokens.assign(tokens.begin() + token_idx, tokens.end());
                new_node->prefix_length = static_cast<int32_t>(tokens.size());
                new_node->is_terminal = true;
                new_node->kv_info.kv_start_pos = kv_start_pos;
                new_node->kv_info.kv_length = kv_length;
                new_node->kv_info.last_access_time = get_current_time();
                new_node->parent = current;
                
                current->children[first_token] = new_node;
                total_cached_tokens_ += kv_length;
                return;
            }
            
            // 找到了匹配的子节点
            auto child = it->second;
            const auto& edge = child->edge_tokens;
            
            // 计算公共前缀长度
            size_t common_len = 0;
            size_t remaining = tokens.size() - token_idx;
            size_t max_common = std::min(edge.size(), remaining);
            
            while (common_len < max_common && 
                   edge[common_len] == tokens[token_idx + common_len]) {
                ++common_len;
            }
            
            if (common_len == edge.size()) {
                // 完全匹配边，继续向下
                token_idx += common_len;
                current = child;
            } else {
                // 部分匹配，需要分裂节点
                auto split_node = std::make_shared<RadixNode>();
                split_node->edge_tokens.assign(edge.begin(), edge.begin() + common_len);
                split_node->prefix_length = current->prefix_length + static_cast<int32_t>(common_len);
                split_node->parent = current;
                
                // 修改原节点的边
                child->edge_tokens.assign(edge.begin() + common_len, edge.end());
                child->parent = split_node;
                
                // 将原节点作为分裂节点的子节点
                split_node->children[child->edge_tokens[0]] = child;
                
                // 更新当前节点的子节点
                current->children[first_token] = split_node;
                
                token_idx += common_len;
                
                if (token_idx < tokens.size()) {
                    // 还有剩余 token，创建新子节点
                    auto new_node = std::make_shared<RadixNode>();
                    new_node->edge_tokens.assign(tokens.begin() + token_idx, tokens.end());
                    new_node->prefix_length = static_cast<int32_t>(tokens.size());
                    new_node->is_terminal = true;
                    new_node->kv_info.kv_start_pos = kv_start_pos;
                    new_node->kv_info.kv_length = kv_length;
                    new_node->kv_info.last_access_time = get_current_time();
                    new_node->parent = split_node;
                    
                    split_node->children[new_node->edge_tokens[0]] = new_node;
                } else {
                    // 分裂点就是终点
                    split_node->is_terminal = true;
                    split_node->kv_info.kv_start_pos = kv_start_pos;
                    split_node->kv_info.kv_length = kv_length;
                    split_node->kv_info.last_access_time = get_current_time();
                }
                
                total_cached_tokens_ += kv_length;
                return;
            }
        }
        
        // 到达现有节点，标记为终端
        if (!current->is_terminal) {
            total_cached_tokens_ += kv_length;
        }
        current->is_terminal = true;
        current->kv_info.kv_start_pos = kv_start_pos;
        current->kv_info.kv_length = kv_length;
        current->kv_info.last_access_time = get_current_time();
    }
    
    RadixMatchResult find_longest_prefix_impl(const std::vector<int32_t>& tokens) const {
        RadixMatchResult result;
        
        if (tokens.empty()) {
            result.matched_node = root_;
            result.matched_length = 0;
            if (root_->is_terminal) {
                result.kv_cache_pos = root_->kv_info.kv_start_pos;
            }
            return result;
        }
        
        std::shared_ptr<RadixNode> current = root_;
        std::shared_ptr<RadixNode> last_terminal = nullptr;
        int32_t last_terminal_length = 0;
        size_t token_idx = 0;
        
        // 检查根节点
        if (root_->is_terminal) {
            last_terminal = root_;
            last_terminal_length = 0;
        }
        
        while (token_idx < tokens.size()) {
            int32_t first_token = tokens[token_idx];
            
            auto it = current->children.find(first_token);
            if (it == current->children.end()) {
                // 没有匹配的子节点
                break;
            }
            
            auto child = it->second;
            const auto& edge = child->edge_tokens;
            
            // 计算公共前缀长度
            size_t common_len = 0;
            size_t remaining = tokens.size() - token_idx;
            size_t max_common = std::min(edge.size(), remaining);
            
            while (common_len < max_common && 
                   edge[common_len] == tokens[token_idx + common_len]) {
                ++common_len;
            }
            
            token_idx += common_len;
            
            if (common_len < edge.size()) {
                // 部分匹配边
                result.matched_node = child;
                result.edge_offset = static_cast<int32_t>(common_len);
                break;
            }
            
            // 完全匹配边
            current = child;
            
            if (child->is_terminal) {
                last_terminal = child;
                last_terminal_length = static_cast<int32_t>(token_idx);
            }
        }
        
        // 设置匹配结果
        if (last_terminal) {
            result.matched_node = last_terminal;
            result.matched_length = last_terminal_length;
            result.kv_cache_pos = last_terminal->kv_info.kv_start_pos;
        } else {
            result.matched_node = current;
            result.matched_length = static_cast<int32_t>(token_idx);
        }
        
        return result;
    }
    
    bool remove_impl(const std::vector<int32_t>& tokens) {
        auto result = find_longest_prefix_impl(tokens);
        
        if (result.matched_length != static_cast<int32_t>(tokens.size()) ||
            !result.matched_node || !result.matched_node->is_terminal) {
            return false;
        }
        
        auto node = result.matched_node;
        total_cached_tokens_ -= node->kv_info.kv_length;
        node->is_terminal = false;
        node->kv_info = RadixNode::KVCacheInfo{};
        
        // 可选：合并单子节点链
        // compact_node(node);
        
        return true;
    }
    
    void collect_evictable_nodes(
        const std::shared_ptr<RadixNode>& node,
        std::vector<std::pair<int64_t, std::shared_ptr<RadixNode>>>& evictable) const {
        
        if (node->is_terminal && node->is_leaf()) {
            evictable.emplace_back(node->kv_info.last_access_time, node);
        }
        
        for (const auto& [token, child] : node->children) {
            collect_evictable_nodes(child, evictable);
        }
    }
    
    void collect_stats(const std::shared_ptr<RadixNode>& node, int depth, Stats& stats) const {
        stats.total_nodes++;
        stats.max_depth = std::max(stats.max_depth, depth);
        
        if (node->is_terminal) {
            stats.terminal_nodes++;
        }
        
        for (const auto& [token, child] : node->children) {
            collect_stats(child, depth + 1, stats);
        }
    }
    
    void dump_impl(const std::shared_ptr<RadixNode>& node, 
                   const std::string& prefix, std::stringstream& ss) const {
        if (node != root_) {
            ss << prefix << "Edge[";
            for (size_t i = 0; i < node->edge_tokens.size() && i < 5; ++i) {
                if (i > 0) ss << ",";
                ss << node->edge_tokens[i];
            }
            if (node->edge_tokens.size() > 5) {
                ss << "...+" << (node->edge_tokens.size() - 5);
            }
            ss << "]";
            
            if (node->is_terminal) {
                ss << " *KV(pos=" << node->kv_info.kv_start_pos 
                   << ",len=" << node->kv_info.kv_length
                   << ",ref=" << node->kv_info.ref_count << ")";
            }
            ss << "\n";
        } else {
            ss << "ROOT\n";
        }
        
        for (const auto& [token, child] : node->children) {
            dump_impl(child, prefix + "  ", ss);
        }
    }
    
    static int64_t get_current_time() {
        return std::chrono::duration_cast<std::chrono::milliseconds>(
            std::chrono::steady_clock::now().time_since_epoch()).count();
    }
    
    std::shared_ptr<RadixNode> root_;
    mutable std::mutex mutex_;
    std::atomic<int64_t> total_cached_tokens_{0};
};

}  // namespace base

#endif  // KUIPER_INCLUDE_BASE_RADIX_TREE_H_
