# jinja.cpp 工程分析及其在 OrinMLLM 中的使用报告

---

## 一、jinja.cpp 工程的作用、使用方法、测试用例及存在必要性

### 1.1 工程概述

`jinja.cpp` 是一个**轻量级、单头文件**的 C++11 Jinja2 模板引擎，专门为 LLM（大语言模型）聊天模板（Chat Template）设计。它的核心目标是在纯 C++ 环境中复现 HuggingFace `transformers` 库中 `tokenizer.apply_chat_template()` 的功能。

**工程结构**：

```
jinja.cpp/
├── jinja.hpp                  # 核心：2137行的单头文件模板引擎
├── CMakeLists.txt             # CMake构建文件
├── third_party/
│   ├── ujson.hpp              # 614行的统一JSON桥接层
│   ├── nlohmann/              # nlohmann/json 头文件（默认后端）
│   └── rapidjson/             # RapidJSON 头文件（可选后端）
├── tests/
│   ├── test_main.cpp          # 227行测试驱动程序
│   ├── test_chat_template.json # 10710行测试数据（390+条用例）
│   └── generate_assets.py     # 327行Python脚本，用于从HuggingFace生成测试数据
└── doc/
    ├── implementation_details.md
    └── implementation_details_CN.md
```

### 1.2 jinja.cpp 的作用

在 LLM 推理系统中，用户输入的自然语言消息需要按照**特定格式**（Chat Template）转换为模型能理解的 prompt 字符串。不同模型使用不同的模板格式，例如：

- **Qwen 系列**使用 `<|im_start|>role\ncontent<|im_end|>` 格式
- **Llama 3** 使用 `<|begin_of_text|><|start_header_id|>role<|end_header_id|>` 格式
- **DeepSeek** 使用类似但又有差异的格式

这些模板都以 **Jinja2 模板语法**定义在 HuggingFace 的 `tokenizer_config.json` 中。在 Python 中，可直接调用 `tokenizer.apply_chat_template()` 完成转换。但在 **C++ 推理引擎**中（如 OrinMLLM、llama.cpp），没有 Python 运行时，需要一个纯 C++ 的 Jinja2 解释器来处理这些模板。

**`jinja.cpp` 正是填补了这一空白**——它用 2137 行 C++ 代码实现了一个完整的 Jinja2 子集解释器，可以直接解析和执行这些 Chat Template。

### 1.3 使用方法

#### 基础渲染

```cpp
#include "jinja.hpp"

// 1. 从模板字符串构造 Template 对象
jinja::Template tpl("Hello {{ name }}!");

// 2. 构造上下文（变量）
jinja::json context;
context["name"] = "World";

// 3. 渲染
std::string result = tpl.render(context);
// result = "Hello World!"
```

#### LLM Chat Template 用法

```cpp
#include "jinja.hpp"

// 1. 加载 chat template（通常从 tokenizer_config.json 读取）
std::string chat_template_str = "{% for message in messages %}...{% endfor %}";
jinja::Template tpl(chat_template_str);

// 2. 构造 messages JSON 数组
jinja::json messages = jinja::json::array({
    {{"role", "user"}, {"content", "你好！"}}
});

// 3. 调用 apply_chat_template（模拟 HuggingFace API）
std::string prompt = tpl.apply_chat_template(
    messages,
    true,                    // add_generation_prompt
    jinja::json::array()     // tools（可选）
);
```

#### 自定义函数注入

```cpp
tpl.add_function("strftime_now", [](const std::vector<jinja::Argument>& args) -> jinja::json {
    std::string format = "%Y-%m-%d";
    if (args.size() > 0) format = args[0].second.get<std::string>();
    std::time_t t = std::time(nullptr);
    std::tm tm = *std::localtime(&t);
    std::stringstream ss;
    ss << std::put_time(&tm, format.c_str());
    return ss.str();
});
```

### 1.4 测试用例详解

测试系统由三部分组成：

#### (1) `generate_assets.py` —— 测试数据生成器

这个 Python 脚本负责使用**官方 Python `transformers` 库**为 40+ 个真实模型生成标准测试数据：

```python
MODELS = [
    "Qwen/Qwen2.5-3B-Instruct",
    "Qwen/Qwen3-4B",
    "deepseek-ai/DeepSeek-R1",
    "LLM-Research/Meta-Llama-3-8B-Instruct",
    # ... 40+ 模型
]
```

对每个模型，脚本定义了 **14 个标准场景**：

| 场景 | 描述 | 测试内容 |
|------|------|----------|
| `basic_user` | 单条用户消息 | 基本模板渲染 |
| `system_user_assistant` | 系统+用户+助手三条消息 | 多角色消息处理 |
| `consecutive_users` | 连续两条用户消息 | 非标准消息序列 |
| `gen_prompt_true` | 启用 generation prompt | `add_generation_prompt=True` |
| `gen_prompt_false` | 禁用 generation prompt | `add_generation_prompt=False` |
| `disable_thinking` | 禁用思考模式 | Qwen3 的 `enable_thinking` 参数 |
| `tools_provided_no_call` | 提供工具但未调用 | Tool 定义注入 |
| `assistant_tool_call_history` | 助手调用工具的历史 | Tool call 序列化 |
| `tool_response_execution` | 工具调用完整流程 | Tool response 处理 |
| `parallel_tool_calls` | 并行工具调用 | 多个 tool call 和 response |
| `reasoning_content` | 推理内容（思维链） | `reasoning_content` 字段 |
| `date_injection_sim` | 日期注入模拟 | 动态日期处理 |
| `empty_assistant_content` | 空助手回复 | 边界情况 |
| `empty_user_content` | 空用户输入 | 边界情况 |

脚本对每个模型×每个场景，调用 `tokenizer.apply_chat_template()` 获取 **Python 标准输出**作为 `expected`，保存到 `test_chat_template.json`（10710 行，40+ 模型 × 14 场景 = 390+ 条测试用例）。

#### (2) `test_chat_template.json` —— 测试数据文件

JSON 结构如下：

```json
{
  "Qwen/Qwen2.5-3B-Instruct": {
    "template": "{%- if tools %}...",           // 原始 Jinja2 模板
    "special_tokens": {                         // 特殊 token
      "bos_token": "", "eos_token": "<|im_end|>", ...
    },
    "cases": [
      {
        "description": "basic_user",
        "messages": [{"role": "user", "content": "Hi"}],
        "add_generation_prompt": false,
        "expected": "<|im_start|>system\nYou are Qwen...<|im_end|>\n<|im_start|>user\nHi<|im_end|>\n"
      },
      // ... 更多用例
    ]
  }
  // ... 更多模型
}
```

#### (3) `test_main.cpp` —— C++ 测试驱动

测试驱动程序的关键逻辑：

```cpp
for (auto it = all_data.begin(); it != all_data.end(); ++it) {
    std::string model_id = it.key();
    json model_data = it.value();
    
    // 1. 提取模板和特殊 token
    std::string template_str = model_data["template"];
    json default_context = model_data["special_tokens"];
    
    // 2. 构造 jinja::Template（编译模板）
    auto chat_template = std::unique_ptr<jinja::Template>(
        new jinja::Template(template_str, default_context));
    
    // 3. 对每个用例，调用 apply_chat_template 并与 expected 比较
    for (const auto& test_case : cases) {
        std::string result = chat_template->apply_chat_template(
            test_case["messages"],
            test_case.value("add_generation_prompt", false),
            test_case.count("tools") ? test_case["tools"] : json::array()
        );
        
        // 4. 模糊日期比较（将动态日期正则替换为 {{DATE}}）
        std::string expected_norm = normalize_date(expected);
        std::string result_norm = normalize_date(result);
        
        if (result_norm == expected_norm) { /* PASS */ }
        else { /* FAIL: 输出 diff */ }
    }
}
```

核心设计思想是：**C++ 的 jinja 引擎输出 与 Python transformers 的输出逐字符精确匹配**。`normalize_date()` 函数将动态日期（如 "2025-12-16" 或 "26 Jul 2024"）统一替换为 `{{DATE}}`，避免时间相关的测试不稳定。

### 1.5 jinja.cpp 存在的必要性

1. **消除 Python 依赖**：C++ 推理引擎（如 OrinMLLM、llama.cpp）在目标设备（如 NVIDIA Orin 嵌入式平台）上通常没有完整的 Python 环境。手工硬编码每个模型的模板不可维护。

2. **格式一致性至关重要**：Chat Template 生成的 prompt 哪怕多一个换行符或空格，都可能影响 LLM 的输出质量。需要与 HuggingFace 的输出**精确一致**。

3. **模型多样性支持**：不同模型（Qwen、Llama、DeepSeek、Gemma 等）使用完全不同的模板语法。有了 jinja.cpp，只需从 `tokenizer_config.json` 读取模板字符串即可自动适配任何模型。

4. **Tool Calling 支持**：现代 LLM 支持函数调用（Function Calling / Tool Use），模板中包含 `tools`、`tool_calls`、`tool_response` 等复杂的 JSON 序列化逻辑，手工实现极其复杂。

---

## 二、jinja.cpp 核心特性的源码实现详解

### 2a. C++11 兼容性

jinja.cpp 在源码中采取了多项措施确保 C++11 兼容：

**`make_unique` polyfill**（[jinja.hpp](../../../jinja.cpp/jinja.hpp) 第114-117行）：

```cpp
// C++14 make_unique polyfill for C++11
template<typename T, typename... Args>
std::unique_ptr<T> make_unique(Args&&... args) {
    return std::unique_ptr<T>(new T(std::forward<Args>(args)...));
}
```

`std::make_unique` 是 C++14 才引入的，jinja.cpp 自行实现了这个辅助函数以兼容 C++11。

**避免结构化绑定**：整个代码中没有使用 C++17 的 `auto [key, value] = ...` 语法，而是使用传统的迭代器模式：

```cpp
for (json::const_iterator it = val.begin(); it != val.end(); ++it) {
    keys.push_back(it.key());
}
```

**避免 `if constexpr`、`std::optional` 等现代特性**：所有分支都用传统的 `if/else` 和指针/默认值处理。

**CMakeLists.txt 中明确设置 C++11**：

```cmake
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
```

这使得 jinja.cpp 可以在 GCC 4.8+、Clang 3.4+ 等旧版编译器和嵌入式工具链（如 NVIDIA Jetson 平台的交叉编译器）上编译。

### 2b. 灵活的 JSON 后端（ujson 桥接层）

`ujson.hpp`（614行）是一个**统一 JSON 抽象层**，它为 `nlohmann/json` 和 `RapidJSON` 提供完全相同的 API 接口。

**编译期后端选择**（[ujson.hpp](../../../jinja.cpp/third_party/ujson.hpp) 第17-24行）：

```cpp
#ifdef UJSON_USE_RAPIDJSON
#include "rapidjson/document.h"
#include "rapidjson/writer.h"
#include "rapidjson/stringbuffer.h"
#include "rapidjson/error/en.h"
#else
#include <nlohmann/json.hpp>
#endif
```

**统一 API 设计**：无论底层使用哪个 JSON 库，`ujson::json` 都提供完全一致的接口：

| API | nlohmann/json 实现 | RapidJSON 实现 |
|-----|-------------------|---------------|
| `json()` | `std::make_shared<nlohmann_json>()` | `std::make_shared<rapidjson::Document>()` |
| `is_string()` | `m_val->is_string()` | `m_val->IsString()` |
| `get<T>()` | `m_val->get<T>()` | `json_getter<T>::get(*this)` |
| `operator[]("key")` | `(*m_val)[key]` | `(*m_val)[key.c_str()]` |
| `contains("key")` | `m_val->contains(key)` | `m_val->HasMember(key.c_str())` |
| `parse(str)` | `nlohmann_json::parse(s)` | `doc->Parse(s.c_str())` |
| `dump()` | `m_val->dump()` | `Writer + StringBuffer` |

**RapidJSON 的 getter 特化**（处理 RapidJSON 没有模板 `get<T>()` 的问题）：

```cpp
template<> struct json_getter<bool> { 
    static bool get(const json& j) { return j.m_val->IsBool() ? j.m_val->GetBool() : false; } 
};
template<> struct json_getter<std::string> { 
    static std::string get(const json& j) { 
        return j.m_val->IsString() ? std::string(j.m_val->GetString(), j.m_val->GetStringLength()) : ""; 
    } 
};
```

**切换后端只需一个编译选项**：

```bash
cmake .. -DUJSON_USE_RAPIDJSON=ON  # 使用 RapidJSON（更快、更省内存）
cmake ..                            # 默认使用 nlohmann/json
```

### 2c. 轻量级设计

**单头文件**：整个引擎只有一个 `jinja.hpp`（2137行），项目无需链接任何动态库。

**零外部构建依赖**：`third_party/` 目录包含了所有必要的第三方头文件：

```
third_party/
├── ujson.hpp           # 自研的 JSON 桥接层
├── nlohmann/           # nlohmann/json 头文件（header-only）
└── rapidjson/          # RapidJSON 头文件（header-only）
```

**header-only 库**（CMakeLists.txt）：

```cmake
add_library(jinja INTERFACE)
target_include_directories(jinja INTERFACE . third_party)
```

`INTERFACE` 库意味着没有编译产物——只需 `#include "jinja.hpp"` 即可，不用链接任何 `.a` 或 `.so`。

### 2d. 专注 LLM：原生支持 messages/tools/add_generation_prompt

**`apply_chat_template` 方法**（[jinja.hpp](../../../jinja.cpp/jinja.hpp) 第2112-2127行）：

```cpp
inline std::string Template::apply_chat_template(
    const json& messages,
    bool add_generation_prompt,
    const json& tools,
    const json& extra_context
) const {
    json context = extra_context;
    context["messages"] = messages;                          // 注入 messages
    if (!tools.empty()) context["tools"] = tools;           // 注入 tools
    if (add_generation_prompt) context["add_generation_prompt"] = true;  // 注入标志
    return render(context);
}
```

这个方法直接模拟了 HuggingFace `tokenizer.apply_chat_template()` 的签名：
- `messages`：对话消息数组，包含 `role` 和 `content`
- `add_generation_prompt`：是否在末尾添加助手的起始标记（如 `<|im_start|>assistant\n`）
- `tools`：Tool/Function 定义数组
- `extra_context`：额外上下文变量（如 `enable_thinking`）

**`tojson` 过滤器的 LLM 特化**（[jinja.hpp](../../../jinja.cpp/jinja.hpp) 第198-220行）：

Tool 定义的 JSON 序列化需要**特定的键排序**以匹配训练数据格式：

```cpp
auto get_prio = [](const std::string& k) -> int {
    if (k == "type") return 1;
    if (k == "function") return 2;
    if (k == "name") return 3;
    if (k == "description") return 4;
    if (k == "parameters") return 5;
    if (k == "properties") return 6;
    if (k == "required") return 7;
    if (k == "enum") return 8;
    return 100;
};

std::sort(keys.begin(), keys.end(), [&](const std::string& a, const std::string& b){
    int pa = get_prio(a); int pb = get_prio(b);
    if (pa != pb) return pa < pb;
    return a < b;
});
```

标准 JSON 库（如 nlohmann/json）输出的键顺序可能是字母序或哈希序，但 LLM 期望看到 `type → function → name → description → parameters → ...` 这样的顺序。自定义的 `to_json_string()` 确保了这一点。

**`to_python_string` / `to_python_repr` 函数**：将 JSON 值转换为 Python 风格的字符串表示（`True`/`False`/`None` 而非 `true`/`false`/`null`），因为很多 LLM 的 chat template 直接在模板中使用 `{{ tool | string }}` 过滤器，期望得到 Python 风格的输出。

### 2e. 统一上下文管理

**类型别名**（[jinja.hpp](../../../jinja.cpp/jinja.hpp) 第38行）：

```cpp
namespace jinja {
    using json = ujson::json;  // jinja::json 是 ujson::json 的别名
    // ...
}
```

**上下文作用域栈**（[jinja.hpp](../../../jinja.cpp/jinja.hpp) 第600-670行）：

```cpp
class Context {
    std::vector<json> scopes;  // 作用域栈：从全局到局部

public:
    explicit Context(const json& global) {
        scopes.push_back(global);  // 第0层：全局上下文
    }

    json get(const std::string& name) {
        // 从栈顶到栈底查找变量（局部优先）
        for (auto it = scopes.rbegin(); it != scopes.rend(); ++it) {
            if (it->contains(name)) return (*it)[name];
        }
        return UNDEFINED;
    }

    void set(const std::string& name, json val) {
        scopes.back()[name] = std::move(val);  // 在当前作用域设置
    }

    void push_scope(json scope = json::object()) { scopes.push_back(std::move(scope)); }
    void pop_scope() { if (scopes.size() > 1) scopes.pop_back(); }
};
```

**渲染时上下文合并**（[jinja.hpp](../../../jinja.cpp/jinja.hpp) 第2092-2102行）：

```cpp
inline std::string Template::render(const json& context) const {
    Context ctx(m_impl->default_context);       // 全局层：default_context（含特殊token等）
    ctx.set_functions(&m_impl->functions);       // 注册自定义函数
    if (!context.empty()) {
        ctx.push_scope(context);                 // 用户层：messages, tools 等
    }
    std::string output;
    for (const auto& node : m_impl->root_nodes) {
        node->render(ctx, output);
    }
    return output;
}
```

这样，`bos_token`（来自 `default_context`）和 `messages`（来自用户 `context`）可以在同一模板中无缝使用。

### 2f. 自定义函数注入

**函数注册**（[jinja.hpp](../../../jinja.cpp/jinja.hpp) 第2104-2109行）：

```cpp
inline void Template::add_function(const std::string& name, UserFunction func) {
    m_impl->functions[name] = std::move(func);
    if (!m_impl->default_context.contains(name)) {
        m_impl->default_context[name] = "<function " + name + ">";
        // 在上下文中注册一个占位值，使 "if strftime_now is defined" 测试通过
    }
}
```

**内置函数注册**（[jinja.hpp](../../../jinja.cpp/jinja.hpp) 第2034-2082行）：

```cpp
inline void Template::register_builtins() {
    // range([start], stop, [step]) - 生成整数序列
    add_function("range", [](const std::vector<Argument>& args) -> json { ... });

    // namespace(...) - 创建可变对象（for循环内更新变量）
    add_function("namespace", [](const std::vector<Argument>& args) -> json { ... });

    // strftime_now(format) - 返回当前时间字符串
    add_function("strftime_now", [](const std::vector<Argument>& args) -> json { ... });
}
```

**函数调用执行**（`CallExpr::evaluate`，[jinja.hpp](../../../jinja.cpp/jinja.hpp) 第1217-1244行）：

```cpp
inline json CallExpr::evaluate(Context& context) {
    // 1. 优先查找用户注册的函数
    if (auto func = context.get_function(func_name)) {
        std::vector<Argument> arg_vals;
        for (auto& arg : args) {
            arg_vals.push_back({arg.first, arg.second->evaluate(context)});
        }
        return func(arg_vals);
    }

    // 2. 其次查找宏（macro）
    if (auto macro = context.get_macro(func_name)) {
        // ... 执行宏
    }
    return "";
}
```

### 2g. 健壮性：390+ 条测试用例验证

**测试数据生成流程**：

```
Python transformers.apply_chat_template() → expected 输出
                  ↓ (写入 JSON)
         test_chat_template.json (40+ 模型 × 14 场景 = 390+ 用例)
                  ↓ (C++ 读取)
         test_main.cpp: jinja::Template::apply_chat_template() → actual 输出
                  ↓ (比较)
               PASS / FAIL
```

**模糊日期匹配**（[test_main.cpp](../../../jinja.cpp/tests/test_main.cpp) 第52-60行）：

```cpp
std::string normalize_date(const std::string& input) {
    std::regex pattern1(R"(\b\d{1,2} [A-Z][a-z]+ \d{4}\b)");  // "26 Jul 2024"
    std::regex pattern2(R"(\b\d{4}-\d{2}-\d{2}\b)");           // "2025-12-16"
    std::string res = std::regex_replace(input, pattern1, "{{DATE}}");
    res = std::regex_replace(res, pattern2, "{{DATE}}");
    return res;
}
```

部分模板中包含 `strftime_now('%Y-%m-%d')` 调用，生成的日期随时间变化。测试通过正则替换将具体日期归一化为 `{{DATE}}`，确保跨时间和环境的一致性。

**覆盖的模型家族**（来自 `generate_assets.py`）：

- **Qwen 2.5**: 3B-Instruct, VL-3B-Instruct, Omni-3B, 7B-Instruct-1M, Math-7B-Instruct, QwQ-32B
- **Qwen 3**: 4B, 4B-Instruct, 4B-Thinking, VL-4B-Instruct, VL-4B-Thinking, Guard-Gen-4B, Coder-30B-A3B-Instruct, Omni-30B
- **DeepSeek**: R1-Distill-Qwen-7B, V3.2, R1
- **GLM**: GLM-4.5V, GLM-4.6V
- **Llama**: llama-2-7b, Meta-Llama-3-8B-Instruct, Llama-3.2-3B-Instruct
- **Gemma**: gemma-3-4b-it, gemma-3n-E4B-it
- **Phi**: Phi-3.5-mini, Phi-3.5-vision, phi-4, Phi-4-mini-reasoning
- **SmolLM**: SmolLM-135M-Instruct, SmolVLM-256M-Instruct, SmolLM2-135M-Instruct, SmolLM3-3B
- **其他**: Yi, Mistral, MobileLLM, HunYuan

---

## 三、jinja.cpp 如何实现 Jinja2 子集以支持 LLM 推理集成

### 3.1 "轻量级、单头文件的 C++11 Jinja2 模板引擎"

**单头文件**：整个引擎的全部代码（词法分析器、解析器、AST、解释器）都在一个 `jinja.hpp` 文件中（2137行）。使用时仅需 `#include "jinja.hpp"` 即可完成集成，**无需编译 .cpp 文件、无需链接库**。

**C++11 兼容**：如第二节 2a 所述，通过 `make_unique` polyfill、避免结构化绑定、避免 `if constexpr` 等手段确保兼容性。

### 3.2 "专为 LLM 聊天模板设计（HuggingFace 风格）"

HuggingFace 的 `tokenizer_config.json` 中的 `chat_template` 字段使用 Jinja2 语法定义如何将消息列表转换为模型 prompt。例如 Qwen 2.5 的模板：

```jinja2
{%- if tools %}
    {{- '<|im_start|>system\n' }}
    ...
{%- endif %}
{%- for message in messages %}
    {{- '<|im_start|>' + message.role + '\n' + message.content + '<|im_end|>' + '\n' }}
{%- endfor %}
{%- if add_generation_prompt %}
    {{- '<|im_start|>assistant\n' }}
{%- endif %}
```

jinja.cpp 精确实现了这些模板所需的 Jinja2 子集。

### 3.3 编译器/解释器管道架构

jinja.cpp 采用经典的**编译器管道**架构：

```
模板字符串 → [Lexer 词法分析] → Token 流 → [Parser 解析] → AST → [Interpreter 解释执行] → 输出字符串
```

#### 阶段一：词法分析（Lexer）

`Lexer` 类（[jinja.hpp](../../../jinja.cpp/jinja.hpp) 第310-510行）将模板字符串扫描为 Token 列表。

**Token 类型定义**：

```cpp
struct Token {
    enum Type {
        Text,              // 纯文本
        ExpressionStart,   // {{
        ExpressionEnd,     // }}
        BlockStart,        // {%
        BlockEnd,          // %}
        Identifier,        // 变量名、关键字
        String,            // 字符串字面量
        Number,            // 数字字面量
        Operator,          // ==, !=, +, -, |, ~, 等
        Punctuation,       // [](){}:.,
        Eof                // 结束
    };
};
```

**扫描过程**：Lexer 维护一个状态机（`State::Text / State::Expression / State::Block`）：
- **Text 状态**：扫描纯文本，检测到 `{{` 转入 Expression 状态，检测到 `{%` 转入 Block 状态，检测到 `{#...#}` 跳过注释
- **Expression/Block 状态**：识别标识符、字符串、数字、运算符等 Token

**空白控制**实现：
- `{%-` / `{{-` 前缀的 `-`：设置 `trim_prev=true`，剥离前导空白
- `-%}` / `-}}` 后缀的 `-`：设置 `trim_next=true`，剥离后继空白
- `lstrip_blocks`：自动剥离 `{%` 前同一行的缩进
- `trim_blocks`：自动移除 `%}` 后的第一个换行符

这些空白控制对 LLM prompt 至关重要——多余的空格和换行符可能影响模型性能。

#### 阶段二：解析（Parser）

`Parser` 类（[jinja.hpp](../../../jinja.cpp/jinja.hpp) 第1425-2010行）是一个**递归下降解析器**，将 Token 流构建为 AST。

**运算符优先级**（从低到高）：

```
parse_expression → parse_or → parse_and → parse_not → parse_compare → parse_add → parse_filter → parse_unary → parse_primary
```

- `parse_expression`：处理三元表达式 `a if cond else b`
- `parse_or`：处理 `or`
- `parse_and`：处理 `and`
- `parse_not`：处理 `not`
- `parse_compare`：处理 `==, !=, <, >, <=, >=, in, not in, is, is not`
- `parse_add`：处理 `+, -, ~`
- `parse_filter`：处理 `| filter`
- `parse_unary`：处理 `-expr`
- `parse_primary`：处理字面量、变量、函数调用、方法调用、下标访问、属性访问

**控制结构解析**：

- `parse_for()`：`{% for x in iterable [if filter] %}...{% endfor %}`
- `parse_if()`：`{% if cond %}...{% elif cond %}...{% else %}...{% endif %}`
- `parse_set()`：`{% set x = expr %}` 和 `{% set ns.x = expr %}`
- `parse_macro()`：`{% macro name(args) %}...{% endmacro %}`

#### 阶段三：AST（抽象语法树）

AST 由两类节点组成：

**节点（Node）层次——代表模板结构**：

| Node 类型 | 对应语法 | 功能 |
|-----------|---------|------|
| `TextNode` | 纯文本 | 直接输出 |
| `PrintNode` | `{{ expr }}` | 计算表达式并输出 |
| `ForStmt` | `{% for %}` | 循环（含 `loop` 变量：`index`, `first`, `last`, `length`） |
| `IfNode` | `{% if %}` | 条件分支（支持 `elif`, `else`） |
| `SetNode` | `{% set %}` | 变量赋值（支持属性赋值 `ns.x = val`） |
| `MacroNode` | `{% macro %}` | 宏定义 |

**表达式（Expr）层次——代表值计算**：

| Expr 类型 | 功能 | 示例 |
|-----------|------|------|
| `LiteralExpr` | 字面量 | `"hello"`, `42`, `true`, `none` |
| `VarExpr` | 变量引用 | `messages` |
| `GetAttrExpr` | 属性访问 | `message.role` |
| `GetItemExpr` | 下标访问 | `messages[0]` |
| `SliceExpr` | 切片 | `arr[1:3]` |
| `CallExpr` | 函数/宏调用 | `range(10)`, `namespace(x=1)` |
| `MethodCallExpr` | 方法调用 | `s.split(",")`, `s.strip()` |
| `FilterExpr` | 过滤器 | `val \| tojson`, `val \| length` |
| `BinaryExpr` | 二元运算 | `a + b`, `x in arr`, `a ~ b` |
| `TestExpr` | 测试 | `x is defined`, `x is not none` |
| `TernaryExpr` | 三元表达式 | `a if cond else b` |
| `ListExpr` | 列表构造 | `[1, 2, 3]` |
| `ObjectExpr` | 对象构造 | `{"key": "val"}` |

#### 阶段四：解释执行

`Template::render()` 遍历 AST 根节点，递归调用 `node->render(context, output)` 完成渲染。

**`ForStmt::render`** 的关键实现：

```cpp
void render(Context& context, std::string& out) override {
    json iter_val = iterable->evaluate(context);
    
    // 1. 过滤（for x in items if x.active）
    std::vector<json> filtered_items;
    for (const auto& item : items) {
        if (filter_expr) {
            context.push_scope(temp_scope);
            if (!is_truthy(filter_expr->evaluate(context))) continue;
            context.pop_scope();
        }
        filtered_items.push_back(item);
    }
    
    // 2. 渲染循环体
    for (const auto& item : filtered_items) {
        json loop_scope;
        loop_scope[loop_vars[0]] = item;
        
        // 注入 loop 变量
        json loop_obj;
        loop_obj["index0"] = index;
        loop_obj["index"] = index + 1;
        loop_obj["first"] = (index == 0);
        loop_obj["last"] = (index == len - 1);
        loop_obj["length"] = len;
        loop_scope["loop"] = loop_obj;
        
        context.push_scope(std::move(loop_scope));
        for (const auto& node : body) node->render(context, out);
        context.pop_scope();
    }
}
```

`loop` 变量提供了 `index`, `index0`, `first`, `last`, `length`，这些在 LLM 模板中被广泛使用（如判断是否是第一条消息、是否是最后一条消息）。

### 3.4 支持具体模型的 Jinja2 子集示例

以 **Qwen 2.5/3 模板**为例，需要以下 Jinja2 特性：

```jinja2
{%- for message in messages %}                          → ForStmt + loop 变量
    {%- if message.role == "user" %}                    → IfNode + BinaryExpr(==)
        {{- '<|im_start|>' + message.role + '\n' }}     → PrintNode + BinaryExpr(+)
    {%- elif message.role == "assistant" %}              → elif 分支
        {%- if message.content %}                       → 嵌套 if
        {%- for tool_call in message.tool_calls %}      → 嵌套 for
            {%- if tool_call.function is defined %}     → TestExpr(defined)
                {%- set tool_call = tool_call.function %} → SetNode
            {%- endif %}
            {{- tool_call.arguments | tojson }}         → FilterExpr(tojson)
        {%- endfor %}
    {%- elif message.role == "tool" %}
        {%- if loop.index0 == 0 %}                      → loop.index0
        {%- if messages[loop.index0 + 1].role != "tool" %} → GetItemExpr + 表达式索引
{%- endfor %}
```

以 **DeepSeek R1 模板**为例，额外需要：

```jinja2
{%- if message.reasoning_content is defined and message.reasoning_content %}
    {{- '<think>\n' + message.reasoning_content + '\n</think>\n\n' }}
```

以 **Qwen3 Thinking 模板**为例，需要 `extra_context` 支持：

```jinja2
{%- if enable_thinking is defined and enable_thinking is false %}
    {{- '<think>\n\n</think>\n\n' }}  {# 禁用思考模式 #}
```

jinja.cpp 实现的 Jinja2 子集精确覆盖了这些模型模板所需的全部语法。

### 3.5 "实现 C++ 环境中的无缝推理集成"

**"无缝集成"的原理**：

1. **API 对齐**：`jinja::Template::apply_chat_template()` 的签名与 HuggingFace `tokenizer.apply_chat_template()` 完全对齐
2. **输出一致**：通过 390+ 条测试用例确保 C++ 输出与 Python 输出逐字符一致
3. **零依赖集成**：只需 `#include "jinja.hpp"` + 一个 JSON 库头文件，无需修改构建系统
4. **运行时灵活性**：可以从 `tokenizer_config.json` 动态加载模板字符串，无需重新编译即可适配新模型

---

## 四、jinja.cpp 在 OrinMLLM 工程中的使用详解

### 4.1 集成方式

OrinMLLM 将 `jinja.hpp` 拷贝到 `kuiper/include/jinja.hpp`，作为项目的头文件依赖。在 `demo/inference_common.h` 中通过 `#include "jinja.hpp"` 引入。

### 4.2 Chat Template 定义

在 [demo/inference_common.h](../demo/inference_common.h) 的**第22-80行**，定义了 Qwen 系列的 Chat Template 字符串：

```cpp
static const std::string QWEN_CHAT_TEMPLATE = R"(
{%- if tools %}
    {{- '<|im_start|>system\n' }}
    {%- if messages[0]['role'] == 'system' %}
        {{- messages[0]['content'] }}
    {%- else %}
        {{- 'You are Qwen, created by Alibaba Cloud. You are a helpful assistant.' }}
    {%- endif %}
    {{- "\n\n# Tools\n\nYou may call one or more functions..." }}
    {%- for tool in tools %}
        {{- "\n" }}
        {{- tool | tojson }}
    {%- endfor %}
    {{- "\n</tools>..." }}
{%- else %}
    {%- if messages[0]['role'] == 'system' %}
        {{- '<|im_start|>system\n' + messages[0]['content'] + '<|im_end|>\n' }}
    {%- else %}
        {{- '<|im_start|>system\nYou are Qwen...<|im_end|>\n' }}
    {%- endif %}
{%- endif %}
{%- for message in messages %}
    {%- if (message.role == "user") or ... %}
        {{- '<|im_start|>' + message.role + '\n' + message.content + '<|im_end|>' + '\n' }}
    {%- elif message.role == "assistant" %}
        {{- '<|im_start|>' + message.role }}
        ...
    {%- elif message.role == "tool" %}
        ...
    {%- endif %}
{%- endfor %}
{%- if add_generation_prompt %}
    {{- '<|im_start|>assistant\n' }}
{%- endif %}
)";
```

### 4.3 核心封装函数

**`apply_chat_template` 封装**（[demo/inference_common.h](../demo/inference_common.h) 第82-85行）：

```cpp
inline std::string apply_chat_template(
    const nlohmann::json& messages, bool add_generation_prompt = true) {
    jinja::Template tpl(QWEN_CHAT_TEMPLATE);    // 每次调用都编译模板
    return tpl.apply_chat_template(messages, add_generation_prompt);
}
```

**`build_messages_json` 辅助函数**（第90-118行）：

```cpp
inline nlohmann::json build_messages_json(
    const std::string& system_prompt,
    const std::vector<std::pair<std::string, std::string>>& history,
    const std::string& user_input) {
    
    nlohmann::json messages = nlohmann::json::array();
    messages.push_back({{"role", "system"}, {"content", system_prompt}});
    for (const auto& [role, content] : history) {
        messages.push_back({{"role", role}, {"content", content}});
    }
    if (!user_input.empty()) {
        messages.push_back({{"role", "user"}, {"content", user_input}});
    }
    return messages;
}
```

### 4.4 完整推理流程中的使用

#### 多轮对话管理器（`MultiTurnConversation`）

`MultiTurnConversation` 类是 OrinMLLM 的核心对话管理器，其中 jinja 的使用贯穿始终：

**获取完整 prompt**（第466-469行）：

```cpp
std::string get_full_prompt(const std::string& user_input) const {
    nlohmann::json messages = build_messages_json(system_prompt_, history_, user_input);
    return apply_chat_template(messages, true);  // 使用 jinja 渲染
}
```

**获取历史 prompt**（第477-495行）——用于 KV cache 同步：

```cpp
std::string get_history_prompt() const {
    nlohmann::json messages = nlohmann::json::array();
    messages.push_back({{"role", "system"}, {"content", system_prompt_}});
    for (const auto& [role, content] : history_) {
        messages.push_back({{"role", role}, {"content", content}});
    }
    return apply_chat_template(messages, false);  // 不添加 generation prompt
}
```

#### 核心推理函数 `generate_response<ModelType>`

模板化推理函数（第1045-1264行）中 jinja 的使用位置：

```
用户输入 user_input
       ↓
MultiTurnConversation::get_full_prompt(user_input)
       ↓
build_messages_json(system_prompt, history, user_input)  → nlohmann::json messages
       ↓
apply_chat_template(messages, true)
       ↓
jinja::Template tpl(QWEN_CHAT_TEMPLATE)  → 编译 Jinja2 模板
tpl.apply_chat_template(messages, true)   → 渲染
       ↓
"<|im_start|>system\nYou are Qwen...<|im_end|>\n
 <|im_start|>user\n你好！<|im_end|>\n
 <|im_start|>assistant\n"
       ↓
model.encode(prompt) → tokens → prefill → decode → response
```

#### KV Cache 同步中的 Jinja 使用

在每轮对话结束后，需要重新 tokenize 完整历史以同步 cached_tokens（第1330行）：

```cpp
// 重新 tokenize 完整历史来同步 cached_tokens_
std::string history_prompt = conv.get_history_prompt();  // 再次调用 jinja 渲染
auto history_tokens = model.encode(history_prompt);
std::vector<int32_t> history_tokens_i32(history_tokens.begin(), history_tokens.end());

// 检测并填补 KV cache 间隙（末尾未填充的 token）
int32_t actual_kv_len = stats.prompt_len + stats.decode_steps;
int32_t retokenized_len = static_cast<int32_t>(history_tokens_i32.size());

if (retokenized_len > actual_kv_len) {
    int32_t gap = retokenized_len - actual_kv_len;
    // 补充 embedding + prefill 填满 KV cache
    std::vector<int> gap_tokens(history_tokens_i32.begin() + actual_kv_len,
                                history_tokens_i32.end());
    const auto& gap_embedding = model.embedding(gap_tokens);
    model.prefill(gap_embedding.input_embeddings, gap, actual_kv_len);
}

conv.update_cached_tokens(history_tokens_i32);
```

这里 Jinja 被调用了两次：
1. **第一次**（`get_full_prompt`）：生成包含用户输入和 `<|im_start|>assistant\n` 后缀的完整 prompt，用于模型推理
2. **第二次**（`get_history_prompt`）：生成不含 `assistant\n` 后缀的历史记录，用于同步 KV cache

两次调用传入不同的 `add_generation_prompt` 参数，Jinja 模板根据该变量决定是否在末尾追加 `<|im_start|>assistant\n`。

### 4.5 具体模型 Demo 的使用

**Qwen2/2.5 Demo**（[demo/main_qwen.cpp](../demo/main_qwen.cpp)）：

```cpp
#include "model/qwen2.h"
#include "inference_common.h"    // 间接引入 jinja.hpp

int main(int argc, char* argv[]) {
    inference::ModelInferConfig model_config;
    model_config.skip_tokens = {151645, 151644};  // EOS + BOS
    model_config.model_name = "Qwen2/2.5";
    
    return inference::run_model_inference<model::Qwen2Model>(
        argc, argv, "Qwen2/Qwen2.5 Model Inference...", model_config, true);
}
```

**Qwen3 Demo**（[demo/main_qwen3.cpp](../demo/main_qwen3.cpp)）：

```cpp
#include "model/qwen3.h"
#include "inference_common.h"

int main(int argc, char* argv[]) {
    inference::ModelInferConfig model_config;
    model_config.skip_tokens = {151645};  // EOS only
    model_config.remove_thinking = true;  // Qwen3 支持 <think> 思考模式
    model_config.model_name = "Qwen3";
    
    return inference::run_model_inference<model::Qwen3Model>(
        argc, argv, "Qwen3 Model Inference...", model_config, true);
}
```

所有模型 Demo 共享同一个 `QWEN_CHAT_TEMPLATE` 和 `apply_chat_template()` 函数，通过 jinja.cpp 统一处理。

### 4.6 数据流全景图

```
┌─────────────────────────────────────────────────────────────────┐
│                        OrinMLLM 推理流程                         │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  用户输入: "你好！"                                               │
│       │                                                         │
│       ▼                                                         │
│  MultiTurnConversation::add_user_message("你好！")              │
│       │                                                         │
│       ▼                                                         │
│  build_messages_json(system_prompt, history, user_input)        │
│       │  输出: [{"role":"system","content":"You are Qwen..."},   │
│       │         {"role":"user","content":"你好！"}]               │
│       ▼                                                         │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │  jinja::Template tpl(QWEN_CHAT_TEMPLATE)                │   │
│  │                                                          │   │
│  │  内部流程：                                                │   │
│  │  1. Lexer: 扫描模板 → Token 流                            │   │
│  │  2. Parser: Token 流 → AST                               │   │
│  │  3. Interpreter:                                         │   │
│  │     - 绑定 context: {messages: [...], add_generation_prompt: true} │
│  │     - 遍历 AST + 执行 for/if/set 逻辑                     │   │
│  │     - 输出: 格式化 prompt 字符串                            │   │
│  └──────────────────────────────────────────────────────────┘   │
│       │                                                         │
│       ▼                                                         │
│  prompt = "<|im_start|>system\nYou are Qwen...<|im_end|>\n     │
│            <|im_start|>user\n你好！<|im_end|>\n                  │
│            <|im_start|>assistant\n"                              │
│       │                                                         │
│       ▼                                                         │
│  model.encode(prompt) → token_ids: [151644, 8948, ...]         │
│       │                                                         │
│       ▼                                                         │
│  model.embedding(tokens) → 词向量                                │
│       │                                                         │
│       ▼                                                         │
│  model.prefill(embeddings, seq_len, start_pos)                  │
│       │                                                         │
│       ▼                                                         │
│  model.decode() → 逐 token 生成回复                              │
│       │                                                         │
│       ▼                                                         │
│  response = "我是通义千问..."                                     │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### 4.7 为什么 OrinMLLM 需要 jinja.cpp

1. **嵌入式部署**：OrinMLLM 目标平台是 NVIDIA Jetson Orin，一个 ARM64 嵌入式设备，没有完整的 Python 环境。jinja.cpp 的 C++11 兼容性和零依赖特性使其完美适配。

2. **KV Cache 一致性**：OrinMLLM 实现了复杂的 KV Cache 复用机制（增量 prefill + RadixTree PrefixCache），这要求每次 tokenize 的结果**严格确定性**。jinja.cpp 保证了相同输入总是产生完全相同的 prompt 字符串。

3. **多轮对话支持**：每轮对话都需要将完整历史重新格式化为 prompt，jinja.cpp 的 `for` 循环和 `if` 条件判断使得多轮对话消息的拼接既正确又简洁。

4. **统一接口**：Qwen2、Qwen2.5、Qwen3 共享同一个 Chat Template，OrinMLLM 只需定义一个 `QWEN_CHAT_TEMPLATE` 字符串，即可通过 jinja.cpp 自动处理所有变体。未来增加新模型时，只需更新模板字符串。

---

## 总结

| 维度 | 说明 |
|------|------|
| **工程定位** | 轻量级 C++11 Jinja2 子集解释器，专为 LLM Chat Template 设计 |
| **核心架构** | Lexer → Parser → AST → Interpreter 四阶段管道 |
| **代码规模** | jinja.hpp 2137行 + ujson.hpp 614行 = 约2750行 |
| **兼容性** | C++11、header-only、nlohmann/json 或 RapidJSON 双后端 |
| **LLM 特化** | `apply_chat_template` API、`tojson` 键排序、Python 风格输出、`loop` 变量 |
| **质量保证** | 390+ 条测试用例，覆盖 40+ 真实模型，与 Python transformers 精确对齐 |
| **在 OrinMLLM 中的角色** | 将用户消息格式化为模型 prompt，支持多轮对话和 KV cache 同步 |
