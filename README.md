# Memory System - Rust 记忆系统

一个基于知识图谱和向量存储的智能记忆系统，支持多平台多语言绑定。

## 架构

```
┌─────────────────────────────────────────────────────────┐
│                    MemorySystem                          │
├─────────────────────────────────────────────────────────┤
│  ┌─────────────────┐     ┌─────────────────────────┐   │
│  │   LanceDB       │     │    KnowledgeGraph       │   │
│  │   (向量存储)     │     │    (petgraph)          │   │
│  └─────────────────┘     └─────────────────────────┘   │
│                                                         │
│  ┌─────────────────┐     ┌─────────────────────────┐   │
│  │ RetrievalService │    │   EmbeddingClient       │   │
│  │  (检索服务)       │    │   (嵌入服务)            │   │
│  └─────────────────┘     └─────────────────────────┘   │
└─────────────────────────────────────────────────────────┘
```

## 项目结构

```
rust/
├── memory-core/          # 核心库
│   ├── src/
│   │   ├── lib.rs        # 主入口
│   │   ├── types.rs      # 类型定义
│   │   ├── error.rs      # 错误处理
│   │   ├── storage/      # 存储层 (LanceDB)
│   │   ├── graph/        # 知识图谱 (petgraph)
│   │   ├── retrieval/    # 检索服务
│   │   ├── embedding/    # 嵌入服务
│   │   └── utils.rs      # 工具函数
│   └── Cargo.toml
├── memory-napi/          # Node.js 绑定
│   └── src/lib.rs
├── memory-uniffi/        # iOS/Android 绑定
│   └── src/lib.rs
└── examples/
    └── cli/              # CLI 示例
```

## 功能特性

- **知识图谱存储**: 基于 petgraph 实现实体-事件-概念三层架构
- **向量检索**: 基于 LanceDB 的高效向量存储和检索
- **HippoRAG 检索**: 向量相似度 + 子图扩展 + 概念桥接 + 重排序
- **多平台支持**:
  - Windows/桌面端: Node.js (napi-rs)
  - iOS/Android: UniFFI
- **嵌入服务**: 通过服务端 API 调用 bge-m3 模型

## 快速开始

### 构建

```bash
# 构建整个工作区
cargo build

# 运行测试
cargo test

# 构建发布版本
cargo build --release
```

### CLI 示例

```bash
# 进入示例目录
cd examples/cli

# 初始化记忆系统
cargo run -- init

# 保存对话
cargo run -- save --content '[{"role":"user","content":"你好","timestamp":1700000000}]'

# 查询记忆
cargo run -- query "你好"

# 交互式对话模式
cargo run -- chat
```

### Rust API 使用

```rust
use memory_core::{MemorySystem, Message, QueryParams};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // 创建记忆系统
    let mut system = MemorySystem::new(Some("./memory_db"))?;
    
    // 初始化
    system.initialize().await?;
    
    // 设置认证（如果需要）
    system.set_auth_token("your-token".to_string()).await;
    
    // 保存对话
    let messages = vec![
        Message {
            role: "user".to_string(),
            content: "你好".to_string(),
            timestamp: Some(chrono::Utc::now().timestamp()),
        },
        Message {
            role: "assistant".to_string(),
            content: "你好！有什么可以帮助你的？".to_string(),
            timestamp: Some(chrono::Utc::now().timestamp()),
        },
    ];
    system.save(&messages).await?;
    
    // 查询记忆
    let params = QueryParams {
        user_message: "你好".to_string(),
        recent_messages: None,
        top_k: Some(5),
        include_raw: Some(true),
    };
    let result = system.query(params).await?;
    println!("{}", result.formatted_context);
    
    Ok(())
}
```

### Node.js 使用

```javascript
const { MemorySystem } = require('./memory-napi');

async function main() {
    // 创建记忆系统
    const system = new MemorySystem('./memory_db');
    
    // 初始化
    await system.initialize();
    
    // 设置认证
    await system.setAuthToken('your-token');
    
    // 保存对话
    await system.save([
        { role: 'user', content: '你好', timestamp: Date.now() / 1000 },
        { role: 'assistant', content: '你好！', timestamp: Date.now() / 1000 },
    ]);
    
    // 查询
    const result = await system.query({
        userMessage: '你好',
        topK: 5,
    });
    console.log(result.formattedContext);
}
```

## 配置

### 嵌入服务配置

在 `memory-core/src/embedding/client.rs` 中修改默认配置：

```rust
impl Default for ServerConfig {
    fn default() -> Self {
        Self {
            server_url: "https://your-server.com/api".to_string(),
            auth_token: None,
            talk_endpoint: "/talk".to_string(),
            embedding_endpoint: "/v1/embeddings".to_string(),
        }
    }
}
```

### 检索配置

```rust
use memory_core::RetrievalConfig;

let config = RetrievalConfig {
    top_k: 10,              // 初始检索数量
    hop_depth: 2,           // 子图扩展深度
    max_subgraph_nodes: 30, // 子图最大节点数
    rerank_top_n: 5,        // 重排序后返回数量
};
```

## 性能优化

- 向量计算使用 SIMD 优化的迭代器
- 子图扩展减少锁持有时间
- 批量查询节点减少数据库往返
- 预分配集合容量减少内存分配

## 测试

```bash
# 运行所有测试
cargo test

# 运行特定包测试
cargo test --package memory-core

# 运行带日志的测试
RUST_LOG=debug cargo test -- --nocapture
```

## 许可证

MIT
