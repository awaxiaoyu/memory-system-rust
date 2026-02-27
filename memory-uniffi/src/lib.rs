//! iOS/Android 原生模块绑定
//!
//! 使用 UniFFI 为移动端提供 Rust 后端

use memory_core::{
    QueryParams as CoreQueryParams, 
    Message as CoreMessage,
    NodeType as CoreNodeType,
};
use std::sync::Arc;
use tokio::runtime::Runtime;

// 生成 UniFFI 脚手架
uniffi::setup_scaffolding!();

// ============================================
// 全局异步运行时
// ============================================

static RUNTIME: std::sync::OnceLock<Runtime> = std::sync::OnceLock::new();

/// 初始化 Rust 运行时（必须在使用其他功能前调用）
#[uniffi::export]
pub fn init_runtime() {
    RUNTIME.get_or_init(|| {
        Runtime::new().expect("Failed to create tokio runtime")
    });
}

fn get_runtime() -> &'static Runtime {
    RUNTIME.get().expect("Runtime not initialized. Call init_runtime() first.")
}

// ============================================
// 错误类型
// ============================================

#[derive(Debug, thiserror::Error, uniffi::Error)]
pub enum MemoryError {
    #[error("存储错误: {msg}")]
    StorageError { msg: String },
    
    #[error("嵌入服务错误: {msg}")]
    EmbeddingError { msg: String },
    
    #[error("检索错误: {msg}")]
    RetrievalError { msg: String },
    
    #[error("无效输入: {msg}")]
    InvalidInput { msg: String },
    
    #[error("系统未初始化")]
    NotInitialized,
    
    #[error("未知错误: {msg}")]
    Unknown { msg: String },
}

impl From<memory_core::MemoryError> for MemoryError {
    fn from(e: memory_core::MemoryError) -> Self {
        match e {
            memory_core::MemoryError::Storage(msg) => MemoryError::StorageError { msg },
            memory_core::MemoryError::Embedding(msg) => MemoryError::EmbeddingError { msg },
            memory_core::MemoryError::Retrieval(msg) => MemoryError::RetrievalError { msg },
            memory_core::MemoryError::InvalidInput(msg) => MemoryError::InvalidInput { msg },
            memory_core::MemoryError::NotInitialized => MemoryError::NotInitialized,
            _ => MemoryError::Unknown { msg: e.to_string() },
        }
    }
}

// ============================================
// 数据结构
// ============================================

/// 查询参数
#[derive(uniffi::Record)]
pub struct QueryParams {
    /// 用户当前消息
    pub user_message: String,
    /// 最近几轮对话
    pub recent_messages: Option<Vec<Message>>,
    /// 返回记忆数量
    pub top_k: Option<u32>,
    /// 是否返回原始数据
    pub include_raw: Option<bool>,
}

/// 查询结果
#[derive(uniffi::Record)]
pub struct QueryResult {
    /// 格式化的记忆上下文
    pub formatted_context: String,
    /// 记忆条数
    pub count: u32,
    /// 原始记忆数据
    pub raw: Option<Vec<RetrievedMemory>>,
}

/// 对话消息
#[derive(Clone, uniffi::Record)]
pub struct Message {
    /// 角色
    pub role: String,
    /// 内容
    pub content: String,
    /// 时间戳
    pub timestamp: Option<i64>,
}

/// 检索到的记忆
#[derive(Clone, uniffi::Record)]
pub struct RetrievedMemory {
    /// 记忆内容
    pub content: String,
    /// 记忆类型
    pub memory_type: String,
    /// 相关度
    pub relevance: f32,
    /// 事件时间
    pub event_time: Option<String>,
    /// 距今时间
    pub time_ago: Option<String>,
}

// ============================================
// 主接口
// ============================================

/// 记忆系统
#[derive(uniffi::Object)]
pub struct MemorySystem {
    inner: std::sync::RwLock<memory_core::MemorySystem>,
}

#[uniffi::export]
impl MemorySystem {
    /// 创建新的记忆系统实例
    #[uniffi::constructor]
    pub fn new(db_path: Option<String>) -> Result<Arc<Self>, MemoryError> {
        let inner = memory_core::MemorySystem::new(db_path.as_deref())
            .map_err(|e| MemoryError::from(e))?;
        
        Ok(Arc::new(Self {
            inner: std::sync::RwLock::new(inner),
        }))
    }

    /// 初始化记忆系统
    pub fn initialize(&self) -> Result<(), MemoryError> {
        let runtime = get_runtime();
        runtime.block_on(async {
            let mut inner = self.inner.write().unwrap();
            inner.initialize().await.map_err(MemoryError::from)
        })
    }

    /// 查询相关记忆
    pub fn query(&self, params: QueryParams) -> Result<QueryResult, MemoryError> {
        let runtime = get_runtime();
        runtime.block_on(async {
            let core_params = CoreQueryParams {
                user_message: params.user_message,
                recent_messages: params.recent_messages.map(|msgs| {
                    msgs.into_iter().map(|m| CoreMessage {
                        role: m.role,
                        content: m.content,
                        timestamp: m.timestamp,
                    }).collect()
                }),
                top_k: params.top_k.map(|k| k as usize),
                include_raw: params.include_raw,
            };

            let inner = self.inner.read().unwrap();
            let result = inner.query(core_params).await
                .map_err(MemoryError::from)?;

            Ok(QueryResult {
                formatted_context: result.formatted_context,
                count: result.count as u32,
                raw: result.raw.map(|memories| {
                    memories.into_iter().map(|m| {
                        let memory_type = match m.memory_type {
                            CoreNodeType::Entity => "entity",
                            CoreNodeType::Event => "event",
                            CoreNodeType::Concept => "concept",
                        };
                        RetrievedMemory {
                            content: m.content,
                            memory_type: memory_type.to_string(),
                            relevance: m.relevance,
                            event_time: m.event_time,
                            time_ago: m.time_ago,
                        }
                    }).collect()
                }),
            })
        })
    }

    /// 保存对话到记忆
    pub fn save(&self, messages: Vec<Message>) -> Result<(), MemoryError> {
        let runtime = get_runtime();
        runtime.block_on(async {
            let core_messages: Vec<CoreMessage> = messages.into_iter()
                .map(|m| CoreMessage {
                    role: m.role,
                    content: m.content,
                    timestamp: m.timestamp,
                })
                .collect();

            let inner = self.inner.read().unwrap();
            inner.save(&core_messages).await.map_err(MemoryError::from)
        })
    }

    /// 检查是否已初始化
    pub fn is_initialized(&self) -> bool {
        let inner = self.inner.read().unwrap();
        inner.is_initialized()
    }

    /// 设置认证 Token
    pub fn set_auth_token(&self, token: String) -> Result<(), MemoryError> {
        let runtime = get_runtime();
        runtime.block_on(async {
            let inner = self.inner.read().unwrap();
            inner.set_auth_token(token).await;
            Ok(())
        })
    }

    /// 设置服务端 URL
    pub fn set_server_url(&self, url: String) -> Result<(), MemoryError> {
        let runtime = get_runtime();
        runtime.block_on(async {
            let inner = self.inner.read().unwrap();
            inner.set_server_url(url).await;
            Ok(())
        })
    }
}
