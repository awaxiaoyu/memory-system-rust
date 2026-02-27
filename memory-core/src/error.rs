//! 错误类型定义

use thiserror::Error;

/// 记忆系统错误类型
#[derive(Error, Debug)]
pub enum MemoryError {
    /// 存储层错误
    #[error("Storage error: {0}")]
    Storage(String),

    /// 嵌入服务错误
    #[error("Embedding error: {0}")]
    Embedding(String),

    /// 检索错误
    #[error("Retrieval error: {0}")]
    Retrieval(String),

    /// 图操作错误
    #[error("Graph error: {0}")]
    Graph(String),

    /// 无效输入
    #[error("Invalid input: {0}")]
    InvalidInput(String),

    /// 系统未初始化
    #[error("Memory system not initialized")]
    NotInitialized,

    /// 序列化错误
    #[error("Serialization error: {0}")]
    Serialization(#[from] serde_json::Error),

    /// IO 错误
    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),

    /// HTTP 请求错误
    #[error("HTTP error: {0}")]
    Http(#[from] reqwest::Error),

    /// 未知错误
    #[error("Unknown error: {0}")]
    Unknown(String),
}

/// 结果类型别名
pub type Result<T> = std::result::Result<T, MemoryError>;

impl From<anyhow::Error> for MemoryError {
    fn from(err: anyhow::Error) -> Self {
        MemoryError::Unknown(err.to_string())
    }
}
