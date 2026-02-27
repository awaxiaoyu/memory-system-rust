//! Node.js 原生模块绑定
//!
//! 使用 napi-rs 为 Windows/桌面端提供 Rust 后端

#![deny(clippy::all)]

use napi::bindgen_prelude::*;
use napi_derive::napi;
use memory_core::{QueryParams as CoreQueryParams, Message as CoreMessage};
use std::sync::Arc;
use tokio::sync::RwLock;

/// JavaScript 可用的记忆系统
#[napi]
pub struct MemorySystem {
    inner: Arc<RwLock<memory_core::MemorySystem>>,
}

#[napi]
impl MemorySystem {
    /// 创建新的记忆系统实例
    #[napi(constructor)]
    pub fn new(db_path: Option<String>) -> Result<Self> {
        let inner = memory_core::MemorySystem::new(db_path.as_deref())
            .map_err(|e| Error::from_reason(e.to_string()))?;
        Ok(Self {
            inner: Arc::new(RwLock::new(inner)),
        })
    }

    /// 初始化记忆系统
    #[napi]
    pub async fn initialize(&self) -> Result<()> {
        let mut inner = self.inner.write().await;
        inner.initialize().await
            .map_err(|e| Error::from_reason(e.to_string()))
    }

    /// 查询相关记忆
    #[napi]
    pub async fn query(&self, params: JsQueryParams) -> Result<JsQueryResult> {
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

        let inner = self.inner.read().await;
        let result = inner.query(core_params).await
            .map_err(|e| Error::from_reason(e.to_string()))?;

        Ok(JsQueryResult {
            formatted_context: result.formatted_context,
            count: result.count as u32,
            raw: result.raw.map(|memories| {
                memories.into_iter().map(|m| JsRetrievedMemory {
                    content: m.content,
                    memory_type: format!("{:?}", m.memory_type).to_lowercase(),
                    relevance: m.relevance as f64,
                    event_time: m.event_time,
                    time_ago: m.time_ago,
                }).collect()
            }),
        })
    }

    /// 保存对话到记忆
    #[napi]
    pub async fn save(&self, messages: Vec<JsMessage>) -> Result<()> {
        let core_messages: Vec<CoreMessage> = messages.into_iter()
            .map(|m| CoreMessage {
                role: m.role,
                content: m.content,
                timestamp: m.timestamp,
            })
            .collect();

        let inner = self.inner.read().await;
        inner.save(&core_messages).await
            .map_err(|e| Error::from_reason(e.to_string()))
    }

    /// 设置认证 Token
    #[napi]
    pub async fn set_auth_token(&self, token: String) -> Result<()> {
        let inner = self.inner.read().await;
        inner.set_auth_token(token).await;
        Ok(())
    }

    /// 设置服务端 URL
    #[napi]
    pub async fn set_server_url(&self, url: String) -> Result<()> {
        let inner = self.inner.read().await;
        inner.set_server_url(url).await;
        Ok(())
    }

    /// 检查是否已初始化
    #[napi]
    pub async fn is_initialized(&self) -> bool {
        let inner = self.inner.read().await;
        inner.is_initialized()
    }
}

/// 查询参数
#[napi(object)]
pub struct JsQueryParams {
    /// 用户当前消息
    pub user_message: String,
    /// 最近几轮对话
    pub recent_messages: Option<Vec<JsMessage>>,
    /// 返回记忆数量
    pub top_k: Option<u32>,
    /// 是否返回原始数据
    pub include_raw: Option<bool>,
}

/// 对话消息
#[napi(object)]
#[derive(Clone)]
pub struct JsMessage {
    /// 角色
    pub role: String,
    /// 内容
    pub content: String,
    /// 时间戳
    pub timestamp: Option<i64>,
}

/// 查询结果
#[napi(object)]
pub struct JsQueryResult {
    /// 格式化的记忆上下文
    pub formatted_context: String,
    /// 记忆条数
    pub count: u32,
    /// 原始记忆数据
    pub raw: Option<Vec<JsRetrievedMemory>>,
}

/// 检索到的记忆
#[napi(object)]
pub struct JsRetrievedMemory {
    /// 记忆内容
    pub content: String,
    /// 记忆类型
    pub memory_type: String,
    /// 相关度
    pub relevance: f64,
    /// 事件时间
    pub event_time: Option<String>,
    /// 距今时间
    pub time_ago: Option<String>,
}
