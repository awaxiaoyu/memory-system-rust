//! Embedding API 客户端
//!
//! 通过服务端转发调用 bge-m3 向量嵌入 API
//! API Key 不在客户端存储，由服务端统一管理

use crate::error::{MemoryError, Result};
use serde::{Deserialize, Serialize};
use std::time::Duration;

/// 服务端配置
/// 
/// 对应 TS 版本的 SERVER_PROXY_CONFIG
#[derive(Debug, Clone)]
pub struct ServerConfig {
    /// 服务端 URL（默认：http://101.200.3.32:3000/api）
    pub server_url: String,
    /// 认证 Token
    pub auth_token: Option<String>,
    /// 对话模型端点
    pub talk_endpoint: String,
    /// 嵌入模型端点
    pub embedding_endpoint: String,
}

impl Default for ServerConfig {
    fn default() -> Self {
        Self {
            server_url: "https://www.yukiwithyou.asia/api".to_string(),
            auth_token: None,
            talk_endpoint: "/talk".to_string(),
            embedding_endpoint: "/v1/embeddings".to_string(),
        }
    }
}

/// Embedding 配置
/// 
/// 对应 TS 版本的 EMBEDDING_CONFIG
#[derive(Debug, Clone)]
pub struct EmbeddingConfig {
    /// 服务端嵌入模型名称（格式: provider-modelName）
    pub server_model: String,
    /// 向量维度
    pub dimension: usize,
    /// 批量处理大小
    pub batch_size: usize,
    /// 请求超时时间（毫秒）
    pub timeout_ms: u64,
    /// 重试次数
    pub max_retries: u32,
    /// 重试间隔（毫秒）
    pub retry_delay_ms: u64,
}

impl Default for EmbeddingConfig {
    fn default() -> Self {
        Self {
            server_model: "siliconflow-BAAI/bge-m3".to_string(),
            dimension: 1024,
            batch_size: 32,
            timeout_ms: 30000,
            max_retries: 3,
            retry_delay_ms: 1000,
        }
    }
}

/// OpenAI 兼容的嵌入请求格式
#[derive(Debug, Serialize)]
struct EmbeddingRequest {
    input: serde_json::Value,  // 可以是 String 或 Vec<String>
    model: String,
    encoding_format: String,
}

/// OpenAI 兼容的嵌入响应格式
#[derive(Debug, Deserialize)]
#[allow(dead_code)]
struct EmbeddingResponse {
    object: String,
    data: Vec<EmbeddingData>,
    model: String,
    usage: EmbeddingUsage,
}

#[derive(Debug, Deserialize)]
#[allow(dead_code)]
struct EmbeddingData {
    object: String,
    index: usize,
    embedding: Vec<f32>,
}

#[derive(Debug, Deserialize)]
#[allow(dead_code)]
struct EmbeddingUsage {
    prompt_tokens: u32,
    total_tokens: u32,
}

/// Embedding 客户端
/// 
/// 负责将文本转换为向量表示，通过服务端 API 转发
pub struct EmbeddingClient {
    client: reqwest::Client,
    server_config: tokio::sync::RwLock<ServerConfig>,
    embedding_config: EmbeddingConfig,
}

impl EmbeddingClient {
    /// 创建新的客户端实例
    pub fn new() -> Result<Self> {
        Self::with_config(ServerConfig::default(), EmbeddingConfig::default())
    }

    /// 使用自定义配置创建客户端
    pub fn with_config(
        server_config: ServerConfig,
        embedding_config: EmbeddingConfig,
    ) -> Result<Self> {
        let client = reqwest::Client::builder()
            .timeout(Duration::from_millis(embedding_config.timeout_ms))
            .build()
            .map_err(|e| MemoryError::Embedding(e.to_string()))?;

        Ok(Self {
            client,
            server_config: tokio::sync::RwLock::new(server_config),
            embedding_config,
        })
    }

    /// 设置认证 Token
    pub async fn set_auth_token(&self, token: String) {
        let mut config = self.server_config.write().await;
        config.auth_token = Some(token);
        log::info!("认证 Token 已设置");
    }

    /// 设置服务端 URL
    pub async fn set_server_url(&self, url: String) {
        let mut config = self.server_config.write().await;
        config.server_url = url;
        log::info!("服务端 URL 已设置");
    }

    /// 获取单个文本的嵌入向量
    pub async fn embed(&self, text: &str) -> Result<Vec<f32>> {
        let embeddings = self.embed_batch(&[text.to_string()]).await?;
        embeddings.into_iter().next()
            .ok_or_else(|| MemoryError::Embedding("Empty embedding result".to_string()))
    }

    /// 批量获取文本的嵌入向量
    pub async fn embed_batch(&self, texts: &[String]) -> Result<Vec<Vec<f32>>> {
        if texts.is_empty() {
            return Ok(vec![]);
        }

        // 过滤空文本
        let valid_texts: Vec<&String> = texts.iter()
            .filter(|t| !t.trim().is_empty())
            .collect();

        if valid_texts.is_empty() {
            return Err(MemoryError::InvalidInput(
                "All input texts are empty".to_string()
            ));
        }

        log::debug!("Embedding {} texts", valid_texts.len());

        // 如果超过批量大小，分批处理
        if valid_texts.len() > self.embedding_config.batch_size {
            return self.embed_large_batch(valid_texts).await;
        }

        self.call_embedding_api(valid_texts).await
    }

    /// 处理大批量文本（分批调用 API）
    async fn embed_large_batch(&self, texts: Vec<&String>) -> Result<Vec<Vec<f32>>> {
        let mut results = Vec::new();
        let batch_size = self.embedding_config.batch_size;
        let total_batches = (texts.len() + batch_size - 1) / batch_size;

        log::info!("Processing {} texts in {} batches", texts.len(), total_batches);

        for (i, chunk) in texts.chunks(batch_size).enumerate() {
            log::debug!("Processing batch {}/{} ({} texts)", i + 1, total_batches, chunk.len());
            
            let batch_results = self.call_embedding_api(chunk.to_vec()).await?;
            results.extend(batch_results);
        }

        Ok(results)
    }

    /// 调用服务端嵌入 API
    async fn call_embedding_api(&self, texts: Vec<&String>) -> Result<Vec<Vec<f32>>> {
        let (auth_token, endpoint) = {
            let config = self.server_config.read().await;
            let auth_token = config.auth_token.clone()
                .ok_or_else(|| MemoryError::Embedding(
                    "未登录账号系统，请先登录".to_string()
                ))?;
            let endpoint = format!(
                "{}{}",
                config.server_url,
                config.embedding_endpoint
            );
            (auth_token, endpoint)
        };

        let input = if texts.len() == 1 {
            serde_json::json!(texts[0])
        } else {
            serde_json::json!(texts)
        };

        let request_body = EmbeddingRequest {
            input,
            model: self.embedding_config.server_model.clone(),
            encoding_format: "float".to_string(),
        };

        log::debug!("Calling embedding API: {}", endpoint);

        // 带重试的请求
        let mut last_error = None;
        for attempt in 0..=self.embedding_config.max_retries {
            if attempt > 0 {
                log::debug!("Retry attempt {}/{}", attempt, self.embedding_config.max_retries);
                tokio::time::sleep(Duration::from_millis(
                    self.embedding_config.retry_delay_ms
                )).await;
            }

            match self.send_request(&endpoint, &auth_token, &request_body).await {
                Ok(embeddings) => return Ok(embeddings),
                Err(e) => {
                    log::warn!("Embedding API error (attempt {}): {}", attempt + 1, e);
                    last_error = Some(e);
                }
            }
        }

        Err(last_error.unwrap_or_else(|| MemoryError::Embedding("Unknown error".to_string())))
    }

    /// 发送单次请求
    async fn send_request(
        &self,
        endpoint: &str,
        auth_token: &str,
        request_body: &EmbeddingRequest,
    ) -> Result<Vec<Vec<f32>>> {
        let response = self.client
            .post(endpoint)
            .header("Authorization", format!("Bearer {}", auth_token))
            .header("Content-Type", "application/json")
            .json(request_body)
            .send()
            .await
            .map_err(|e| MemoryError::Embedding(format!("请求失败: {}", e)))?;

        if !response.status().is_success() {
            let status = response.status();
            let error_text = response.text().await.unwrap_or_default();
            return Err(MemoryError::Embedding(
                format!("嵌入 API 调用失败: {} - {}", status, error_text)
            ));
        }

        let data: EmbeddingResponse = response.json().await
            .map_err(|e| MemoryError::Embedding(format!("解析响应失败: {}", e)))?;

        // 验证响应格式
        if data.data.is_empty() {
            return Err(MemoryError::Embedding("嵌入 API 返回空数据".to_string()));
        }

        // 按 index 排序并提取嵌入向量
        let mut sorted_data = data.data;
        sorted_data.sort_by_key(|item| item.index);

        let embeddings: Vec<Vec<f32>> = sorted_data.into_iter()
            .map(|item| item.embedding)
            .collect();

        // 验证向量维度
        for (i, embedding) in embeddings.iter().enumerate() {
            if embedding.len() != self.embedding_config.dimension {
                log::warn!(
                    "Embedding dimension mismatch: expected {}, got {} (index {})",
                    self.embedding_config.dimension,
                    embedding.len(),
                    i
                );
            }
        }

        log::debug!(
            "Embedding API success: {} vectors, dimension {}",
            embeddings.len(),
            embeddings.first().map(|v| v.len()).unwrap_or(0)
        );

        Ok(embeddings)
    }

    /// 获取当前配置（克隆返回）
    pub async fn get_config(&self) -> (ServerConfig, EmbeddingConfig) {
        let server_config = self.server_config.read().await.clone();
        (server_config, self.embedding_config.clone())
    }
}

impl Default for EmbeddingClient {
    fn default() -> Self {
        Self::new().expect("Failed to create default EmbeddingClient")
    }
}
