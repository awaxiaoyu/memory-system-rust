//! # Memory Core
//!
//! 记忆系统核心库 - 提供知识图谱存储和向量检索功能
//!
//! ## 架构
//!
//! ```text
//! ┌─────────────────────────────────────────────────────────┐
//! │                    MemorySystem                          │
//! ├─────────────────────────────────────────────────────────┤
//! │  ┌─────────────────┐     ┌─────────────────────────┐   │
//! │  │   LanceDB       │     │    KnowledgeGraph       │   │
//! │  │   (向量存储)     │     │    (petgraph)          │   │
//! │  └─────────────────┘     └─────────────────────────┘   │
//! │                                                         │
//! │  ┌─────────────────┐     ┌─────────────────────────┐   │
//! │  │ RetrievalService │    │   EmbeddingClient       │   │
//! │  │  (检索服务)       │    │   (嵌入服务)            │   │
//! │  └─────────────────┘     └─────────────────────────┘   │
//! └─────────────────────────────────────────────────────────┘
//! ```

pub mod error;
pub mod types;
pub mod storage;
pub mod graph;
pub mod retrieval;
pub mod embedding;
pub mod utils;

// Re-export main types
pub use error::{MemoryError, Result};
pub use types::*;

use std::sync::Arc;
use tokio::sync::RwLock;

/// 记忆系统主结构
pub struct MemorySystem {
    storage: Arc<RwLock<storage::LanceDBStorage>>,
    graph: Arc<RwLock<graph::KnowledgeGraph>>,
    retrieval: Arc<retrieval::RetrievalService>,
    embedding_client: Arc<embedding::EmbeddingClient>,
    initialized: bool,
}

impl MemorySystem {
    /// 创建新的记忆系统实例
    ///
    /// # Arguments
    /// * `db_path` - 数据库路径，为 None 时使用默认路径
    ///
    /// # Returns
    /// 新的 MemorySystem 实例
    pub fn new(db_path: Option<&str>) -> Result<Self> {
        let db_path = db_path.unwrap_or("./memory_db");

        let storage = Arc::new(RwLock::new(storage::LanceDBStorage::new(db_path)?));
        let graph = Arc::new(RwLock::new(graph::KnowledgeGraph::new()));
        let embedding_client = Arc::new(embedding::EmbeddingClient::new()?);
        let retrieval = Arc::new(retrieval::RetrievalService::new(
            storage.clone(),
            graph.clone(),
            embedding_client.clone(),
        ));

        Ok(Self {
            storage,
            graph,
            retrieval,
            embedding_client,
            initialized: false,
        })
    }

    /// 初始化记忆系统
    ///
    /// 创建数据库表、加载图结构到内存
    pub async fn initialize(&mut self) -> Result<()> {
        if self.initialized {
            return Ok(());
        }

        log::info!("正在初始化记忆系统...");

        // 初始化存储
        {
            let mut storage = self.storage.write().await;
            storage.initialize().await?;
        }

        // 从存储加载图结构
        {
            let storage = self.storage.read().await;
            let mut graph = self.graph.write().await;

            let edges = storage.get_all_edges().await?;
            log::info!("从存储加载了 {} 条边到图结构", edges.len());
            for edge in edges {
                graph.add_edge(edge);
            }
        }

        self.initialized = true;
        log::info!("记忆系统初始化完成");
        Ok(())
    }

    /// 查询相关记忆
    ///
    /// # Arguments
    /// * `params` - 查询参数
    ///
    /// # Returns
    /// 格式化的记忆上下文和元数据
    pub async fn query(&self, params: QueryParams) -> Result<QueryResult> {
        if !self.initialized {
            return Err(MemoryError::NotInitialized);
        }

        self.retrieval.retrieve(&params).await
    }

    /// 保存对话到记忆
    ///
    /// 核心流程：
    /// 1. 将消息转换为事件节点
    /// 2. 简单实体提取（关键词匹配）
    /// 3. 概念归类
    /// 4. 生成嵌入向量
    /// 5. 写入 LanceDB 和图结构
    ///
    /// # Arguments
    /// * `messages` - 对话消息列表
    pub async fn save(&self, messages: &[Message]) -> Result<()> {
        if !self.initialized {
            return Err(MemoryError::NotInitialized);
        }

        if messages.is_empty() {
            return Ok(());
        }

        log::info!("正在保存 {} 条消息到记忆", messages.len());

        // Step 1: 消息 → 事件节点
        let event_nodes = self.messages_to_events(messages);
        log::debug!("创建了 {} 个事件节点", event_nodes.len());

        // Step 2: 简单实体提取
        let (entity_nodes, entity_to_events) = self.extract_entities(messages, &event_nodes);
        log::debug!("提取了 {} 个实体节点", entity_nodes.len());

        // Step 3: 概念归类
        let (concept_nodes, concept_edges) =
            self.conceptualize_entities(&entity_nodes);
        log::debug!("创建了 {} 个概念节点", concept_nodes.len());

        // Step 4: 生成嵌入向量
        let mut all_nodes: Vec<MemoryNode> = Vec::new();
        all_nodes.extend(event_nodes);
        all_nodes.extend(entity_nodes);
        all_nodes.extend(concept_nodes);

        // 批量嵌入
        let texts: Vec<String> = all_nodes.iter().map(|n| n.content.clone()).collect();
        match self.embedding_client.embed_batch(&texts).await {
            Ok(embeddings) => {
                for (node, emb) in all_nodes.iter_mut().zip(embeddings) {
                    node.embedding = emb;
                }
            }
            Err(e) => {
                log::error!("嵌入向量生成失败: {}，节点将不带向量存储", e);
                // 不采用回退策略，错误已记录，节点以空向量存储
            }
        }

        // Step 5: 构建边（实体 ↔ 事件的参与关系）
        let mut all_edges: Vec<Edge> = Vec::new();
        for (entity_id, event_ids) in &entity_to_events {
            for event_id in event_ids {
                all_edges.push(Edge::new(
                    *entity_id,
                    *event_id,
                    graph::relation_types::PARTICIPATES_IN.to_string(),
                ));
            }
        }
        all_edges.extend(concept_edges);

        // Step 6: 写入 LanceDB
        {
            let storage = self.storage.read().await;
            storage.add_nodes(&all_nodes).await?;
            storage.add_edges(&all_edges).await?;

            // 更新概念池
            let concept_names: Vec<String> = all_nodes.iter()
                .filter(|n| n.node_type() == NodeType::Concept)
                .map(|n| n.content.clone())
                .collect();
            if !concept_names.is_empty() {
                storage.upsert_concepts(&concept_names).await?;
            }
        }

        // Step 7: 更新内存图结构
        {
            let mut graph = self.graph.write().await;
            for edge in &all_edges {
                graph.add_edge(edge.clone());
            }
        }

        log::info!(
            "保存完成: {} 个节点, {} 条边",
            all_nodes.len(),
            all_edges.len()
        );
        Ok(())
    }

    /// 检查系统是否已初始化
    pub fn is_initialized(&self) -> bool {
        self.initialized
    }

    /// 设置认证 Token
    ///
    /// 用于嵌入服务的身份验证
    pub async fn set_auth_token(&self, token: String) {
        self.embedding_client.set_auth_token(token).await;
    }

    /// 设置服务端 URL
    ///
    /// 用于更改嵌入服务的端点地址
    pub async fn set_server_url(&self, url: String) {
        self.embedding_client.set_server_url(url).await;
    }

    // ============================================
    // save() 内部工具方法
    // ============================================

    /// 将消息列表转换为事件节点
    fn messages_to_events(&self, messages: &[Message]) -> Vec<MemoryNode> {
        let now_time = utils::now_event_time();
        let mut events = Vec::new();

        // 将用户消息和助手回复配对形成对话事件
        let mut i = 0;
        while i < messages.len() {
            let msg = &messages[i];

            if msg.role == "user" {
                // 查看是否有紧跟的 assistant 回复
                let reply = messages.get(i + 1)
                    .filter(|m| m.role == "assistant");

                let event_content = if let Some(reply_msg) = reply {
                    format!("用户说：{}\n回复：{}", msg.content, reply_msg.content)
                } else {
                    format!("用户说：{}", msg.content)
                };

                let event_time = msg.timestamp
                    .map(|ts| {
                        chrono::DateTime::from_timestamp(ts, 0)
                            .map(|dt| dt.format("%Y-%m-%d-%H-%M").to_string())
                            .unwrap_or_else(|| now_time.clone())
                    })
                    .unwrap_or_else(|| now_time.clone());

                let mut event = MemoryNode::new_event(event_content, event_time);
                // 事件重要性：根据消息长度简单判断
                let msg_len = msg.content.len();
                event.importance = if msg_len > 200 { 0.8 }
                    else if msg_len > 50 { 0.6 }
                    else { 0.4 };

                events.push(event);

                if reply.is_some() {
                    i += 2;
                } else {
                    i += 1;
                }
            } else {
                // 单独的 assistant/system 消息，跳过或作为独立事件
                i += 1;
            }
        }

        events
    }

    /// 从消息中提取实体（基于关键词的简单提取）
    fn extract_entities(
        &self,
        messages: &[Message],
        event_nodes: &[MemoryNode],
    ) -> (Vec<MemoryNode>, std::collections::HashMap<uuid::Uuid, Vec<uuid::Uuid>>) {
        use std::collections::{HashMap, HashSet};

        let mut entity_map: HashMap<String, MemoryNode> = HashMap::new();
        let mut entity_to_events: HashMap<uuid::Uuid, Vec<uuid::Uuid>> = HashMap::new();
        let mut seen_entities: HashSet<String> = HashSet::new();

        for (msg_idx, msg) in messages.iter().enumerate() {
            if msg.role != "user" {
                continue;
            }

            let content = &msg.content;

            // 简单的实体抽取：按中文标点和空格分词
            let segments = Self::simple_segment(content);

            for segment in &segments {
                let trimmed = segment.trim();
                if trimmed.len() < 2 || trimmed.len() > 20 {
                    continue;
                }

                let entity_type = graph::infer_entity_type(trimmed);
                if entity_type == EntityType::Other {
                    continue; // 只提取可识别类型的实体
                }

                let key = trimmed.to_string();
                if !seen_entities.contains(&key) {
                    seen_entities.insert(key.clone());
                    let node = MemoryNode::new_entity(key.clone(), entity_type);
                    entity_map.insert(key, node);
                }

                // 关联到对应的事件节点
                if let Some(entity) = entity_map.get(segment.trim()) {
                    // 对应的事件索引（每个 user 消息可能映射到一个事件）
                    let event_idx = messages.iter().take(msg_idx + 1)
                        .filter(|m| m.role == "user")
                        .count()
                        .saturating_sub(1);

                    if let Some(event_node) = event_nodes.get(event_idx) {
                        entity_to_events
                            .entry(entity.id)
                            .or_default()
                            .push(event_node.id);
                    }
                }
            }
        }

        let entities: Vec<MemoryNode> = entity_map.into_values().collect();
        (entities, entity_to_events)
    }

    /// 将实体节点归类为概念并创建概念化边
    fn conceptualize_entities(
        &self,
        entities: &[MemoryNode],
    ) -> (Vec<MemoryNode>, Vec<Edge>) {
        use std::collections::HashMap;

        let mut concept_map: HashMap<&str, MemoryNode> = HashMap::new();
        let mut edges = Vec::new();

        for entity in entities {
            let concept_name = match entity.entity_type() {
                Some(EntityType::Person) => "人物",
                Some(EntityType::Place) => "地点",
                Some(EntityType::Time) => "时间",
                Some(EntityType::Object) => "物品",
                _ => continue,
            };

            let concept_node = concept_map
                .entry(concept_name)
                .or_insert_with(|| MemoryNode::new_concept(concept_name.to_string()));

            // 创建 entity → concept 边
            edges.push(Edge::new(
                entity.id,
                concept_node.id,
                graph::relation_types::CONCEPTUALIZED_AS.to_string(),
            ));
        }

        let concepts: Vec<MemoryNode> = concept_map.into_values().collect();
        (concepts, edges)
    }

    /// 简单中文分词（按标点和空格切分）
    fn simple_segment(text: &str) -> Vec<String> {
        let delimiters = [
            '，', '。', '！', '？', '、', '；', '：', '"', '"',
            '（', '）', '《', '》', '\n', '\r', '\t', ' ',
            ',', '.', '!', '?', ':', ';', '"', '\'', '(', ')',
        ];

        let mut result = Vec::new();
        let mut current = String::new();

        for ch in text.chars() {
            if delimiters.contains(&ch) {
                let trimmed = current.trim().to_string();
                if !trimmed.is_empty() {
                    result.push(trimmed);
                }
                current.clear();
            } else {
                current.push(ch);
            }
        }

        let trimmed = current.trim().to_string();
        if !trimmed.is_empty() {
            result.push(trimmed);
        }

        result
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_memory_system_creation() {
        let system = MemorySystem::new(Some("./test_db"));
        assert!(system.is_ok());
    }
}
