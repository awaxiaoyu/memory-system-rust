//! 检索服务模块
//!
//! 基于 HippoRAG 检索方法实现：
//! - 向量相似度检索
//! - 子图扩展（多跳邻居）
//! - 概念桥接（通过概念节点连接不同子图）
//! - 重排序

mod vector;
mod subgraph;
mod rerank;

pub use vector::*;
pub use subgraph::*;
pub use rerank::*;

use crate::error::Result;
use crate::types::*;
use crate::utils;
use crate::storage::LanceDBStorage;
use crate::graph::KnowledgeGraph;
use crate::embedding::EmbeddingClient;
use std::sync::Arc;
use std::collections::HashSet;
use tokio::sync::RwLock;

/// 检索服务
///
/// 负责从记忆图谱中检索相关记忆
pub struct RetrievalService {
    storage: Arc<RwLock<LanceDBStorage>>,
    graph: Arc<RwLock<KnowledgeGraph>>,
    embedding_client: Arc<EmbeddingClient>,
    config: RetrievalConfig,
    /// 使用 RwLock 允许通过 &self 修改（解决可变引用问题）
    custom_memory_ids: RwLock<HashSet<uuid::Uuid>>,
}

impl RetrievalService {
    /// 创建新的检索服务
    pub fn new(
        storage: Arc<RwLock<LanceDBStorage>>,
        graph: Arc<RwLock<KnowledgeGraph>>,
        embedding_client: Arc<EmbeddingClient>,
    ) -> Self {
        Self {
            storage,
            graph,
            embedding_client,
            config: RetrievalConfig::default(),
            custom_memory_ids: RwLock::new(HashSet::new()),
        }
    }

    /// 使用自定义配置创建检索服务
    pub fn with_config(
        storage: Arc<RwLock<LanceDBStorage>>,
        graph: Arc<RwLock<KnowledgeGraph>>,
        embedding_client: Arc<EmbeddingClient>,
        config: RetrievalConfig,
    ) -> Self {
        Self {
            storage,
            graph,
            embedding_client,
            config,
            custom_memory_ids: RwLock::new(HashSet::new()),
        }
    }

    /// 检索相关记忆
    ///
    /// 完整检索流程：
    /// 1. 向量化查询
    /// 2. 初始向量检索
    /// 3. 子图扩展
    /// 4. 概念桥接
    /// 5. 计算权重并排序
    /// 6. 格式化输出
    pub async fn retrieve(&self, params: &QueryParams) -> Result<QueryResult> {
        log::info!("开始检索: \"{}...\"",
            &params.user_message.chars().take(50).collect::<String>());

        // 加载自定义记忆标记（通过 RwLock 实现内部可变性）
        self.load_custom_memory_ids().await?;

        // Step 1: 向量化查询
        let query_embedding = self.embedding_client.embed(&params.user_message).await?;
        log::debug!("查询已向量化");

        // Step 2: 初始向量检索
        let top_k = params.top_k.unwrap_or(self.config.top_k);
        let initial_results = {
            let storage = self.storage.read().await;
            storage.vector_search(&query_embedding, top_k, None).await?
        };
        log::debug!("初始检索返回 {} 个节点", initial_results.len());

        if initial_results.is_empty() {
            log::info!("未找到相关记忆");
            return Ok(QueryResult {
                formatted_context: String::new(),
                count: 0,
                raw: None,
            });
        }

        let initial_nodes: Vec<MemoryNode> = initial_results.into_iter()
            .map(|(node, _)| node)
            .collect();

        // Step 3: 子图扩展
        let subgraph_nodes = self.expand_subgraph(
            &initial_nodes,
            self.config.hop_depth,
            self.config.max_subgraph_nodes,
        ).await?;
        log::debug!("子图扩展至 {} 个节点", subgraph_nodes.len());

        // Step 4: 概念桥接
        let bridged_nodes = self.find_concept_bridged_nodes(&subgraph_nodes).await?;
        log::debug!("概念桥接发现 {} 个额外节点", bridged_nodes.len());

        // Step 5: 合并所有候选节点
        let all_candidates = self.merge_and_dedupe(subgraph_nodes, bridged_nodes);
        log::debug!("总候选数: {}", all_candidates.len());

        // Step 6: 计算权重并排序
        let scored_memories = self.score_and_rank(
            all_candidates,
            &query_embedding,
            self.config.rerank_top_n,
        ).await;

        log::info!("检索完成，返回 {} 条记忆", scored_memories.len());

        // 格式化输出
        let formatted_context = self.format_memories(&scored_memories);
        let count = scored_memories.len();
        let raw = if params.include_raw.unwrap_or(false) {
            Some(scored_memories)
        } else {
            None
        };

        Ok(QueryResult {
            formatted_context,
            count,
            raw,
        })
    }

    /// 从存储加载自定义记忆标记（实际实现）
    async fn load_custom_memory_ids(&self) -> Result<()> {
        let storage = self.storage.read().await;
        let ids = storage.get_custom_memory_ids().await?;
        let mut custom_ids = self.custom_memory_ids.write().await;
        *custom_ids = ids;
        log::debug!("已加载 {} 个自定义记忆标记", custom_ids.len());
        Ok(())
    }

    /// 子图扩展（获取多跳邻居）
    /// 
    /// 优化：减少锁持有时间，使用批量查询
    async fn expand_subgraph(
        &self,
        seed_nodes: &[MemoryNode],
        hop_depth: usize,
        max_nodes: usize,
    ) -> Result<Vec<MemoryNode>> {
        let mut visited_ids = HashSet::with_capacity(max_nodes);
        let mut all_nodes = Vec::with_capacity(max_nodes);

        let mut current_layer_ids: Vec<uuid::Uuid> = seed_nodes.iter()
            .map(|n| n.id)
            .collect();

        // 添加种子节点
        for node in seed_nodes {
            if visited_ids.insert(node.id) {
                all_nodes.push(node.clone());
            }
        }

        // 逐层扩展
        for hop in 0..hop_depth {
            if all_nodes.len() >= max_nodes {
                break;
            }

            // 收集需要查询的邻居ID（减少锁持有时间）
            let neighbor_ids_to_query: Vec<uuid::Uuid> = {
                let graph = self.graph.read().await;
                let mut new_neighbors = Vec::new();
                
                for node_id in &current_layer_ids {
                    if all_nodes.len() + new_neighbors.len() >= max_nodes {
                        break;
                    }
                    
                    for neighbor_id in graph.get_neighbors(node_id, 1) {
                        if visited_ids.insert(neighbor_id) {
                            new_neighbors.push(neighbor_id);
                            if all_nodes.len() + new_neighbors.len() >= max_nodes {
                                break;
                            }
                        }
                    }
                }
                new_neighbors
            };

            if neighbor_ids_to_query.is_empty() {
                log::debug!("第 {} 跳无更多邻居", hop + 1);
                break;
            }

            // 批量查询节点（释放graph锁后再获取storage锁）
            {
                let storage = self.storage.read().await;
                // 限制批量查询数量
                let batch: Vec<uuid::Uuid> = neighbor_ids_to_query
                    .into_iter()
                    .take(max_nodes - all_nodes.len())
                    .collect();
                
                let nodes = storage.get_nodes(&batch).await?;
                all_nodes.extend(nodes);
            }

            // 准备下一层
            current_layer_ids = visited_ids.iter()
                .skip(all_nodes.len() - current_layer_ids.len())
                .copied()
                .collect();
        }

        Ok(all_nodes)
    }

    /// 通过概念节点发现相关的其他节点
    async fn find_concept_bridged_nodes(
        &self,
        existing_nodes: &[MemoryNode],
    ) -> Result<Vec<MemoryNode>> {
        let storage = self.storage.read().await;
        let graph = self.graph.read().await;

        let mut bridged_nodes = Vec::new();
        let existing_ids: HashSet<_> = existing_nodes.iter().map(|n| n.id).collect();

        let concept_node_ids: HashSet<_> = existing_nodes.iter()
            .filter(|n| n.node_type() == NodeType::Concept)
            .map(|n| n.id)
            .collect();

        if concept_node_ids.is_empty() {
            return Ok(vec![]);
        }

        let source_ids: Vec<_> = existing_nodes.iter().map(|n| n.id).collect();
        let bridged_ids = graph.find_concept_bridged(&source_ids, &concept_node_ids);

        for bridged_id in bridged_ids {
            if !existing_ids.contains(&bridged_id) {
                if let Some(node) = storage.get_node(&bridged_id).await? {
                    bridged_nodes.push(node);
                }
            }
        }

        Ok(bridged_nodes)
    }

    /// 合并并去重节点
    fn merge_and_dedupe(
        &self,
        nodes1: Vec<MemoryNode>,
        nodes2: Vec<MemoryNode>,
    ) -> Vec<MemoryNode> {
        let mut seen = HashSet::new();
        let mut result = Vec::new();

        for node in nodes1.into_iter().chain(nodes2) {
            if !seen.contains(&node.id) {
                seen.insert(node.id);
                result.push(node);
            }
        }

        result
    }

    /// 计算权重并排序
    async fn score_and_rank(
        &self,
        candidates: Vec<MemoryNode>,
        query_embedding: &[f32],
        top_n: usize,
    ) -> Vec<RetrievedMemory> {
        // 读取自定义记忆 ID 快照
        let custom_ids = self.custom_memory_ids.read().await;

        let mut scored_items: Vec<(MemoryNode, f32)> = candidates.into_iter()
            .map(|node| {
                let score = Self::calculate_node_weight(&node, query_embedding, &custom_ids);
                (node, score)
            })
            .collect();

        scored_items.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

        scored_items.into_iter()
            .take(top_n)
            .map(|(node, score)| Self::node_to_retrieved_memory(node, score))
            .collect()
    }

    /// 计算节点权重
    ///
    /// 权重计算：
    /// - 向量相似度 40%
    /// - 时间新鲜度 20%
    /// - 重要性 25% + 自定义标记加成
    /// - 访问频率 15%
    fn calculate_node_weight(
        node: &MemoryNode,
        query_embedding: &[f32],
        custom_ids: &HashSet<uuid::Uuid>,
    ) -> f32 {
        const SIMILARITY_WEIGHT: f32 = 0.4;
        const RECENCY_WEIGHT: f32 = 0.2;
        const IMPORTANCE_WEIGHT: f32 = 0.25;
        const ACCESS_FREQ_WEIGHT: f32 = 0.15;
        const CUSTOM_MARK_BONUS: f32 = 0.3;

        let similarity = if !node.embedding.is_empty() {
            cosine_similarity(&node.embedding, query_embedding)
        } else {
            0.0
        };

        let recency = utils::calculate_recency(node.updated_at);

        let importance = if custom_ids.contains(&node.id) {
            (node.importance + CUSTOM_MARK_BONUS).min(1.0)
        } else {
            node.importance
        };

        let access_freq = ((node.access_count as f32 + 1.0).ln() / 10.0).min(1.0);

        SIMILARITY_WEIGHT * similarity
            + RECENCY_WEIGHT * recency
            + IMPORTANCE_WEIGHT * importance
            + ACCESS_FREQ_WEIGHT * access_freq
    }

    /// 转换节点为检索结果
    fn node_to_retrieved_memory(node: MemoryNode, score: f32) -> RetrievedMemory {
        let time_ago = node.event_time()
            .map(utils::calculate_time_ago);
        let event_time = node.event_time().map(|s| s.to_string());

        RetrievedMemory {
            content: node.content.clone(),
            memory_type: node.node_type(),
            relevance: score,
            event_time,
            time_ago,
        }
    }

    /// 格式化记忆输出
    fn format_memories(&self, memories: &[RetrievedMemory]) -> String {
        if memories.is_empty() {
            return String::new();
        }

        let formatted: Vec<String> = memories.iter()
            .map(|m| {
                let type_label = match m.memory_type {
                    NodeType::Entity => "实体",
                    NodeType::Event => "事件",
                    NodeType::Concept => "概念",
                };

                match &m.time_ago {
                    Some(time) => format!("- [{}] {} ({})", type_label, m.content, time),
                    None => format!("- [{}] {}", type_label, m.content),
                }
            })
            .collect();

        format!("## 相关记忆\n{}", formatted.join("\n"))
    }
}
