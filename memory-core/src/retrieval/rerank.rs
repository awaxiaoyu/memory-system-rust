//! 重排序
//!
//! 对检索结果进行重新排序，综合考虑多种因素

use crate::types::{MemoryNode, NodeType, RetrievedMemory};
use crate::retrieval::vector::cosine_similarity;
use crate::utils;
use std::collections::HashSet;
use uuid::Uuid;

/// 重排序配置
#[derive(Debug, Clone)]
pub struct RerankConfig {
    /// 向量相似度权重
    pub similarity_weight: f32,
    /// 重要性权重
    pub importance_weight: f32,
    /// 时间衰减权重
    pub recency_weight: f32,
    /// 访问频率权重
    pub frequency_weight: f32,
    /// 自定义记忆加分
    pub custom_memory_bonus: f32,
    /// 时间衰减率（每天）
    pub decay_rate: f32,
    /// 返回的最大结果数
    pub top_n: usize,
}

impl Default for RerankConfig {
    fn default() -> Self {
        Self {
            similarity_weight: 0.4,
            importance_weight: 0.25,
            recency_weight: 0.2,
            frequency_weight: 0.15,
            custom_memory_bonus: 0.2,
            decay_rate: 0.01,
            top_n: 5,
        }
    }
}

/// 评分后的记忆项
#[derive(Debug, Clone)]
pub struct ScoredMemory {
    /// 节点
    pub node: MemoryNode,
    /// 向量相似度分数
    pub similarity_score: f32,
    /// 最终综合分数
    pub final_score: f32,
    /// 是否为自定义记忆
    pub is_custom: bool,
}

/// 重排序器
pub struct Reranker {
    config: RerankConfig,
    custom_memory_ids: HashSet<Uuid>,
}

impl Reranker {
    /// 创建新的重排序器
    pub fn new(config: RerankConfig) -> Self {
        Self {
            config,
            custom_memory_ids: HashSet::new(),
        }
    }

    /// 使用默认配置创建
    pub fn with_defaults() -> Self {
        Self::new(RerankConfig::default())
    }

    /// 设置自定义记忆 ID 集合
    pub fn set_custom_memories(&mut self, ids: HashSet<Uuid>) {
        self.custom_memory_ids = ids;
    }

    /// 添加自定义记忆 ID
    pub fn add_custom_memory(&mut self, id: Uuid) {
        self.custom_memory_ids.insert(id);
    }

    /// 对检索结果进行重排序
    pub fn rerank(
        &self,
        query_embedding: &[f32],
        candidates: Vec<MemoryNode>,
    ) -> Vec<ScoredMemory> {
        let now = chrono::Utc::now().timestamp();
        
        let mut scored: Vec<ScoredMemory> = candidates
            .into_iter()
            .map(|node| {
                let similarity_score = if node.embedding.is_empty() {
                    0.0
                } else {
                    cosine_similarity(query_embedding, &node.embedding)
                };

                let final_score = self.calculate_final_score(&node, similarity_score, now);
                let is_custom = self.custom_memory_ids.contains(&node.id);

                ScoredMemory {
                    node,
                    similarity_score,
                    final_score,
                    is_custom,
                }
            })
            .collect();

        // 按最终分数降序排序
        scored.sort_by(|a, b| {
            b.final_score
                .partial_cmp(&a.final_score)
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        // 截取 top_n
        scored.truncate(self.config.top_n);

        scored
    }

    /// 计算最终综合分数
    fn calculate_final_score(&self, node: &MemoryNode, similarity: f32, now: i64) -> f32 {
        // 1. 向量相似度分量
        let similarity_component = similarity * self.config.similarity_weight;

        // 2. 重要性分量
        let importance_component = node.importance * self.config.importance_weight;

        // 3. 时间衰减分量
        let age_days = (now - node.created_at) as f32 / 86400.0;
        let recency_score = (-self.config.decay_rate * age_days).exp();
        let recency_component = recency_score * self.config.recency_weight;

        // 4. 访问频率分量（对数缩放）
        let frequency_score = (1.0 + node.access_count as f32).ln() / 5.0; // 归一化
        let frequency_component = frequency_score.min(1.0) * self.config.frequency_weight;

        // 5. 自定义记忆加分
        let custom_bonus = if self.custom_memory_ids.contains(&node.id) {
            self.config.custom_memory_bonus
        } else {
            0.0
        };

        // 6. 类型加权
        let type_multiplier = match node.node_type() {
            NodeType::Event => 1.0,
            NodeType::Entity => 0.9,
            NodeType::Concept => 0.7,
        };

        (similarity_component + importance_component + recency_component + frequency_component + custom_bonus)
            * type_multiplier
    }

    /// 多样性重排序
    /// 
    /// 使用 MMR (Maximal Marginal Relevance) 算法确保结果多样性
    pub fn rerank_with_diversity(
        &self,
        query_embedding: &[f32],
        candidates: Vec<MemoryNode>,
        lambda: f32,  // 相关性与多样性的权衡参数 (0-1)
    ) -> Vec<ScoredMemory> {
        if candidates.is_empty() {
            return Vec::new();
        }

        let now = chrono::Utc::now().timestamp();
        
        // 先计算所有候选的相似度
        let scored: Vec<ScoredMemory> = candidates
            .into_iter()
            .map(|node| {
                let similarity_score = if node.embedding.is_empty() {
                    0.0
                } else {
                    cosine_similarity(query_embedding, &node.embedding)
                };
                let final_score = self.calculate_final_score(&node, similarity_score, now);
                let is_custom = self.custom_memory_ids.contains(&node.id);
                
                ScoredMemory {
                    node,
                    similarity_score,
                    final_score,
                    is_custom,
                }
            })
            .collect();

        // MMR 选择
        let mut selected: Vec<ScoredMemory> = Vec::new();
        let mut remaining = scored;

        while selected.len() < self.config.top_n && !remaining.is_empty() {
            let mut best_idx = 0;
            let mut best_mmr_score = f32::MIN;

            for (i, candidate) in remaining.iter().enumerate() {
                // 计算与已选结果的最大相似度
                let max_sim_to_selected = selected
                    .iter()
                    .filter(|s| !s.node.embedding.is_empty() && !candidate.node.embedding.is_empty())
                    .map(|s| cosine_similarity(&s.node.embedding, &candidate.node.embedding))
                    .fold(0.0f32, f32::max);

                // MMR 分数 = λ * 相关性 - (1-λ) * 与已选结果的相似度
                let mmr_score = lambda * candidate.final_score - (1.0 - lambda) * max_sim_to_selected;

                if mmr_score > best_mmr_score {
                    best_mmr_score = mmr_score;
                    best_idx = i;
                }
            }

            selected.push(remaining.remove(best_idx));
        }

        selected
    }
}

/// 将评分后的记忆转换为检索结果
pub fn scored_to_retrieved(scored: &[ScoredMemory]) -> Vec<RetrievedMemory> {
    scored
        .iter()
        .map(|s| RetrievedMemory {
            content: s.node.content.clone(),
            memory_type: s.node.node_type(),
            relevance: s.final_score,
            event_time: s.node.event_time().map(|s| s.to_string()),
            time_ago: s.node.event_time().map(utils::calculate_time_ago),
        })
        .collect()
}

/// 去重：移除内容重复的记忆
pub fn deduplicate(memories: Vec<ScoredMemory>, threshold: f32) -> Vec<ScoredMemory> {
    let mut result: Vec<ScoredMemory> = Vec::new();

    for memory in memories {
        let is_duplicate = result.iter().any(|existing| {
            if existing.node.embedding.is_empty() || memory.node.embedding.is_empty() {
                return existing.node.content == memory.node.content;
            }
            cosine_similarity(&existing.node.embedding, &memory.node.embedding) > threshold
        });

        if !is_duplicate {
            result.push(memory);
        }
    }

    result
}

/// 按时间排序
pub fn sort_by_time(memories: &mut [ScoredMemory], ascending: bool) {
    memories.sort_by(|a, b| {
        let time_a = a.node.event_time().unwrap_or("");
        let time_b = b.node.event_time().unwrap_or("");
        
        if ascending {
            time_a.cmp(time_b)
        } else {
            time_b.cmp(time_a)
        }
    });
}

/// 分组：按事件时间分组记忆
pub fn group_by_date(memories: &[ScoredMemory]) -> Vec<(String, Vec<&ScoredMemory>)> {
    use std::collections::BTreeMap;
    
    let mut groups: BTreeMap<String, Vec<&ScoredMemory>> = BTreeMap::new();

    for memory in memories {
        let date = memory
            .node
            .event_time()
            .map(utils::format_date)
            .unwrap_or_else(|| "未知日期".to_string());

        groups.entry(date).or_insert_with(Vec::new).push(memory);
    }

    groups.into_iter().collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_reranker_basic() {
        let config = RerankConfig::default();
        let reranker = Reranker::new(config);

        let node = MemoryNode::new_event("测试事件".to_string(), "2026-02-06-12-00".to_string());
        let query = vec![0.0f32; 1024];
        
        let results = reranker.rerank(&query, vec![node]);
        
        assert_eq!(results.len(), 1);
    }

    #[test]
    fn test_deduplicate() {
        let node1 = MemoryNode::new_event("事件A".to_string(), "2026-01-01-00-00".to_string());
        let mut node2 = MemoryNode::new_event("事件A".to_string(), "2026-01-01-00-00".to_string());
        node2.id = uuid::Uuid::new_v4(); // 不同 ID 但相同内容

        let scored = vec![
            ScoredMemory {
                node: node1,
                similarity_score: 0.9,
                final_score: 0.8,
                is_custom: false,
            },
            ScoredMemory {
                node: node2,
                similarity_score: 0.9,
                final_score: 0.7,
                is_custom: false,
            },
        ];

        // 使用高阈值时应该去重（内容相同，应该被去重）
        let deduped = deduplicate(scored, 0.95);
        assert_eq!(deduped.len(), 1); // 内容相同，应该被去重为1个
    }

    #[test]
    fn test_group_by_date() {
        let node1 = MemoryNode::new_event("事件1".to_string(), "2026-01-01-10-00".to_string());
        let node2 = MemoryNode::new_event("事件2".to_string(), "2026-01-01-15-00".to_string());
        let node3 = MemoryNode::new_event("事件3".to_string(), "2026-01-02-10-00".to_string());

        let scored: Vec<ScoredMemory> = vec![node1, node2, node3]
            .into_iter()
            .map(|n| ScoredMemory {
                node: n,
                similarity_score: 0.5,
                final_score: 0.5,
                is_custom: false,
            })
            .collect();

        let groups = group_by_date(&scored);
        
        assert_eq!(groups.len(), 2);
        assert!(groups.iter().any(|(date, _)| date == "2026-01-01"));
        assert!(groups.iter().any(|(date, _)| date == "2026-01-02"));
    }
}
