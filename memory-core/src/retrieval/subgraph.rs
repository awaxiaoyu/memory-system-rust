//! 子图扩展
//!
//! 基于 HippoRAG 的子图扩展策略
//! 从初始检索结果扩展到相关的邻居节点

use crate::types::{MemoryNode, NodeType, Edge, RetrievedMemory};
use crate::graph::KnowledgeGraph;
use crate::utils;
use std::collections::{HashMap, HashSet};
use uuid::Uuid;

/// 子图
#[derive(Debug, Clone)]
pub struct Subgraph {
    /// 节点列表
    pub nodes: Vec<MemoryNode>,
    /// 边列表
    pub edges: Vec<Edge>,
    /// 节点 ID 到节点的映射（用于快速查找）
    node_map: HashMap<Uuid, usize>,
}

impl Subgraph {
    /// 创建新的空子图
    pub fn new() -> Self {
        Self {
            nodes: Vec::new(),
            edges: Vec::new(),
            node_map: HashMap::new(),
        }
    }

    /// 从节点列表创建子图
    pub fn from_nodes(nodes: Vec<MemoryNode>) -> Self {
        let mut subgraph = Self::new();
        for node in nodes {
            subgraph.add_node(node);
        }
        subgraph
    }

    /// 添加节点
    pub fn add_node(&mut self, node: MemoryNode) {
        if !self.node_map.contains_key(&node.id) {
            let idx = self.nodes.len();
            self.node_map.insert(node.id, idx);
            self.nodes.push(node);
        }
    }

    /// 添加边
    pub fn add_edge(&mut self, edge: Edge) {
        // 只添加两端节点都在子图中的边
        if self.node_map.contains_key(&edge.source) 
            && self.node_map.contains_key(&edge.target) {
            self.edges.push(edge);
        }
    }

    /// 获取节点
    pub fn get_node(&self, id: &Uuid) -> Option<&MemoryNode> {
        self.node_map.get(id).map(|&idx| &self.nodes[idx])
    }

    /// 检查是否包含节点
    pub fn contains_node(&self, id: &Uuid) -> bool {
        self.node_map.contains_key(id)
    }

    /// 节点数量
    pub fn node_count(&self) -> usize {
        self.nodes.len()
    }

    /// 边数量
    pub fn edge_count(&self) -> usize {
        self.edges.len()
    }

    /// 按类型筛选节点
    pub fn nodes_by_type(&self, node_type: NodeType) -> Vec<&MemoryNode> {
        self.nodes
            .iter()
            .filter(|n| n.node_type() == node_type)
            .collect()
    }

    /// 获取所有节点 ID
    pub fn node_ids(&self) -> HashSet<Uuid> {
        self.node_map.keys().copied().collect()
    }

    /// 合并另一个子图
    pub fn merge(&mut self, other: Subgraph) {
        for node in other.nodes {
            self.add_node(node);
        }
        for edge in other.edges {
            self.add_edge(edge);
        }
    }
}

impl Default for Subgraph {
    fn default() -> Self {
        Self::new()
    }
}

/// 子图扩展配置
#[derive(Debug, Clone)]
pub struct SubgraphExpansionConfig {
    /// 扩展跳数
    pub hop_depth: usize,
    /// 最大节点数
    pub max_nodes: usize,
    /// 是否包含概念节点
    pub include_concepts: bool,
    /// 最小边权重阈值
    pub min_edge_weight: f32,
}

impl Default for SubgraphExpansionConfig {
    fn default() -> Self {
        Self {
            hop_depth: 2,
            max_nodes: 30,
            include_concepts: true,
            min_edge_weight: 0.1,
        }
    }
}

/// 子图扩展器
pub struct SubgraphExpander {
    config: SubgraphExpansionConfig,
}

impl SubgraphExpander {
    /// 创建新的扩展器
    pub fn new(config: SubgraphExpansionConfig) -> Self {
        Self { config }
    }

    /// 使用默认配置创建
    pub fn with_defaults() -> Self {
        Self::new(SubgraphExpansionConfig::default())
    }

    /// 从种子节点扩展子图
    /// 
    /// # Arguments
    /// * `graph` - 完整的知识图谱
    /// * `seed_nodes` - 种子节点（初始检索结果）
    /// * `node_lookup` - 节点 ID 到节点的映射
    pub fn expand(
        &self,
        graph: &KnowledgeGraph,
        seed_nodes: &[Uuid],
        node_lookup: &HashMap<Uuid, MemoryNode>,
    ) -> Subgraph {
        let mut subgraph = Subgraph::new();
        let mut to_visit: Vec<(Uuid, usize)> = seed_nodes.iter().map(|&id| (id, 0)).collect();
        let mut visited: HashSet<Uuid> = HashSet::new();

        while let Some((node_id, depth)) = to_visit.pop() {
            if visited.contains(&node_id) {
                continue;
            }
            if subgraph.node_count() >= self.config.max_nodes {
                break;
            }
            visited.insert(node_id);

            // 添加节点
            if let Some(node) = node_lookup.get(&node_id) {
                // 如果是概念节点，检查配置
                if node.node_type() == NodeType::Concept && !self.config.include_concepts {
                    continue;
                }
                subgraph.add_node(node.clone());
            }

            // 如果还没到达最大深度，继续扩展
            if depth < self.config.hop_depth {
                let neighbors = graph.get_neighbors(&node_id, 1);
                for neighbor_id in neighbors {
                    if !visited.contains(&neighbor_id) {
                        to_visit.push((neighbor_id, depth + 1));
                    }
                }
            }
        }

        // 收集子图内的边
        let mut edges_to_add = Vec::new();
        for node in &subgraph.nodes {
            let outgoing = graph.get_outgoing_edges(&node.id);
            for edge in outgoing {
                if edge.weight >= self.config.min_edge_weight 
                    && subgraph.contains_node(&edge.target) {
                    edges_to_add.push(edge.clone());
                }
            }
        }
        
        // 批量添加边
        for edge in edges_to_add {
            subgraph.add_edge(edge);
        }

        subgraph
    }

    /// 概念桥接扩展
    /// 
    /// 通过概念节点找到语义相关但不直接相连的节点
    pub fn expand_via_concepts(
        &self,
        graph: &KnowledgeGraph,
        subgraph: &mut Subgraph,
        node_lookup: &HashMap<Uuid, MemoryNode>,
    ) {
        // 找出当前子图中的所有概念节点
        let concept_ids: HashSet<Uuid> = subgraph
            .nodes_by_type(NodeType::Concept)
            .iter()
            .map(|n| n.id)
            .collect();

        if concept_ids.is_empty() {
            return;
        }

        // 当前子图中的非概念节点
        let source_ids: Vec<Uuid> = subgraph
            .nodes
            .iter()
            .filter(|n| n.node_type() != NodeType::Concept)
            .map(|n| n.id)
            .collect();

        // 通过概念节点桥接找到更多节点
        let bridged = graph.find_concept_bridged(&source_ids, &concept_ids);

        // 添加桥接找到的节点
        for bridged_id in bridged {
            if subgraph.node_count() >= self.config.max_nodes {
                break;
            }
            if !subgraph.contains_node(&bridged_id) {
                if let Some(node) = node_lookup.get(&bridged_id) {
                    subgraph.add_node(node.clone());
                }
            }
        }
    }
}

/// 子图评分：计算子图与查询的相关性
pub fn score_subgraph(
    subgraph: &Subgraph,
    query_embedding: &[f32],
) -> f32 {
    if subgraph.nodes.is_empty() {
        return 0.0;
    }

    let mut total_score = 0.0f32;
    let mut count = 0;

    for node in &subgraph.nodes {
        if !node.embedding.is_empty() {
            let similarity = crate::retrieval::vector::cosine_similarity(query_embedding, &node.embedding);
            let type_weight = match node.node_type() {
                NodeType::Event => 1.0,
                NodeType::Entity => 0.8,
                NodeType::Concept => 0.5,
            };
            total_score += similarity * type_weight * node.importance;
            count += 1;
        }
    }

    if count == 0 {
        0.0
    } else {
        total_score / count as f32
    }
}

/// 子图序列化：将子图转换为检索结果列表
pub fn subgraph_to_memories(subgraph: &Subgraph) -> Vec<RetrievedMemory> {
    subgraph
        .nodes
        .iter()
        .filter(|n| n.node_type() == NodeType::Event)
        .map(|node| RetrievedMemory {
            content: node.content.clone(),
            memory_type: node.node_type(),
            relevance: node.importance,
            event_time: node.event_time().map(|s| s.to_string()),
            time_ago: node.event_time().map(utils::calculate_time_ago),
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_subgraph_creation() {
        let mut subgraph = Subgraph::new();
        let node = MemoryNode::new_event("测试事件".to_string(), "2026-02-06-12-00".to_string());
        let node_id = node.id;
        
        subgraph.add_node(node);
        
        assert_eq!(subgraph.node_count(), 1);
        assert!(subgraph.contains_node(&node_id));
    }

    #[test]
    fn test_subgraph_merge() {
        let mut sg1 = Subgraph::new();
        let mut sg2 = Subgraph::new();
        
        sg1.add_node(MemoryNode::new_event("事件1".to_string(), "2026-01-01-00-00".to_string()));
        sg2.add_node(MemoryNode::new_event("事件2".to_string(), "2026-01-02-00-00".to_string()));
        
        sg1.merge(sg2);
        
        assert_eq!(sg1.node_count(), 2);
    }
}
