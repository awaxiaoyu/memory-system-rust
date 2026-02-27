//! 知识图谱模块

mod nodes;
mod edges;
mod traversal;

pub use nodes::*;
pub use edges::*;
pub use traversal::*;

use crate::types::*;
use petgraph::graph::{DiGraph, NodeIndex};
use std::collections::{HashMap, HashSet};
use uuid::Uuid;

/// 知识图谱
/// 
/// 使用 petgraph 实现的有向图结构
pub struct KnowledgeGraph {
    /// 内部图结构（节点存储 UUID，边存储 Edge）
    graph: DiGraph<Uuid, Edge>,
    /// UUID 到节点索引的映射
    id_to_index: HashMap<Uuid, NodeIndex>,
}

impl KnowledgeGraph {
    /// 创建新的空图谱
    pub fn new() -> Self {
        Self {
            graph: DiGraph::new(),
            id_to_index: HashMap::new(),
        }
    }

    /// 添加节点到图中
    pub fn add_node(&mut self, id: Uuid) -> NodeIndex {
        if let Some(&idx) = self.id_to_index.get(&id) {
            return idx;
        }
        let idx = self.graph.add_node(id);
        self.id_to_index.insert(id, idx);
        idx
    }

    /// 添加边到图中
    pub fn add_edge(&mut self, edge: Edge) {
        let source_idx = self.add_node(edge.source);
        let target_idx = self.add_node(edge.target);
        self.graph.add_edge(source_idx, target_idx, edge);
    }

    /// 批量添加边
    pub fn add_edges(&mut self, edges: impl IntoIterator<Item = Edge>) {
        for edge in edges {
            self.add_edge(edge);
        }
    }

    /// 删除节点及其所有边
    ///
    /// 注意：petgraph 的 remove_node 会导致 NodeIndex 重新编号，
    /// 因此必须重建 id_to_index 映射
    pub fn remove_node(&mut self, id: &Uuid) -> bool {
        if let Some(idx) = self.id_to_index.remove(id) {
            self.graph.remove_node(idx);
            // 重建 id_to_index 映射，因为 petgraph 会重新编号
            self.rebuild_index();
            return true;
        }
        false
    }

    /// 重建 UUID → NodeIndex 映射
    fn rebuild_index(&mut self) {
        self.id_to_index.clear();
        for idx in self.graph.node_indices() {
            if let Some(uuid) = self.graph.node_weight(idx) {
                self.id_to_index.insert(*uuid, idx);
            }
        }
    }

    /// 检查节点是否存在
    pub fn contains_node(&self, id: &Uuid) -> bool {
        self.id_to_index.contains_key(id)
    }

    /// 获取节点数量
    pub fn node_count(&self) -> usize {
        self.graph.node_count()
    }

    /// 获取边数量
    pub fn edge_count(&self) -> usize {
        self.graph.edge_count()
    }

    /// 获取 N 跳邻居
    /// 
    /// # Arguments
    /// * `node_id` - 起始节点 ID
    /// * `hops` - 跳数
    /// 
    /// # Returns
    /// 所有在 N 跳范围内的节点 ID 集合
    pub fn get_neighbors(&self, node_id: &Uuid, hops: usize) -> HashSet<Uuid> {
        let mut visited = HashSet::new();
        let mut current_level = HashSet::new();
        
        if let Some(&start_idx) = self.id_to_index.get(node_id) {
            current_level.insert(start_idx);
            visited.insert(*node_id);
            
            for _ in 0..hops {
                let mut next_level = HashSet::new();
                
                for idx in &current_level {
                    // 遍历所有邻居（包括出边和入边方向）
                    for neighbor in self.graph.neighbors_undirected(*idx) {
                        if let Some(neighbor_id) = self.graph.node_weight(neighbor) {
                            if !visited.contains(neighbor_id) {
                                visited.insert(*neighbor_id);
                                next_level.insert(neighbor);
                            }
                        }
                    }
                }
                
                if next_level.is_empty() {
                    break;
                }
                current_level = next_level;
            }
        }
        
        visited
    }

    /// 查找概念桥接的节点
    /// 
    /// 通过概念节点找到相关的其他节点
    /// 
    /// # Arguments
    /// * `source_nodes` - 源节点 ID 列表
    /// * `concept_nodes` - 概念节点 ID 集合
    /// 
    /// # Returns
    /// 通过概念节点桥接找到的节点 ID 集合
    pub fn find_concept_bridged(
        &self,
        source_nodes: &[Uuid],
        concept_nodes: &HashSet<Uuid>,
    ) -> HashSet<Uuid> {
        let mut bridged = HashSet::new();
        
        for source_id in source_nodes {
            if let Some(&source_idx) = self.id_to_index.get(source_id) {
                // 找到连接到概念节点的边
                for neighbor_idx in self.graph.neighbors_undirected(source_idx) {
                    if let Some(neighbor_id) = self.graph.node_weight(neighbor_idx) {
                        if concept_nodes.contains(neighbor_id) {
                            // 找到通过该概念连接的其他节点
                            for bridged_idx in self.graph.neighbors_undirected(neighbor_idx) {
                                if let Some(bridged_id) = self.graph.node_weight(bridged_idx) {
                                    if !source_nodes.contains(bridged_id) {
                                        bridged.insert(*bridged_id);
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
        
        bridged
    }

    /// 获取两个节点之间的边
    pub fn get_edge(&self, source: &Uuid, target: &Uuid) -> Option<&Edge> {
        let source_idx = self.id_to_index.get(source)?;
        let target_idx = self.id_to_index.get(target)?;
        
        self.graph.find_edge(*source_idx, *target_idx)
            .and_then(|e| self.graph.edge_weight(e))
    }

    /// 获取节点的所有出边
    pub fn get_outgoing_edges(&self, node_id: &Uuid) -> Vec<&Edge> {
        let mut edges = Vec::new();
        
        if let Some(&idx) = self.id_to_index.get(node_id) {
            for edge_idx in self.graph.edges(idx) {
                edges.push(edge_idx.weight());
            }
        }
        
        edges
    }

    /// 获取所有节点 ID
    pub fn get_all_node_ids(&self) -> Vec<Uuid> {
        self.id_to_index.keys().copied().collect()
    }

    /// 清空图
    pub fn clear(&mut self) {
        self.graph.clear();
        self.id_to_index.clear();
    }
}

impl Default for KnowledgeGraph {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_basic_operations() {
        let mut graph = KnowledgeGraph::new();
        
        let id1 = Uuid::new_v4();
        let id2 = Uuid::new_v4();
        
        graph.add_node(id1);
        graph.add_node(id2);
        
        assert_eq!(graph.node_count(), 2);
        assert!(graph.contains_node(&id1));
        assert!(graph.contains_node(&id2));
    }

    #[test]
    fn test_neighbors() {
        let mut graph = KnowledgeGraph::new();
        
        let id1 = Uuid::new_v4();
        let id2 = Uuid::new_v4();
        let id3 = Uuid::new_v4();
        
        graph.add_edge(Edge::new(id1, id2, "relates".to_string()));
        graph.add_edge(Edge::new(id2, id3, "relates".to_string()));
        
        let neighbors = graph.get_neighbors(&id1, 1);
        assert!(neighbors.contains(&id1));
        assert!(neighbors.contains(&id2));
        assert!(!neighbors.contains(&id3));
        
        let neighbors = graph.get_neighbors(&id1, 2);
        assert!(neighbors.contains(&id3));
    }
}
