//! 图遍历算法
//!
//! 提供多种图遍历和路径搜索算法

use crate::types::{Edge, NodeType};
use petgraph::graph::{DiGraph, NodeIndex};
use petgraph::visit::EdgeRef;
use std::collections::{HashMap, HashSet, VecDeque};
use uuid::Uuid;

/// 遍历结果
#[derive(Debug, Clone)]
pub struct TraversalResult {
    /// 访问的节点 ID 列表（按访问顺序）
    pub visited_nodes: Vec<Uuid>,
    /// 每个节点的深度
    pub depths: HashMap<Uuid, usize>,
    /// 到达每个节点的路径
    pub paths: HashMap<Uuid, Vec<Uuid>>,
}

/// BFS（广度优先搜索）遍历
/// 
/// 从起始节点开始，逐层扩展访问邻居节点
pub fn bfs_traverse(
    graph: &DiGraph<Uuid, Edge>,
    id_to_index: &HashMap<Uuid, NodeIndex>,
    start: &Uuid,
    max_depth: usize,
) -> TraversalResult {
    let mut result = TraversalResult {
        visited_nodes: Vec::new(),
        depths: HashMap::new(),
        paths: HashMap::new(),
    };

    let Some(&start_idx) = id_to_index.get(start) else {
        return result;
    };

    let mut queue: VecDeque<(NodeIndex, usize, Vec<Uuid>)> = VecDeque::new();
    let mut visited: HashSet<NodeIndex> = HashSet::new();

    queue.push_back((start_idx, 0, vec![*start]));
    visited.insert(start_idx);
    result.visited_nodes.push(*start);
    result.depths.insert(*start, 0);
    result.paths.insert(*start, vec![*start]);

    while let Some((current_idx, depth, path)) = queue.pop_front() {
        if depth >= max_depth {
            continue;
        }

        // 遍历所有邻居（包括入边和出边）
        for neighbor_idx in graph.neighbors_undirected(current_idx) {
            if !visited.contains(&neighbor_idx) {
                visited.insert(neighbor_idx);
                
                if let Some(neighbor_id) = graph.node_weight(neighbor_idx) {
                    let mut new_path = path.clone();
                    new_path.push(*neighbor_id);
                    
                    result.visited_nodes.push(*neighbor_id);
                    result.depths.insert(*neighbor_id, depth + 1);
                    result.paths.insert(*neighbor_id, new_path.clone());
                    
                    queue.push_back((neighbor_idx, depth + 1, new_path));
                }
            }
        }
    }

    result
}

/// DFS（深度优先搜索）遍历
pub fn dfs_traverse(
    graph: &DiGraph<Uuid, Edge>,
    id_to_index: &HashMap<Uuid, NodeIndex>,
    start: &Uuid,
    max_depth: usize,
) -> TraversalResult {
    let mut result = TraversalResult {
        visited_nodes: Vec::new(),
        depths: HashMap::new(),
        paths: HashMap::new(),
    };

    let Some(&start_idx) = id_to_index.get(start) else {
        return result;
    };

    fn dfs_helper(
        graph: &DiGraph<Uuid, Edge>,
        current: NodeIndex,
        depth: usize,
        max_depth: usize,
        path: &mut Vec<Uuid>,
        visited: &mut HashSet<NodeIndex>,
        result: &mut TraversalResult,
    ) {
        if depth > max_depth {
            return;
        }

        visited.insert(current);
        
        if let Some(node_id) = graph.node_weight(current) {
            path.push(*node_id);
            result.visited_nodes.push(*node_id);
            result.depths.insert(*node_id, depth);
            result.paths.insert(*node_id, path.clone());

            for neighbor in graph.neighbors_undirected(current) {
                if !visited.contains(&neighbor) {
                    dfs_helper(graph, neighbor, depth + 1, max_depth, path, visited, result);
                }
            }

            path.pop();
        }
    }

    let mut visited = HashSet::new();
    let mut path = Vec::new();
    dfs_helper(graph, start_idx, 0, max_depth, &mut path, &mut visited, &mut result);

    result
}

/// 查找两点之间的最短路径
pub fn find_shortest_path(
    graph: &DiGraph<Uuid, Edge>,
    id_to_index: &HashMap<Uuid, NodeIndex>,
    from: &Uuid,
    to: &Uuid,
) -> Option<Vec<Uuid>> {
    let start_idx = id_to_index.get(from)?;
    let end_idx = id_to_index.get(to)?;

    // BFS 寻找最短路径
    let mut queue: VecDeque<(NodeIndex, Vec<Uuid>)> = VecDeque::new();
    let mut visited: HashSet<NodeIndex> = HashSet::new();

    queue.push_back((*start_idx, vec![*from]));
    visited.insert(*start_idx);

    while let Some((current, path)) = queue.pop_front() {
        if current == *end_idx {
            return Some(path);
        }

        for neighbor in graph.neighbors_undirected(current) {
            if !visited.contains(&neighbor) {
                visited.insert(neighbor);
                if let Some(neighbor_id) = graph.node_weight(neighbor) {
                    let mut new_path = path.clone();
                    new_path.push(*neighbor_id);
                    queue.push_back((neighbor, new_path));
                }
            }
        }
    }

    None
}

/// 子图提取配置
#[derive(Debug, Clone)]
pub struct SubgraphConfig {
    /// 最大节点数
    pub max_nodes: usize,
    /// 是否包含概念节点
    pub include_concepts: bool,
    /// 节点类型过滤（None 表示不过滤）
    pub node_type_filter: Option<Vec<NodeType>>,
}

impl Default for SubgraphConfig {
    fn default() -> Self {
        Self {
            max_nodes: 30,
            include_concepts: true,
            node_type_filter: None,
        }
    }
}

/// 提取子图
/// 
/// 从一组种子节点开始，提取包含相关节点和边的子图
pub fn extract_subgraph(
    graph: &DiGraph<Uuid, Edge>,
    id_to_index: &HashMap<Uuid, NodeIndex>,
    seed_nodes: &[Uuid],
    hop_depth: usize,
    config: &SubgraphConfig,
) -> (HashSet<Uuid>, Vec<Edge>) {
    let mut collected_nodes: HashSet<Uuid> = HashSet::new();
    let mut collected_edges: Vec<Edge> = Vec::new();
    let mut edge_set: HashSet<(Uuid, Uuid)> = HashSet::new();

    // 从每个种子节点进行 BFS 扩展
    for seed in seed_nodes {
        if collected_nodes.len() >= config.max_nodes {
            break;
        }

        let result = bfs_traverse(graph, id_to_index, seed, hop_depth);
        
        for node_id in result.visited_nodes {
            if collected_nodes.len() >= config.max_nodes {
                break;
            }
            collected_nodes.insert(node_id);
        }
    }

    // 收集节点之间的边
    for &node_id in &collected_nodes {
        if let Some(&idx) = id_to_index.get(&node_id) {
            for edge_ref in graph.edges(idx) {
                let target_idx = edge_ref.target();
                if let Some(target_id) = graph.node_weight(target_idx) {
                    if collected_nodes.contains(target_id) {
                        let edge_key = (node_id, *target_id);
                        if !edge_set.contains(&edge_key) {
                            edge_set.insert(edge_key);
                            collected_edges.push(edge_ref.weight().clone());
                        }
                    }
                }
            }
        }
    }

    (collected_nodes, collected_edges)
}

/// 计算节点的中心性分数
/// 
/// 基于节点的连接数量
pub fn calculate_centrality(
    graph: &DiGraph<Uuid, Edge>,
    id_to_index: &HashMap<Uuid, NodeIndex>,
    node_id: &Uuid,
) -> f64 {
    let Some(&idx) = id_to_index.get(node_id) else {
        return 0.0;
    };

    let in_degree = graph.edges_directed(idx, petgraph::Direction::Incoming).count();
    let out_degree = graph.edges_directed(idx, petgraph::Direction::Outgoing).count();
    
    let total_nodes = graph.node_count() as f64;
    if total_nodes <= 1.0 {
        return 0.0;
    }

    // 归一化度中心性
    (in_degree + out_degree) as f64 / (2.0 * (total_nodes - 1.0))
}

/// 查找所有连通分量
pub fn find_connected_components(
    graph: &DiGraph<Uuid, Edge>,
) -> Vec<HashSet<NodeIndex>> {
    let mut visited: HashSet<NodeIndex> = HashSet::new();
    let mut components: Vec<HashSet<NodeIndex>> = Vec::new();

    for node in graph.node_indices() {
        if !visited.contains(&node) {
            let mut component: HashSet<NodeIndex> = HashSet::new();
            let mut stack = vec![node];

            while let Some(current) = stack.pop() {
                if !visited.contains(&current) {
                    visited.insert(current);
                    component.insert(current);

                    for neighbor in graph.neighbors_undirected(current) {
                        if !visited.contains(&neighbor) {
                            stack.push(neighbor);
                        }
                    }
                }
            }

            if !component.is_empty() {
                components.push(component);
            }
        }
    }

    components
}

/// 获取节点的邻居信息
#[derive(Debug, Clone)]
pub struct NeighborInfo {
    pub node_id: Uuid,
    pub distance: usize,
    pub relation: Option<String>,
}

/// 获取节点的所有邻居及其关系
pub fn get_neighbors_with_relations(
    graph: &DiGraph<Uuid, Edge>,
    id_to_index: &HashMap<Uuid, NodeIndex>,
    node_id: &Uuid,
) -> Vec<NeighborInfo> {
    let mut neighbors = Vec::new();
    
    let Some(&idx) = id_to_index.get(node_id) else {
        return neighbors;
    };

    // 出边邻居
    for edge in graph.edges(idx) {
        if let Some(target_id) = graph.node_weight(edge.target()) {
            neighbors.push(NeighborInfo {
                node_id: *target_id,
                distance: 1,
                relation: Some(edge.weight().relation.clone()),
            });
        }
    }

    // 入边邻居
    for edge in graph.edges_directed(idx, petgraph::Direction::Incoming) {
        if let Some(source_id) = graph.node_weight(edge.source()) {
            neighbors.push(NeighborInfo {
                node_id: *source_id,
                distance: 1,
                relation: Some(edge.weight().relation.clone()),
            });
        }
    }

    neighbors
}

#[cfg(test)]
mod tests {
    use super::*;

    fn create_test_graph() -> (DiGraph<Uuid, Edge>, HashMap<Uuid, NodeIndex>) {
        let mut graph = DiGraph::new();
        let mut id_to_index = HashMap::new();

        let ids: Vec<Uuid> = (0..5).map(|_| Uuid::new_v4()).collect();
        
        for &id in &ids {
            let idx = graph.add_node(id);
            id_to_index.insert(id, idx);
        }

        // 0 -> 1 -> 2 -> 3
        //      |
        //      v
        //      4
        graph.add_edge(id_to_index[&ids[0]], id_to_index[&ids[1]], 
            Edge::new(ids[0], ids[1], "a".to_string()));
        graph.add_edge(id_to_index[&ids[1]], id_to_index[&ids[2]], 
            Edge::new(ids[1], ids[2], "b".to_string()));
        graph.add_edge(id_to_index[&ids[2]], id_to_index[&ids[3]], 
            Edge::new(ids[2], ids[3], "c".to_string()));
        graph.add_edge(id_to_index[&ids[1]], id_to_index[&ids[4]], 
            Edge::new(ids[1], ids[4], "d".to_string()));

        (graph, id_to_index)
    }

    #[test]
    fn test_bfs_traverse() {
        let (graph, id_to_index) = create_test_graph();
        let start = *id_to_index.keys().next().unwrap();
        
        let result = bfs_traverse(&graph, &id_to_index, &start, 2);
        
        assert!(result.visited_nodes.len() >= 1);
        assert!(result.depths.contains_key(&start));
        assert_eq!(result.depths[&start], 0);
    }

    #[test]
    fn test_shortest_path() {
        let (graph, id_to_index) = create_test_graph();
        let ids: Vec<Uuid> = id_to_index.keys().copied().collect();
        
        let path = find_shortest_path(&graph, &id_to_index, &ids[0], &ids[0]);
        assert!(path.is_some());
        assert_eq!(path.unwrap().len(), 1);
    }
}
