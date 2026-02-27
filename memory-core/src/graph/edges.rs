//! 边操作
//!
//! 提供边的创建、验证和操作辅助函数

use crate::types::Edge;
use uuid::Uuid;
// use chrono::Utc;
use std::collections::HashMap;

/// 预定义的关系类型
pub mod relation_types {
    /// 实体-实体关系
    pub const RELATES_TO: &str = "relates_to";
    pub const BELONGS_TO: &str = "belongs_to";
    pub const CREATED_BY: &str = "created_by";
    pub const LOCATED_AT: &str = "located_at";
    pub const KNOWS: &str = "knows";
    pub const OWNS: &str = "owns";
    
    /// 实体-事件关系
    pub const PARTICIPATES_IN: &str = "participates_in";
    pub const INITIATED: &str = "initiated";
    pub const AFFECTED_BY: &str = "affected_by";
    
    /// 事件-事件关系
    pub const BEFORE: &str = "before";
    pub const AFTER: &str = "after";
    pub const AT_SAME_TIME: &str = "at_same_time";
    pub const BECAUSE: &str = "because";
    pub const AS_RESULT: &str = "as_result";
    
    /// 实体/事件-概念关系
    pub const IS_A: &str = "is_a";
    pub const INSTANCE_OF: &str = "instance_of";
    pub const CONCEPTUALIZED_AS: &str = "conceptualized_as";
}

/// 边构建器
pub struct EdgeBuilder {
    edge: Edge,
}

impl EdgeBuilder {
    /// 创建新的边构建器
    pub fn new(source: Uuid, target: Uuid, relation: impl Into<String>) -> Self {
        Self {
            edge: Edge::new(source, target, relation.into()),
        }
    }

    /// 创建"参与"关系边
    pub fn participates(entity_id: Uuid, event_id: Uuid) -> Self {
        Self::new(entity_id, event_id, relation_types::PARTICIPATES_IN)
    }

    /// 创建"概念化"关系边
    pub fn conceptualizes(node_id: Uuid, concept_id: Uuid) -> Self {
        Self::new(node_id, concept_id, relation_types::CONCEPTUALIZED_AS)
    }

    /// 创建时序关系边（事件之间）
    pub fn temporal(from_event: Uuid, to_event: Uuid, relation: &str) -> Self {
        Self::new(from_event, to_event, relation)
    }

    /// 设置边权重
    pub fn weight(mut self, weight: f32) -> Self {
        self.edge.weight = weight.clamp(0.0, 1.0);
        self
    }

    /// 设置元数据
    pub fn metadata(mut self, metadata: serde_json::Value) -> Self {
        self.edge.metadata = Some(metadata);
        self
    }

    /// 构建边
    pub fn build(self) -> Edge {
        self.edge
    }
}

/// 边验证结果
#[derive(Debug)]
pub struct EdgeValidationResult {
    pub is_valid: bool,
    pub errors: Vec<String>,
}

/// 验证边
pub fn validate_edge(edge: &Edge) -> EdgeValidationResult {
    let mut errors = Vec::new();

    // 检查关系不为空
    if edge.relation.trim().is_empty() {
        errors.push("关系类型不能为空".to_string());
    }

    // 检查权重范围
    if edge.weight < 0.0 || edge.weight > 1.0 {
        errors.push(format!("边权重必须在 0-1 之间，当前: {}", edge.weight));
    }

    // 检查不能自环（可选，根据业务需求）
    if edge.source == edge.target {
        errors.push("不允许自环边".to_string());
    }

    EdgeValidationResult {
        is_valid: errors.is_empty(),
        errors,
    }
}

/// 边索引
/// 
/// 用于快速查找边
pub struct EdgeIndex {
    /// 源节点 -> 边列表
    by_source: HashMap<Uuid, Vec<Edge>>,
    /// 目标节点 -> 边列表
    by_target: HashMap<Uuid, Vec<Edge>>,
    /// 关系类型 -> 边列表
    by_relation: HashMap<String, Vec<Edge>>,
}

impl EdgeIndex {
    /// 创建新的边索引
    pub fn new() -> Self {
        Self {
            by_source: HashMap::new(),
            by_target: HashMap::new(),
            by_relation: HashMap::new(),
        }
    }

    /// 从边列表创建索引
    pub fn from_edges(edges: impl IntoIterator<Item = Edge>) -> Self {
        let mut index = Self::new();
        for edge in edges {
            index.add(edge);
        }
        index
    }

    /// 添加边到索引
    pub fn add(&mut self, edge: Edge) {
        self.by_source
            .entry(edge.source)
            .or_insert_with(Vec::new)
            .push(edge.clone());
        
        self.by_target
            .entry(edge.target)
            .or_insert_with(Vec::new)
            .push(edge.clone());
        
        self.by_relation
            .entry(edge.relation.clone())
            .or_insert_with(Vec::new)
            .push(edge);
    }

    /// 按源节点查找边
    pub fn find_by_source(&self, source: &Uuid) -> Vec<&Edge> {
        self.by_source
            .get(source)
            .map(|v| v.iter().collect())
            .unwrap_or_default()
    }

    /// 按目标节点查找边
    pub fn find_by_target(&self, target: &Uuid) -> Vec<&Edge> {
        self.by_target
            .get(target)
            .map(|v| v.iter().collect())
            .unwrap_or_default()
    }

    /// 按关系类型查找边
    pub fn find_by_relation(&self, relation: &str) -> Vec<&Edge> {
        self.by_relation
            .get(relation)
            .map(|v| v.iter().collect())
            .unwrap_or_default()
    }

    /// 查找两个节点之间的边
    pub fn find_between(&self, source: &Uuid, target: &Uuid) -> Option<&Edge> {
        self.by_source
            .get(source)?
            .iter()
            .find(|e| &e.target == target)
    }

    /// 获取节点的所有关联边（入边和出边）
    pub fn find_all_for_node(&self, node_id: &Uuid) -> Vec<&Edge> {
        let mut edges: Vec<&Edge> = Vec::new();
        
        if let Some(outgoing) = self.by_source.get(node_id) {
            edges.extend(outgoing.iter());
        }
        if let Some(incoming) = self.by_target.get(node_id) {
            edges.extend(incoming.iter());
        }
        
        edges
    }

    /// 获取索引中的边总数
    pub fn count(&self) -> usize {
        self.by_source.values().map(|v| v.len()).sum()
    }
}

impl Default for EdgeIndex {
    fn default() -> Self {
        Self::new()
    }
}

/// 批量创建参与关系边
/// 
/// 将多个实体连接到同一个事件
pub fn create_participation_edges(entities: &[Uuid], event_id: Uuid) -> Vec<Edge> {
    entities
        .iter()
        .map(|&entity_id| EdgeBuilder::participates(entity_id, event_id).build())
        .collect()
}

/// 批量创建概念化边
/// 
/// 将一个节点连接到多个概念
pub fn create_conceptualization_edges(node_id: Uuid, concepts: &[Uuid]) -> Vec<Edge> {
    concepts
        .iter()
        .map(|&concept_id| EdgeBuilder::conceptualizes(node_id, concept_id).build())
        .collect()
}

/// 规范化关系类型
/// 
/// 将用户输入的关系类型映射到标准类型
pub fn normalize_relation(relation: &str) -> String {
    let relation_lower = relation.to_lowercase().trim().to_string();
    
    match relation_lower.as_str() {
        // 时序关系
        "之前" | "前" | "早于" => relation_types::BEFORE.to_string(),
        "之后" | "后" | "晚于" => relation_types::AFTER.to_string(),
        "同时" | "一起" => relation_types::AT_SAME_TIME.to_string(),
        "因为" | "由于" | "因" => relation_types::BECAUSE.to_string(),
        "导致" | "结果" | "所以" => relation_types::AS_RESULT.to_string(),
        
        // 归属关系
        "属于" | "归属" => relation_types::BELONGS_TO.to_string(),
        "拥有" | "有" => relation_types::OWNS.to_string(),
        "认识" | "知道" => relation_types::KNOWS.to_string(),
        "位于" | "在" => relation_types::LOCATED_AT.to_string(),
        
        // 默认保持原样
        _ => relation_lower,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_edge_builder() {
        let source = Uuid::new_v4();
        let target = Uuid::new_v4();
        
        let edge = EdgeBuilder::new(source, target, "relates_to")
            .weight(0.8)
            .build();
        
        assert_eq!(edge.source, source);
        assert_eq!(edge.target, target);
        assert_eq!(edge.relation, "relates_to");
        assert_eq!(edge.weight, 0.8);
    }

    #[test]
    fn test_edge_index() {
        let id1 = Uuid::new_v4();
        let id2 = Uuid::new_v4();
        let id3 = Uuid::new_v4();
        
        let edges = vec![
            Edge::new(id1, id2, "relates".to_string()),
            Edge::new(id2, id3, "relates".to_string()),
        ];
        
        let index = EdgeIndex::from_edges(edges);
        
        assert_eq!(index.count(), 2);
        assert_eq!(index.find_by_source(&id1).len(), 1);
        assert_eq!(index.find_by_target(&id3).len(), 1);
    }

    #[test]
    fn test_normalize_relation() {
        assert_eq!(normalize_relation("之前"), relation_types::BEFORE);
        assert_eq!(normalize_relation("因为"), relation_types::BECAUSE);
        assert_eq!(normalize_relation("custom"), "custom");
    }
}
