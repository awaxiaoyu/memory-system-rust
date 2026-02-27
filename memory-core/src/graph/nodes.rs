//! 节点操作
//!
//! 提供节点的创建、验证和转换辅助函数

use crate::types::*;
use uuid::Uuid;
use chrono::Utc;

/// 节点构建器
/// 
/// 使用流式 API 构建 MemoryNode
pub struct NodeBuilder {
    node: MemoryNode,
}

impl NodeBuilder {
    /// 创建实体节点构建器
    pub fn entity(content: impl Into<String>) -> Self {
        Self {
            node: MemoryNode::new_entity(content.into(), EntityType::Other),
        }
    }

    /// 创建事件节点构建器
    pub fn event(content: impl Into<String>) -> Self {
        let now = Utc::now().format("%Y-%m-%d-%H-%M").to_string();
        Self {
            node: MemoryNode::new_event(content.into(), now),
        }
    }

    /// 创建概念节点构建器
    pub fn concept(content: impl Into<String>) -> Self {
        Self {
            node: MemoryNode::new_concept(content.into()),
        }
    }

    /// 设置实体类型
    pub fn entity_type(mut self, entity_type: EntityType) -> Self {
        if let NodeData::Entity { entity_type: ref mut et, .. } = self.node.data {
            *et = entity_type;
        }
        self
    }

    /// 设置事件时间
    pub fn event_time(mut self, time: impl Into<String>) -> Self {
        if let NodeData::Event { event_time: ref mut et, .. } = self.node.data {
            *et = time.into();
        }
        self
    }

    /// 设置参与实体
    pub fn participants(mut self, participants: Vec<Uuid>) -> Self {
        if let NodeData::Event { participants: ref mut p, .. } = self.node.data {
            *p = participants;
        }
        self
    }

    /// 添加参与实体
    pub fn add_participant(mut self, participant: Uuid) -> Self {
        if let NodeData::Event { participants: ref mut p, .. } = self.node.data {
            p.push(participant);
        }
        self
    }

    /// 设置重要性
    pub fn importance(mut self, importance: f32) -> Self {
        self.node.importance = importance.clamp(0.0, 1.0);
        self
    }

    /// 设置嵌入向量
    pub fn embedding(mut self, embedding: Vec<f32>) -> Self {
        self.node.embedding = embedding;
        self
    }

    /// 设置属性
    pub fn attributes(mut self, attrs: serde_json::Value) -> Self {
        if let NodeData::Entity { attributes: ref mut a, .. } = self.node.data {
            *a = Some(attrs);
        }
        self
    }

    /// 设置来源对话ID
    pub fn source_conversation(mut self, conv_id: impl Into<String>) -> Self {
        if let NodeData::Event { source_conversation_id: ref mut s, .. } = self.node.data {
            *s = Some(conv_id.into());
        }
        self
    }

    /// 构建节点
    pub fn build(self) -> MemoryNode {
        self.node
    }
}

/// 节点验证结果
#[derive(Debug)]
pub struct NodeValidationResult {
    pub is_valid: bool,
    pub errors: Vec<String>,
}

/// 验证节点
pub fn validate_node(node: &MemoryNode) -> NodeValidationResult {
    let mut errors = Vec::new();

    // 检查内容不为空
    if node.content.trim().is_empty() {
        errors.push("节点内容不能为空".to_string());
    }

    // 检查重要性范围
    if node.importance < 0.0 || node.importance > 1.0 {
        errors.push(format!("重要性值必须在 0-1 之间，当前: {}", node.importance));
    }

    // 检查向量维度（如果已设置）
    if !node.embedding.is_empty() && node.embedding.len() != 1024 {
        errors.push(format!("向量维度应为 1024，当前: {}", node.embedding.len()));
    }

    // 类型特定检查
    match &node.data {
        NodeData::Entity { entity_type: _, .. } => {
            // Entity 节点已通过 enum 保证有 entity_type
        }
        NodeData::Event { event_time, .. } => {
            if event_time.is_empty() {
                errors.push("事件节点必须指定 event_time".to_string());
            }
        }
        NodeData::Concept { .. } => {
            // 概念节点无特殊要求
        }
    }

    NodeValidationResult {
        is_valid: errors.is_empty(),
        errors,
    }
}

/// 合并两个节点（用于去重）
/// 
/// 保留更高重要性和更新时间戳的节点信息
pub fn merge_nodes(existing: &mut MemoryNode, new_node: &MemoryNode) {
    // 更新访问次数
    existing.access_count += 1;
    
    // 更新时间戳
    existing.updated_at = Utc::now().timestamp();
    
    // 取更高的重要性
    if new_node.importance > existing.importance {
        existing.importance = new_node.importance;
    }
    
    // 更新嵌入（如果新节点有更新的嵌入）
    if !new_node.embedding.is_empty() {
        existing.embedding = new_node.embedding.clone();
    }
    
    // 合并参与者列表（用于事件节点）
    if let (NodeData::Event { participants: ref mut existing_parts, .. },
            NodeData::Event { participants: ref new_parts, .. }) =
        (&mut existing.data, &new_node.data)
    {
        for p in new_parts {
            if !existing_parts.contains(p) {
                existing_parts.push(*p);
            }
        }
    }
}

/// 计算节点衰减后的权重
/// 
/// 基于时间的记忆衰减公式: weight = importance * decay_factor
pub fn calculate_decayed_weight(node: &MemoryNode, decay_rate: f64) -> f64 {
    let now = Utc::now().timestamp();
    let age_seconds = (now - node.created_at) as f64;
    let age_days = age_seconds / 86400.0;
    
    // 指数衰减
    let decay_factor = (-decay_rate * age_days).exp();
    
    // 考虑访问次数的增强
    let access_boost = 1.0 + (node.access_count as f64).ln().max(0.0) * 0.1;
    
    (node.importance as f64) * decay_factor * access_boost
}

/// 从内容推断实体类型
///
/// 中文不需要 to_lowercase，直接使用原始字符匹配
pub fn infer_entity_type(content: &str) -> EntityType {
    // 人物关键词
    let person_keywords = ["我", "你", "他", "她", "哥", "姐", "弟", "妹", 
                          "爸", "妈", "老师", "朋友", "同学"];
    if person_keywords.iter().any(|k| content.contains(k)) {
        return EntityType::Person;
    }
    
    // 地点关键词
    let place_keywords = ["家", "学校", "公司", "商店", "餐厅", "公园", 
                         "医院", "车站", "机场"];
    if place_keywords.iter().any(|k| content.contains(k)) {
        return EntityType::Place;
    }
    
    // 时间关键词
    let time_keywords = ["今天", "昨天", "明天", "上午", "下午", "晚上", 
                        "周一", "周末", "月", "年"];
    if time_keywords.iter().any(|k| content.contains(k)) {
        return EntityType::Time;
    }
    
    EntityType::Other
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_node_builder() {
        let node = NodeBuilder::entity("张三")
            .entity_type(EntityType::Person)
            .importance(0.8)
            .build();
        
        assert_eq!(node.content, "张三");
        assert_eq!(node.node_type(), NodeType::Entity);
        assert_eq!(node.entity_type(), Some(EntityType::Person));
        assert_eq!(node.importance, 0.8);
    }

    #[test]
    fn test_validate_node() {
        let node = NodeBuilder::entity("")
            .entity_type(EntityType::Person)
            .build();
        
        let result = validate_node(&node);
        assert!(!result.is_valid);
        assert!(result.errors.iter().any(|e| e.contains("内容不能为空")));
    }

    #[test]
    fn test_infer_entity_type() {
        assert_eq!(infer_entity_type("我的朋友张三"), EntityType::Person);
        assert_eq!(infer_entity_type("北京的公园"), EntityType::Place);
        assert_eq!(infer_entity_type("今天下午"), EntityType::Time);
        assert_eq!(infer_entity_type("一本书"), EntityType::Other);
    }
}
