//! 类型定义
//!
//! 基于 AutoSchemaKG 论文的实体-事件-概念三层知识图谱架构
//! 使用 NodeData enum 保证类型安全

use serde::{Deserialize, Serialize};
use uuid::Uuid;

// ============================================
// 节点类型
// ============================================

/// 节点类型枚举（用于过滤和序列化标识）
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq, Hash)]
#[serde(rename_all = "lowercase")]
pub enum NodeType {
    /// 实体节点 - 具体的人物、物品、地点
    Entity,
    /// 事件节点 - 完整的动作或状态描述
    Event,
    /// 概念节点 - 抽象化的类型标签
    Concept,
}

/// 实体类型
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "lowercase")]
pub enum EntityType {
    Person,
    Place,
    Object,
    Time,
    Other,
}

/// 节点类型特有数据（enum variant 保证类型安全）
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum NodeData {
    /// 实体特有数据
    Entity {
        entity_type: EntityType,
        #[serde(default)]
        attributes: Option<serde_json::Value>,
    },
    /// 事件特有数据
    Event {
        /// 参与实体 ID 列表
        #[serde(default)]
        participants: Vec<Uuid>,
        /// 事件时间，格式：YYYY-MM-DD-HH-MM
        event_time: String,
        /// 来源对话 ID
        #[serde(default)]
        source_conversation_id: Option<String>,
    },
    /// 概念特有数据
    Concept {
        /// 关联实例数量
        #[serde(default = "default_instance_count")]
        instance_count: u32,
        /// 最后使用时间
        last_used_at: i64,
    },
}

fn default_instance_count() -> u32 {
    1
}

/// 记忆节点
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemoryNode {
    /// 唯一标识 (UUID)
    pub id: Uuid,
    /// 节点文本内容
    pub content: String,
    /// 向量嵌入 (bge-m3, 1024维)
    pub embedding: Vec<f32>,
    /// 重要性权重 (0.0 - 1.0)
    pub importance: f32,
    /// 访问次数
    pub access_count: u32,
    /// 创建时间戳 (Unix timestamp)
    pub created_at: i64,
    /// 更新时间戳
    pub updated_at: i64,
    /// 类型特有数据
    pub data: NodeData,
}

impl MemoryNode {
    /// 获取节点类型（从 data enum variant 推导）
    pub fn node_type(&self) -> NodeType {
        match &self.data {
            NodeData::Entity { .. } => NodeType::Entity,
            NodeData::Event { .. } => NodeType::Event,
            NodeData::Concept { .. } => NodeType::Concept,
        }
    }

    // ============================================
    // 便捷访问方法
    // ============================================

    /// 获取实体类型（仅 Entity 节点有效）
    pub fn entity_type(&self) -> Option<EntityType> {
        match &self.data {
            NodeData::Entity { entity_type, .. } => Some(*entity_type),
            _ => None,
        }
    }

    /// 获取属性（仅 Entity 节点有效）
    pub fn attributes(&self) -> Option<&serde_json::Value> {
        match &self.data {
            NodeData::Entity { attributes, .. } => attributes.as_ref(),
            _ => None,
        }
    }

    /// 获取事件时间（仅 Event 节点有效）
    pub fn event_time(&self) -> Option<&str> {
        match &self.data {
            NodeData::Event { event_time, .. } => Some(event_time.as_str()),
            _ => None,
        }
    }

    /// 获取参与实体列表（仅 Event 节点有效）
    pub fn participants(&self) -> Option<&[Uuid]> {
        match &self.data {
            NodeData::Event { participants, .. } => Some(participants.as_slice()),
            _ => None,
        }
    }

    /// 获取来源对话 ID（仅 Event 节点有效）
    pub fn source_conversation_id(&self) -> Option<&str> {
        match &self.data {
            NodeData::Event { source_conversation_id, .. } => source_conversation_id.as_deref(),
            _ => None,
        }
    }

    /// 获取实例数量（仅 Concept 节点有效）
    pub fn instance_count(&self) -> Option<u32> {
        match &self.data {
            NodeData::Concept { instance_count, .. } => Some(*instance_count),
            _ => None,
        }
    }

    /// 获取最后使用时间（仅 Concept 节点有效）
    pub fn last_used_at(&self) -> Option<i64> {
        match &self.data {
            NodeData::Concept { last_used_at, .. } => Some(*last_used_at),
            _ => None,
        }
    }

    // ============================================
    // 构造方法
    // ============================================

    /// 创建新的实体节点
    pub fn new_entity(content: String, entity_type: EntityType) -> Self {
        let now = chrono::Utc::now().timestamp();
        Self {
            id: Uuid::new_v4(),
            content,
            embedding: Vec::new(),
            importance: 0.5,
            access_count: 0,
            created_at: now,
            updated_at: now,
            data: NodeData::Entity {
                entity_type,
                attributes: None,
            },
        }
    }

    /// 创建新的事件节点
    pub fn new_event(content: String, event_time: String) -> Self {
        let now = chrono::Utc::now().timestamp();
        Self {
            id: Uuid::new_v4(),
            content,
            embedding: Vec::new(),
            importance: 0.5,
            access_count: 0,
            created_at: now,
            updated_at: now,
            data: NodeData::Event {
                participants: Vec::new(),
                event_time,
                source_conversation_id: None,
            },
        }
    }

    /// 创建新的概念节点
    pub fn new_concept(content: String) -> Self {
        let now = chrono::Utc::now().timestamp();
        Self {
            id: Uuid::new_v4(),
            content,
            embedding: Vec::new(),
            importance: 0.5,
            access_count: 0,
            created_at: now,
            updated_at: now,
            data: NodeData::Concept {
                instance_count: 1,
                last_used_at: now,
            },
        }
    }
}

// ============================================
// 边类型
// ============================================

/// 边定义
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Edge {
    /// 唯一标识
    pub id: Uuid,
    /// 源节点ID
    pub source: Uuid,
    /// 目标节点ID
    pub target: Uuid,
    /// 关系类型
    pub relation: String,
    /// 边权重 (0.0 - 1.0)
    pub weight: f32,
    /// 创建时间
    pub created_at: i64,
    /// 额外元数据
    #[serde(skip_serializing_if = "Option::is_none")]
    pub metadata: Option<serde_json::Value>,
}

impl Edge {
    /// 创建新边
    pub fn new(source: Uuid, target: Uuid, relation: String) -> Self {
        Self {
            id: Uuid::new_v4(),
            source,
            target,
            relation,
            weight: 1.0,
            created_at: chrono::Utc::now().timestamp(),
            metadata: None,
        }
    }
}

// ============================================
// 查询相关类型
// ============================================

/// 查询参数
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QueryParams {
    /// 用户当前消息（必需）
    pub user_message: String,
    /// 最近几轮对话（可选）
    #[serde(skip_serializing_if = "Option::is_none")]
    pub recent_messages: Option<Vec<Message>>,
    /// 返回记忆数量
    #[serde(skip_serializing_if = "Option::is_none")]
    pub top_k: Option<usize>,
    /// 是否返回原始数据
    #[serde(skip_serializing_if = "Option::is_none")]
    pub include_raw: Option<bool>,
}

/// 查询结果
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QueryResult {
    /// 格式化好的记忆上下文文本
    pub formatted_context: String,
    /// 记忆条数
    pub count: usize,
    /// 原始记忆数据（仅当 include_raw = true 时返回）
    #[serde(skip_serializing_if = "Option::is_none")]
    pub raw: Option<Vec<RetrievedMemory>>,
}

/// 检索到的记忆
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RetrievedMemory {
    /// 记忆内容
    pub content: String,
    /// 记忆类型
    pub memory_type: NodeType,
    /// 相关度得分
    pub relevance: f32,
    /// 事件时间
    #[serde(skip_serializing_if = "Option::is_none")]
    pub event_time: Option<String>,
    /// 距今时间描述
    #[serde(skip_serializing_if = "Option::is_none")]
    pub time_ago: Option<String>,
}

/// 对话消息
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Message {
    /// 角色 (user/assistant/system)
    pub role: String,
    /// 消息内容
    pub content: String,
    /// 时间戳
    #[serde(skip_serializing_if = "Option::is_none")]
    pub timestamp: Option<i64>,
}

// ============================================
// 三元组提取相关
// ============================================

/// 原始三元组（LLM 提取结果）
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RawTriple {
    /// 头实体
    #[serde(rename = "Head")]
    pub head: String,
    /// 关系
    #[serde(rename = "Relation")]
    pub relation: String,
    /// 尾实体
    #[serde(rename = "Tail")]
    pub tail: String,
}

/// 提取的事件
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExtractedEvent {
    /// 事件描述句
    #[serde(rename = "Event")]
    pub event: String,
    /// 事件时间
    #[serde(rename = "EventTime")]
    pub event_time: String,
    /// 参与实体列表
    #[serde(rename = "Entities")]
    pub entities: Vec<String>,
}

// ============================================
// 配置类型
// ============================================

/// 检索配置
#[derive(Debug, Clone)]
pub struct RetrievalConfig {
    /// 初始检索数量
    pub top_k: usize,
    /// 子图扩展深度
    pub hop_depth: usize,
    /// 子图最大节点数
    pub max_subgraph_nodes: usize,
    /// 重排序后返回数量
    pub rerank_top_n: usize,
}

impl Default for RetrievalConfig {
    fn default() -> Self {
        Self {
            top_k: 10,
            hop_depth: 2,
            max_subgraph_nodes: 30,
            rerank_top_n: 5,
        }
    }
}
