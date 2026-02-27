//! LanceDB 表结构定义与 Arrow 批量转换
//!
//! 负责定义 LanceDB 表 schema、在领域类型与 Arrow RecordBatch 之间互转

use crate::error::{MemoryError, Result};
use crate::types::*;
use arrow_array::{
    Array, Float32Array, Int64Array, RecordBatch, RecordBatchIterator,
    StringArray, UInt32Array, FixedSizeListArray, ArrayRef,
};
use arrow_schema::{DataType, Field, Schema, SchemaRef};
use serde::{Deserialize, Serialize};
use std::sync::Arc;
use uuid::Uuid;

/// 向量维度常量
pub const VECTOR_DIM: i32 = 1024;

// ============================================
// 中间记录类型（领域类型 ↔ 平坦记录 ↔ Arrow）
// ============================================

/// LanceDB 节点记录（平坦结构，对应 LanceDB 表的一行）
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NodeRecord {
    pub id: String,
    pub node_type: String,
    pub content: String,
    pub vector: Vec<f32>,
    pub importance: f32,
    pub access_count: u32,
    pub event_time: Option<String>,
    pub created_at: i64,
    pub updated_at: i64,
    pub metadata: String, // JSON 序列化的类型特有字段
}

impl NodeRecord {
    /// 从 MemoryNode 转换
    pub fn from_node(node: &MemoryNode) -> Self {
        let node_type_str = match node.node_type() {
            NodeType::Entity => "entity",
            NodeType::Event => "event",
            NodeType::Concept => "concept",
        };

        // 从 NodeData 提取 event_time 和 metadata
        let (event_time, metadata) = match &node.data {
            NodeData::Entity { entity_type, attributes } => {
                let meta = serde_json::json!({
                    "entity_type": entity_type,
                    "attributes": attributes,
                });
                (None, meta.to_string())
            }
            NodeData::Event { participants, event_time, source_conversation_id } => {
                let meta = serde_json::json!({
                    "participants": participants,
                    "source_conversation_id": source_conversation_id,
                });
                (Some(event_time.clone()), meta.to_string())
            }
            NodeData::Concept { instance_count, last_used_at } => {
                let meta = serde_json::json!({
                    "instance_count": instance_count,
                    "last_used_at": last_used_at,
                });
                (None, meta.to_string())
            }
        };

        Self {
            id: node.id.to_string(),
            node_type: node_type_str.to_string(),
            content: node.content.clone(),
            vector: if node.embedding.is_empty() {
                vec![0.0f32; VECTOR_DIM as usize]
            } else {
                node.embedding.clone()
            },
            importance: node.importance,
            access_count: node.access_count,
            event_time,
            created_at: node.created_at,
            updated_at: node.updated_at,
            metadata,
        }
    }

    /// 转换为 MemoryNode
    pub fn to_node(&self) -> Result<MemoryNode> {
        let metadata: serde_json::Value = serde_json::from_str(&self.metadata)
            .map_err(|e| MemoryError::Storage(format!("解析节点 metadata 失败: {}", e)))?;

        let data = match self.node_type.as_str() {
            "entity" => {
                let entity_type = metadata.get("entity_type")
                    .and_then(|v| serde_json::from_value(v.clone()).ok())
                    .unwrap_or(EntityType::Other);
                let attributes = metadata.get("attributes")
                    .and_then(|v| if v.is_null() { None } else { Some(v.clone()) });
                NodeData::Entity { entity_type, attributes }
            }
            "event" => {
                let participants = metadata.get("participants")
                    .and_then(|v| serde_json::from_value(v.clone()).ok())
                    .unwrap_or_default();
                let event_time = self.event_time.clone().unwrap_or_default();
                let source_conversation_id = metadata.get("source_conversation_id")
                    .and_then(|v| v.as_str())
                    .map(|s| s.to_string());
                NodeData::Event { participants, event_time, source_conversation_id }
            }
            "concept" => {
                let instance_count = metadata.get("instance_count")
                    .and_then(|v| v.as_u64())
                    .map(|v| v as u32)
                    .unwrap_or(1);
                let last_used_at = metadata.get("last_used_at")
                    .and_then(|v| v.as_i64())
                    .unwrap_or(0);
                NodeData::Concept { instance_count, last_used_at }
            }
            _ => {
                log::warn!("未知节点类型 '{}', 默认为 Entity", self.node_type);
                NodeData::Entity { entity_type: EntityType::Other, attributes: None }
            }
        };

        // 判断向量是否全零（即未嵌入）
        let embedding = if self.vector.iter().all(|&v| v == 0.0) {
            Vec::new()
        } else {
            self.vector.clone()
        };

        Ok(MemoryNode {
            id: Uuid::parse_str(&self.id)
                .unwrap_or_else(|_| { log::error!("解析节点 UUID 失败: {}", self.id); Uuid::new_v4() }),
            content: self.content.clone(),
            embedding,
            importance: self.importance,
            access_count: self.access_count,
            created_at: self.created_at,
            updated_at: self.updated_at,
            data,
        })
    }
}

/// LanceDB 边记录
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EdgeRecord {
    pub id: String,
    pub source_id: String,
    pub target_id: String,
    pub relation: String,
    pub weight: f32,
    pub created_at: i64,
}

impl EdgeRecord {
    pub fn from_edge(edge: &Edge) -> Self {
        Self {
            id: edge.id.to_string(),
            source_id: edge.source.to_string(),
            target_id: edge.target.to_string(),
            relation: edge.relation.clone(),
            weight: edge.weight,
            created_at: edge.created_at,
        }
    }

    pub fn to_edge(&self) -> Edge {
        Edge {
            id: Uuid::parse_str(&self.id).unwrap_or_else(|_| Uuid::new_v4()),
            source: Uuid::parse_str(&self.source_id).unwrap_or_else(|_| Uuid::new_v4()),
            target: Uuid::parse_str(&self.target_id).unwrap_or_else(|_| Uuid::new_v4()),
            relation: self.relation.clone(),
            weight: self.weight,
            created_at: self.created_at,
            metadata: None,
        }
    }
}

/// 概念池条目
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConceptPoolEntry {
    pub name: String,
    pub instance_count: u32,
    pub last_used_at: i64,
}

/// 自定义记忆标记记录
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CustomMemoryRecord {
    pub node_id: String,
    pub marked_at: i64,
}

/// 同步元数据记录
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SyncMetadataRecord {
    pub last_sync_at: i64,
    pub version: String,
}

/// 数据库配置
pub struct LanceDBConfig {
    pub db_path: String,
    pub vector_dimension: usize,
}

impl Default for LanceDBConfig {
    fn default() -> Self {
        Self {
            db_path: "./memory_db".to_string(),
            vector_dimension: VECTOR_DIM as usize,
        }
    }
}

/// 表名常量
pub mod table_names {
    pub const NODES: &str = "memory_nodes";
    pub const EDGES: &str = "memory_edges";
    pub const CONCEPT_POOL: &str = "concept_pool";
    pub const CUSTOM_MEMORIES: &str = "custom_memories";
    pub const SYNC_METADATA: &str = "sync_metadata";
}

// ============================================
// Arrow Schema 定义
// ============================================

/// 获取节点表 schema
pub fn nodes_schema() -> SchemaRef {
    Arc::new(Schema::new(vec![
        Field::new("id", DataType::Utf8, false),
        Field::new("node_type", DataType::Utf8, false),
        Field::new("content", DataType::Utf8, false),
        Field::new(
            "vector",
            DataType::FixedSizeList(
                Arc::new(Field::new("item", DataType::Float32, true)),
                VECTOR_DIM,
            ),
            true,
        ),
        Field::new("importance", DataType::Float32, false),
        Field::new("access_count", DataType::UInt32, false),
        Field::new("event_time", DataType::Utf8, true),
        Field::new("created_at", DataType::Int64, false),
        Field::new("updated_at", DataType::Int64, false),
        Field::new("metadata", DataType::Utf8, true),
    ]))
}

/// 获取边表 schema
pub fn edges_schema() -> SchemaRef {
    Arc::new(Schema::new(vec![
        Field::new("id", DataType::Utf8, false),
        Field::new("source_id", DataType::Utf8, false),
        Field::new("target_id", DataType::Utf8, false),
        Field::new("relation", DataType::Utf8, false),
        Field::new("weight", DataType::Float32, false),
        Field::new("created_at", DataType::Int64, false),
    ]))
}

/// 获取概念池表 schema
pub fn concept_pool_schema() -> SchemaRef {
    Arc::new(Schema::new(vec![
        Field::new("name", DataType::Utf8, false),
        Field::new("instance_count", DataType::UInt32, false),
        Field::new("last_used_at", DataType::Int64, false),
    ]))
}

/// 获取自定义记忆标记表 schema
pub fn custom_memories_schema() -> SchemaRef {
    Arc::new(Schema::new(vec![
        Field::new("node_id", DataType::Utf8, false),
        Field::new("marked_at", DataType::Int64, false),
    ]))
}

/// 获取同步元数据表 schema
pub fn sync_metadata_schema() -> SchemaRef {
    Arc::new(Schema::new(vec![
        Field::new("last_sync_at", DataType::Int64, false),
        Field::new("version", DataType::Utf8, false),
    ]))
}

// ============================================
// Arrow RecordBatch 转换（写：领域类型 → Arrow）
// ============================================

/// 将节点列表转换为 Arrow RecordBatch
pub fn nodes_to_batch(records: &[NodeRecord]) -> Result<RecordBatch> {
    let schema = nodes_schema();

    let ids: Vec<&str> = records.iter().map(|r| r.id.as_str()).collect();
    let types: Vec<&str> = records.iter().map(|r| r.node_type.as_str()).collect();
    let contents: Vec<&str> = records.iter().map(|r| r.content.as_str()).collect();
    let importances: Vec<f32> = records.iter().map(|r| r.importance).collect();
    let access_counts: Vec<u32> = records.iter().map(|r| r.access_count).collect();
    let event_times: Vec<Option<&str>> = records.iter()
        .map(|r| r.event_time.as_deref())
        .collect();
    let created_ats: Vec<i64> = records.iter().map(|r| r.created_at).collect();
    let updated_ats: Vec<i64> = records.iter().map(|r| r.updated_at).collect();
    let metadatas: Vec<Option<&str>> = records.iter()
        .map(|r| Some(r.metadata.as_str()))
        .collect();

    // 构建向量列（FixedSizeListArray）
    let flat_vectors: Vec<f32> = records.iter()
        .flat_map(|r| {
            if r.vector.len() == VECTOR_DIM as usize {
                r.vector.clone()
            } else {
                vec![0.0f32; VECTOR_DIM as usize]
            }
        })
        .collect();
    let values_array = Float32Array::from(flat_vectors);
    let field = Arc::new(Field::new("item", DataType::Float32, true));
    let vector_array = FixedSizeListArray::new(field, VECTOR_DIM, Arc::new(values_array), None);

    let columns: Vec<ArrayRef> = vec![
        Arc::new(StringArray::from(ids)),
        Arc::new(StringArray::from(types)),
        Arc::new(StringArray::from(contents)),
        Arc::new(vector_array),
        Arc::new(Float32Array::from(importances)),
        Arc::new(UInt32Array::from(access_counts)),
        Arc::new(StringArray::from(event_times)),
        Arc::new(Int64Array::from(created_ats)),
        Arc::new(Int64Array::from(updated_ats)),
        Arc::new(StringArray::from(metadatas)),
    ];

    RecordBatch::try_new(schema, columns)
        .map_err(|e| MemoryError::Storage(format!("创建节点 RecordBatch 失败: {}", e)))
}

/// 将边列表转换为 Arrow RecordBatch
pub fn edges_to_batch(records: &[EdgeRecord]) -> Result<RecordBatch> {
    let schema = edges_schema();

    let columns: Vec<ArrayRef> = vec![
        Arc::new(StringArray::from(records.iter().map(|r| r.id.as_str()).collect::<Vec<_>>())),
        Arc::new(StringArray::from(records.iter().map(|r| r.source_id.as_str()).collect::<Vec<_>>())),
        Arc::new(StringArray::from(records.iter().map(|r| r.target_id.as_str()).collect::<Vec<_>>())),
        Arc::new(StringArray::from(records.iter().map(|r| r.relation.as_str()).collect::<Vec<_>>())),
        Arc::new(Float32Array::from(records.iter().map(|r| r.weight).collect::<Vec<_>>())),
        Arc::new(Int64Array::from(records.iter().map(|r| r.created_at).collect::<Vec<_>>())),
    ];

    RecordBatch::try_new(schema, columns)
        .map_err(|e| MemoryError::Storage(format!("创建边 RecordBatch 失败: {}", e)))
}

/// 将概念池条目转换为 Arrow RecordBatch
pub fn concepts_to_batch(entries: &[ConceptPoolEntry]) -> Result<RecordBatch> {
    let schema = concept_pool_schema();

    let columns: Vec<ArrayRef> = vec![
        Arc::new(StringArray::from(entries.iter().map(|e| e.name.as_str()).collect::<Vec<_>>())),
        Arc::new(UInt32Array::from(entries.iter().map(|e| e.instance_count).collect::<Vec<_>>())),
        Arc::new(Int64Array::from(entries.iter().map(|e| e.last_used_at).collect::<Vec<_>>())),
    ];

    RecordBatch::try_new(schema, columns)
        .map_err(|e| MemoryError::Storage(format!("创建概念池 RecordBatch 失败: {}", e)))
}

/// 将自定义记忆标记转换为 Arrow RecordBatch
pub fn custom_memories_to_batch(records: &[CustomMemoryRecord]) -> Result<RecordBatch> {
    let schema = custom_memories_schema();

    let columns: Vec<ArrayRef> = vec![
        Arc::new(StringArray::from(records.iter().map(|r| r.node_id.as_str()).collect::<Vec<_>>())),
        Arc::new(Int64Array::from(records.iter().map(|r| r.marked_at).collect::<Vec<_>>())),
    ];

    RecordBatch::try_new(schema, columns)
        .map_err(|e| MemoryError::Storage(format!("创建自定义记忆 RecordBatch 失败: {}", e)))
}

// ============================================
// Arrow RecordBatch 转换（读：Arrow → 领域类型）
// ============================================

/// 从 Arrow RecordBatch 提取节点记录
pub fn batch_to_node_records(batch: &RecordBatch) -> Result<Vec<NodeRecord>> {
    let num_rows = batch.num_rows();
    if num_rows == 0 {
        return Ok(vec![]);
    }

    let id_col = col_as_string(batch, "id")?;
    let type_col = col_as_string(batch, "node_type")?;
    let content_col = col_as_string(batch, "content")?;
    let importance_col = col_as_f32(batch, "importance")?;
    let access_col = col_as_u32(batch, "access_count")?;
    let event_time_col = col_as_string_nullable(batch, "event_time");
    let created_col = col_as_i64(batch, "created_at")?;
    let updated_col = col_as_i64(batch, "updated_at")?;
    let metadata_col = col_as_string_nullable(batch, "metadata");

    // 向量列
    let vector_col = batch.column_by_name("vector")
        .and_then(|c| c.as_any().downcast_ref::<FixedSizeListArray>().map(|a| a.clone()));

    let mut records = Vec::with_capacity(num_rows);
    for i in 0..num_rows {
        let vector = if let Some(ref vc) = vector_col {
            extract_vector(vc, i)
        } else {
            vec![0.0f32; VECTOR_DIM as usize]
        };

        let event_time = event_time_col.as_ref()
            .and_then(|c| if c.is_null(i) { None } else { Some(c.value(i).to_string()) });

        let metadata = metadata_col.as_ref()
            .map(|c| if c.is_null(i) { "{}".to_string() } else { c.value(i).to_string() })
            .unwrap_or_else(|| "{}".to_string());

        records.push(NodeRecord {
            id: id_col.value(i).to_string(),
            node_type: type_col.value(i).to_string(),
            content: content_col.value(i).to_string(),
            vector,
            importance: importance_col.value(i),
            access_count: access_col.value(i),
            event_time,
            created_at: created_col.value(i),
            updated_at: updated_col.value(i),
            metadata,
        });
    }

    Ok(records)
}

/// 从 Arrow RecordBatch 提取边记录
pub fn batch_to_edge_records(batch: &RecordBatch) -> Result<Vec<EdgeRecord>> {
    let num_rows = batch.num_rows();
    if num_rows == 0 {
        return Ok(vec![]);
    }

    let id_col = col_as_string(batch, "id")?;
    let source_col = col_as_string(batch, "source_id")?;
    let target_col = col_as_string(batch, "target_id")?;
    let relation_col = col_as_string(batch, "relation")?;
    let weight_col = col_as_f32(batch, "weight")?;
    let created_col = col_as_i64(batch, "created_at")?;

    let mut records = Vec::with_capacity(num_rows);
    for i in 0..num_rows {
        records.push(EdgeRecord {
            id: id_col.value(i).to_string(),
            source_id: source_col.value(i).to_string(),
            target_id: target_col.value(i).to_string(),
            relation: relation_col.value(i).to_string(),
            weight: weight_col.value(i),
            created_at: created_col.value(i),
        });
    }

    Ok(records)
}

/// 从 Arrow RecordBatch 提取概念池条目
pub fn batch_to_concept_entries(batch: &RecordBatch) -> Result<Vec<ConceptPoolEntry>> {
    let num_rows = batch.num_rows();
    if num_rows == 0 {
        return Ok(vec![]);
    }

    let name_col = col_as_string(batch, "name")?;
    let count_col = col_as_u32(batch, "instance_count")?;
    let last_used_col = col_as_i64(batch, "last_used_at")?;

    let mut entries = Vec::with_capacity(num_rows);
    for i in 0..num_rows {
        entries.push(ConceptPoolEntry {
            name: name_col.value(i).to_string(),
            instance_count: count_col.value(i),
            last_used_at: last_used_col.value(i),
        });
    }

    Ok(entries)
}

/// 从 Arrow RecordBatch 提取自定义记忆 node_id
pub fn batch_to_custom_memory_ids(batch: &RecordBatch) -> Result<Vec<String>> {
    let num_rows = batch.num_rows();
    if num_rows == 0 {
        return Ok(vec![]);
    }

    let id_col = col_as_string(batch, "node_id")?;
    Ok((0..num_rows).map(|i| id_col.value(i).to_string()).collect())
}

// ============================================
// Arrow RecordBatchIterator 构建
// ============================================

/// 创建用于 LanceDB 写入的 RecordBatchIterator
pub fn make_batch_reader(
    batch: RecordBatch,
    schema: SchemaRef,
) -> RecordBatchIterator<std::vec::IntoIter<std::result::Result<RecordBatch, arrow_schema::ArrowError>>> {
    RecordBatchIterator::new(vec![Ok(batch)].into_iter(), schema)
}

// ============================================
// 内部辅助：安全列提取
// ============================================

fn col_as_string<'a>(batch: &'a RecordBatch, name: &str) -> Result<&'a StringArray> {
    batch.column_by_name(name)
        .ok_or_else(|| MemoryError::Storage(format!("缺少列: {}", name)))?
        .as_any()
        .downcast_ref::<StringArray>()
        .ok_or_else(|| MemoryError::Storage(format!("列 {} 类型不匹配，期望 StringArray", name)))
}

fn col_as_string_nullable<'a>(batch: &'a RecordBatch, name: &str) -> Option<&'a StringArray> {
    batch.column_by_name(name)
        .and_then(|c| c.as_any().downcast_ref::<StringArray>())
}

fn col_as_f32<'a>(batch: &'a RecordBatch, name: &str) -> Result<&'a Float32Array> {
    batch.column_by_name(name)
        .ok_or_else(|| MemoryError::Storage(format!("缺少列: {}", name)))?
        .as_any()
        .downcast_ref::<Float32Array>()
        .ok_or_else(|| MemoryError::Storage(format!("列 {} 类型不匹配，期望 Float32Array", name)))
}

fn col_as_u32<'a>(batch: &'a RecordBatch, name: &str) -> Result<&'a UInt32Array> {
    batch.column_by_name(name)
        .ok_or_else(|| MemoryError::Storage(format!("缺少列: {}", name)))?
        .as_any()
        .downcast_ref::<UInt32Array>()
        .ok_or_else(|| MemoryError::Storage(format!("列 {} 类型不匹配，期望 UInt32Array", name)))
}

fn col_as_i64<'a>(batch: &'a RecordBatch, name: &str) -> Result<&'a Int64Array> {
    batch.column_by_name(name)
        .ok_or_else(|| MemoryError::Storage(format!("缺少列: {}", name)))?
        .as_any()
        .downcast_ref::<Int64Array>()
        .ok_or_else(|| MemoryError::Storage(format!("列 {} 类型不匹配，期望 Int64Array", name)))
}

/// 从 FixedSizeListArray 中提取第 i 个向量
fn extract_vector(array: &FixedSizeListArray, i: usize) -> Vec<f32> {
    let dim = VECTOR_DIM as usize;
    let start = i * dim;
    let end = start + dim;

    if let Some(values) = array.values().as_any().downcast_ref::<Float32Array>() {
        if end <= values.len() {
            return (start..end).map(|j| values.value(j)).collect();
        }
    }

    vec![0.0f32; dim]
}
