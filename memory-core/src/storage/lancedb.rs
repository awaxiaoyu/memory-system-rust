//! LanceDB 存储层实现
//!
//! 基于 LanceDB 的本地向量数据库存储

use crate::error::{MemoryError, Result};
use crate::storage::schema::*;
use crate::types::*;
use futures::TryStreamExt;
use lancedb::query::{QueryBase, ExecutableQuery};
use std::collections::HashSet;
use std::sync::Arc;

/// LanceDB 存储服务
///
/// 提供本地向量数据库的所有操作
pub struct LanceDBStorage {
    db_path: String,
    db: Option<lancedb::Connection>,
    initialized: bool,
}

impl LanceDBStorage {
    /// 创建新的存储实例
    pub fn new(db_path: &str) -> Result<Self> {
        Ok(Self {
            db_path: db_path.to_string(),
            db: None,
            initialized: false,
        })
    }

    /// 初始化数据库连接和表结构
    pub async fn initialize(&mut self) -> Result<()> {
        if self.initialized {
            return Ok(());
        }

        log::info!("正在连接 LanceDB: {}", self.db_path);

        let db = lancedb::connect(&self.db_path)
            .execute()
            .await
            .map_err(|e| MemoryError::Storage(format!("连接 LanceDB 失败: {}", e)))?;

        // 确保所有必需的表存在
        self.ensure_tables_exist(&db).await?;

        self.db = Some(db);
        self.initialized = true;
        log::info!("LanceDB 初始化成功");
        Ok(())
    }

    /// 确保所有必需的表都存在
    async fn ensure_tables_exist(&self, db: &lancedb::Connection) -> Result<()> {
        let existing = db.table_names()
            .execute()
            .await
            .map_err(|e| MemoryError::Storage(format!("获取表名列表失败: {}", e)))?;

        let existing_set: HashSet<&str> = existing.iter().map(|s| s.as_str()).collect();

        // 节点表
        if !existing_set.contains(table_names::NODES) {
            log::info!("创建节点表: {}", table_names::NODES);
            db.create_empty_table(table_names::NODES, nodes_schema())
                .execute()
                .await
                .map_err(|e| MemoryError::Storage(format!("创建节点表失败: {}", e)))?;
        }

        // 边表
        if !existing_set.contains(table_names::EDGES) {
            log::info!("创建边表: {}", table_names::EDGES);
            db.create_empty_table(table_names::EDGES, edges_schema())
                .execute()
                .await
                .map_err(|e| MemoryError::Storage(format!("创建边表失败: {}", e)))?;
        }

        // 概念池表
        if !existing_set.contains(table_names::CONCEPT_POOL) {
            log::info!("创建概念池表: {}", table_names::CONCEPT_POOL);
            db.create_empty_table(table_names::CONCEPT_POOL, concept_pool_schema())
                .execute()
                .await
                .map_err(|e| MemoryError::Storage(format!("创建概念池表失败: {}", e)))?;
        }

        // 自定义记忆标记表
        if !existing_set.contains(table_names::CUSTOM_MEMORIES) {
            log::info!("创建自定义记忆标记表: {}", table_names::CUSTOM_MEMORIES);
            db.create_empty_table(table_names::CUSTOM_MEMORIES, custom_memories_schema())
                .execute()
                .await
                .map_err(|e| MemoryError::Storage(format!("创建自定义记忆标记表失败: {}", e)))?;
        }

        // 同步元数据表
        if !existing_set.contains(table_names::SYNC_METADATA) {
            log::info!("创建同步元数据表: {}", table_names::SYNC_METADATA);
            db.create_empty_table(table_names::SYNC_METADATA, sync_metadata_schema())
                .execute()
                .await
                .map_err(|e| MemoryError::Storage(format!("创建同步元数据表失败: {}", e)))?;
        }

        Ok(())
    }

    /// 获取数据库连接（内部辅助）
    fn db(&self) -> Result<&lancedb::Connection> {
        self.db.as_ref().ok_or(MemoryError::NotInitialized)
    }

    /// 打开指定表
    async fn open_table(&self, name: &str) -> Result<lancedb::Table> {
        self.db()?.open_table(name)
            .execute()
            .await
            .map_err(|e| MemoryError::Storage(format!("打开表 {} 失败: {}", name, e)))
    }

    // ============================================
    // 节点操作
    // ============================================

    /// 添加单个节点
    pub async fn add_node(&self, node: &MemoryNode) -> Result<()> {
        self.check_initialized()?;
        self.add_nodes(&[node.clone()]).await
    }

    /// 批量添加节点
    pub async fn add_nodes(&self, nodes: &[MemoryNode]) -> Result<()> {
        if nodes.is_empty() {
            return Ok(());
        }
        self.check_initialized()?;

        let records: Vec<NodeRecord> = nodes.iter().map(NodeRecord::from_node).collect();
        let batch = nodes_to_batch(&records)?;
        let schema = nodes_schema();

        let table = self.open_table(table_names::NODES).await?;
        let reader = make_batch_reader(batch, schema);
        table.add(reader)
            .execute()
            .await
            .map_err(|e| MemoryError::Storage(format!("添加节点失败: {}", e)))?;

        log::debug!("成功添加 {} 个节点", nodes.len());
        Ok(())
    }

    /// 获取单个节点
    pub async fn get_node(&self, id: &uuid::Uuid) -> Result<Option<MemoryNode>> {
        self.check_initialized()?;

        let table = self.open_table(table_names::NODES).await?;
        let filter = format!("id = '{}'", id);

        let batches = table.query()
            .only_if(filter)
            .limit(1)
            .execute()
            .await
            .map_err(|e| MemoryError::Storage(format!("查询节点失败: {}", e)))?
            .try_collect::<Vec<_>>()
            .await
            .map_err(|e| MemoryError::Storage(format!("收集查询结果失败: {}", e)))?;

        for batch in &batches {
            let records = batch_to_node_records(batch)?;
            if let Some(record) = records.into_iter().next() {
                return Ok(Some(record.to_node()?));
            }
        }

        Ok(None)
    }

    /// 批量获取节点
    pub async fn get_nodes(&self, ids: &[uuid::Uuid]) -> Result<Vec<MemoryNode>> {
        self.check_initialized()?;
        if ids.is_empty() {
            return Ok(vec![]);
        }

        let id_list: Vec<String> = ids.iter().map(|id| format!("'{}'", id)).collect();
        let filter = format!("id IN ({})", id_list.join(", "));

        let table = self.open_table(table_names::NODES).await?;
        let batches = table.query()
            .only_if(filter)
            .execute()
            .await
            .map_err(|e| MemoryError::Storage(format!("批量查询节点失败: {}", e)))?
            .try_collect::<Vec<_>>()
            .await
            .map_err(|e| MemoryError::Storage(format!("收集查询结果失败: {}", e)))?;

        let mut nodes = Vec::new();
        for batch in &batches {
            let records = batch_to_node_records(batch)?;
            for record in records {
                match record.to_node() {
                    Ok(node) => nodes.push(node),
                    Err(e) => log::error!("反序列化节点失败: {}", e),
                }
            }
        }

        Ok(nodes)
    }

    /// 更新节点（删除旧记录并插入新记录）
    pub async fn update_node(&self, node: &MemoryNode) -> Result<()> {
        self.check_initialized()?;

        let table = self.open_table(table_names::NODES).await?;
        let predicate = format!("id = '{}'", node.id);

        table.delete(&predicate)
            .await
            .map_err(|e| MemoryError::Storage(format!("删除旧节点失败: {}", e)))?;

        self.add_node(node).await?;

        log::debug!("更新节点: {}", node.id);
        Ok(())
    }

    /// 删除节点
    pub async fn delete_node(&self, id: &uuid::Uuid) -> Result<()> {
        self.check_initialized()?;

        let table = self.open_table(table_names::NODES).await?;
        let predicate = format!("id = '{}'", id);

        table.delete(&predicate)
            .await
            .map_err(|e| MemoryError::Storage(format!("删除节点失败: {}", e)))?;

        log::debug!("已删除节点: {}", id);
        Ok(())
    }

    /// 按类型获取所有节点
    pub async fn get_nodes_by_type(&self, node_type: NodeType) -> Result<Vec<MemoryNode>> {
        self.check_initialized()?;

        let type_str = match node_type {
            NodeType::Entity => "entity",
            NodeType::Event => "event",
            NodeType::Concept => "concept",
        };

        let table = self.open_table(table_names::NODES).await?;
        let filter = format!("node_type = '{}'", type_str);

        let batches = table.query()
            .only_if(filter)
            .execute()
            .await
            .map_err(|e| MemoryError::Storage(format!("按类型查询节点失败: {}", e)))?
            .try_collect::<Vec<_>>()
            .await
            .map_err(|e| MemoryError::Storage(format!("收集查询结果失败: {}", e)))?;

        let mut nodes = Vec::new();
        for batch in &batches {
            let records = batch_to_node_records(batch)?;
            for record in records {
                match record.to_node() {
                    Ok(node) => nodes.push(node),
                    Err(e) => log::error!("反序列化节点失败: {}", e),
                }
            }
        }

        Ok(nodes)
    }

    /// 获取所有节点
    pub async fn get_all_nodes(&self) -> Result<Vec<MemoryNode>> {
        self.check_initialized()?;

        let table = self.open_table(table_names::NODES).await?;

        let batches = table.query()
            .execute()
            .await
            .map_err(|e| MemoryError::Storage(format!("查询所有节点失败: {}", e)))?
            .try_collect::<Vec<_>>()
            .await
            .map_err(|e| MemoryError::Storage(format!("收集查询结果失败: {}", e)))?;

        let mut nodes = Vec::new();
        for batch in &batches {
            let records = batch_to_node_records(batch)?;
            for record in records {
                match record.to_node() {
                    Ok(node) => nodes.push(node),
                    Err(e) => log::error!("反序列化节点失败: {}", e),
                }
            }
        }

        Ok(nodes)
    }

    // ============================================
    // 向量检索
    // ============================================

    /// 向量相似度检索
    pub async fn vector_search(
        &self,
        query_vector: &[f32],
        limit: usize,
        filter: Option<VectorSearchFilter>,
    ) -> Result<Vec<(MemoryNode, f32)>> {
        self.check_initialized()?;

        let table = self.open_table(table_names::NODES).await?;

        let mut query = table.vector_search(query_vector)
            .map_err(|e| MemoryError::Storage(format!("创建向量搜索失败: {}", e)))?;

        query = query.limit(limit);

        // 应用过滤条件
        if let Some(f) = filter {
            if let Some(nt) = f.node_type {
                let type_str = match nt {
                    NodeType::Entity => "entity",
                    NodeType::Event => "event",
                    NodeType::Concept => "concept",
                };
                query = query.only_if(format!("node_type = '{}'", type_str));
            }
        }

        let batches = query.execute()
            .await
            .map_err(|e| MemoryError::Storage(format!("执行向量搜索失败: {}", e)))?
            .try_collect::<Vec<_>>()
            .await
            .map_err(|e| MemoryError::Storage(format!("收集向量搜索结果失败: {}", e)))?;

        let mut results = Vec::new();
        for batch in &batches {
            let records = batch_to_node_records(batch)?;
            // LanceDB 向量搜索结果包含 _distance 列
            let distance_col = batch.column_by_name("_distance")
                .and_then(|c| c.as_any().downcast_ref::<arrow_array::Float32Array>().map(|a| a.clone()));

            for (i, record) in records.into_iter().enumerate() {
                match record.to_node() {
                    Ok(node) => {
                        // 距离 → 相似度（1 / (1 + distance)）
                        let distance = distance_col.as_ref()
                            .map(|d| d.value(i))
                            .unwrap_or(0.0);
                        let similarity = 1.0 / (1.0 + distance);
                        results.push((node, similarity));
                    }
                    Err(e) => log::error!("反序列化向量搜索结果节点失败: {}", e),
                }
            }
        }

        Ok(results)
    }

    // ============================================
    // 边操作
    // ============================================

    /// 添加边
    pub async fn add_edge(&self, edge: &Edge) -> Result<()> {
        self.add_edges(&[edge.clone()]).await
    }

    /// 批量添加边
    pub async fn add_edges(&self, edges: &[Edge]) -> Result<()> {
        if edges.is_empty() {
            return Ok(());
        }
        self.check_initialized()?;

        let records: Vec<EdgeRecord> = edges.iter().map(EdgeRecord::from_edge).collect();
        let batch = edges_to_batch(&records)?;
        let schema = edges_schema();

        let table = self.open_table(table_names::EDGES).await?;
        let reader = make_batch_reader(batch, schema);
        table.add(reader)
            .execute()
            .await
            .map_err(|e| MemoryError::Storage(format!("添加边失败: {}", e)))?;

        log::debug!("成功添加 {} 条边", edges.len());
        Ok(())
    }

    /// 获取节点的所有出边
    pub async fn get_outgoing_edges(&self, node_id: &uuid::Uuid) -> Result<Vec<Edge>> {
        self.check_initialized()?;

        let table = self.open_table(table_names::EDGES).await?;
        let filter = format!("source_id = '{}'", node_id);

        let batches = table.query()
            .only_if(filter)
            .execute()
            .await
            .map_err(|e| MemoryError::Storage(format!("查询出边失败: {}", e)))?
            .try_collect::<Vec<_>>()
            .await
            .map_err(|e| MemoryError::Storage(format!("收集查询结果失败: {}", e)))?;

        let mut edges = Vec::new();
        for batch in &batches {
            let records = batch_to_edge_records(batch)?;
            edges.extend(records.into_iter().map(|r| r.to_edge()));
        }

        Ok(edges)
    }

    /// 获取节点的所有入边
    pub async fn get_incoming_edges(&self, node_id: &uuid::Uuid) -> Result<Vec<Edge>> {
        self.check_initialized()?;

        let table = self.open_table(table_names::EDGES).await?;
        let filter = format!("target_id = '{}'", node_id);

        let batches = table.query()
            .only_if(filter)
            .execute()
            .await
            .map_err(|e| MemoryError::Storage(format!("查询入边失败: {}", e)))?
            .try_collect::<Vec<_>>()
            .await
            .map_err(|e| MemoryError::Storage(format!("收集查询结果失败: {}", e)))?;

        let mut edges = Vec::new();
        for batch in &batches {
            let records = batch_to_edge_records(batch)?;
            edges.extend(records.into_iter().map(|r| r.to_edge()));
        }

        Ok(edges)
    }

    /// 获取节点的所有边（出边 + 入边）
    pub async fn get_node_edges(&self, node_id: &uuid::Uuid) -> Result<Vec<Edge>> {
        let (outgoing, incoming) = tokio::join!(
            self.get_outgoing_edges(node_id),
            self.get_incoming_edges(node_id)
        );
        let mut edges = outgoing?;
        edges.extend(incoming?);
        Ok(edges)
    }

    /// 获取相邻节点 ID
    pub async fn get_neighbor_ids(&self, node_id: &uuid::Uuid) -> Result<Vec<uuid::Uuid>> {
        let edges = self.get_node_edges(node_id).await?;
        let mut neighbor_ids = HashSet::new();
        for edge in edges {
            if edge.source == *node_id {
                neighbor_ids.insert(edge.target);
            } else {
                neighbor_ids.insert(edge.source);
            }
        }
        Ok(neighbor_ids.into_iter().collect())
    }

    /// 获取所有边
    pub async fn get_all_edges(&self) -> Result<Vec<Edge>> {
        self.check_initialized()?;

        let table = self.open_table(table_names::EDGES).await?;

        let batches = table.query()
            .execute()
            .await
            .map_err(|e| MemoryError::Storage(format!("查询所有边失败: {}", e)))?
            .try_collect::<Vec<_>>()
            .await
            .map_err(|e| MemoryError::Storage(format!("收集查询结果失败: {}", e)))?;

        let mut edges = Vec::new();
        for batch in &batches {
            let records = batch_to_edge_records(batch)?;
            edges.extend(records.into_iter().map(|r| r.to_edge()));
        }

        Ok(edges)
    }

    /// 删除边
    pub async fn delete_edge(&self, id: &uuid::Uuid) -> Result<()> {
        self.check_initialized()?;

        let table = self.open_table(table_names::EDGES).await?;
        table.delete(&format!("id = '{}'", id))
            .await
            .map_err(|e| MemoryError::Storage(format!("删除边失败: {}", e)))?;

        log::debug!("已删除边: {}", id);
        Ok(())
    }

    /// 删除与节点相关的所有边
    pub async fn delete_node_edges(&self, node_id: &uuid::Uuid) -> Result<()> {
        self.check_initialized()?;

        let table = self.open_table(table_names::EDGES).await?;
        let predicate = format!("source_id = '{}' OR target_id = '{}'", node_id, node_id);

        table.delete(&predicate)
            .await
            .map_err(|e| MemoryError::Storage(format!("删除节点边失败: {}", e)))?;

        log::debug!("已删除节点 {} 的所有边", node_id);
        Ok(())
    }

    // ============================================
    // 概念池操作
    // ============================================

    /// 添加或更新概念
    pub async fn upsert_concept(&self, name: &str) -> Result<()> {
        self.check_initialized()?;

        let table = self.open_table(table_names::CONCEPT_POOL).await?;
        let filter = format!("name = '{}'", name.replace('\'', "''"));

        // 检查是否已存在
        let batches = table.query()
            .only_if(filter.clone())
            .limit(1)
            .execute()
            .await
            .map_err(|e| MemoryError::Storage(format!("查询概念失败: {}", e)))?
            .try_collect::<Vec<_>>()
            .await
            .map_err(|e| MemoryError::Storage(format!("收集查询结果失败: {}", e)))?;

        let existing = batches.iter()
            .flat_map(|b| batch_to_concept_entries(b).unwrap_or_default())
            .next();

        let now = chrono::Utc::now().timestamp();

        if let Some(existing_entry) = existing {
            // 已存在：删除旧记录，插入更新后的记录
            table.delete(&filter)
                .await
                .map_err(|e| MemoryError::Storage(format!("删除旧概念失败: {}", e)))?;

            let updated = ConceptPoolEntry {
                name: existing_entry.name,
                instance_count: existing_entry.instance_count + 1,
                last_used_at: now,
            };
            let batch = concepts_to_batch(&[updated])?;
            let reader = make_batch_reader(batch, concept_pool_schema());
            table.add(reader)
                .execute()
                .await
                .map_err(|e| MemoryError::Storage(format!("更新概念失败: {}", e)))?;
        } else {
            // 不存在：插入新记录
            let entry = ConceptPoolEntry {
                name: name.to_string(),
                instance_count: 1,
                last_used_at: now,
            };
            let batch = concepts_to_batch(&[entry])?;
            let reader = make_batch_reader(batch, concept_pool_schema());
            table.add(reader)
                .execute()
                .await
                .map_err(|e| MemoryError::Storage(format!("添加概念失败: {}", e)))?;
        }

        log::debug!("已 upsert 概念: {}", name);
        Ok(())
    }

    /// 批量添加或更新概念
    pub async fn upsert_concepts(&self, names: &[String]) -> Result<()> {
        for name in names {
            self.upsert_concept(name).await?;
        }
        Ok(())
    }

    /// 获取活跃概念（按使用频率排序）
    pub async fn get_active_concepts(&self, limit: usize) -> Result<Vec<ConceptPoolEntry>> {
        self.check_initialized()?;

        let table = self.open_table(table_names::CONCEPT_POOL).await?;

        let batches = table.query()
            .limit(limit * 3) // 多取一些以便排序
            .execute()
            .await
            .map_err(|e| MemoryError::Storage(format!("查询活跃概念失败: {}", e)))?
            .try_collect::<Vec<_>>()
            .await
            .map_err(|e| MemoryError::Storage(format!("收集查询结果失败: {}", e)))?;

        let mut entries = Vec::new();
        for batch in &batches {
            entries.extend(batch_to_concept_entries(batch)?);
        }

        // 按 instance_count 降序排序
        entries.sort_by(|a, b| b.instance_count.cmp(&a.instance_count));
        entries.truncate(limit);

        Ok(entries)
    }

    /// 获取所有概念
    pub async fn get_all_concepts(&self) -> Result<Vec<ConceptPoolEntry>> {
        self.check_initialized()?;

        let table = self.open_table(table_names::CONCEPT_POOL).await?;

        let batches = table.query()
            .execute()
            .await
            .map_err(|e| MemoryError::Storage(format!("查询所有概念失败: {}", e)))?
            .try_collect::<Vec<_>>()
            .await
            .map_err(|e| MemoryError::Storage(format!("收集查询结果失败: {}", e)))?;

        let mut entries = Vec::new();
        for batch in &batches {
            entries.extend(batch_to_concept_entries(batch)?);
        }

        Ok(entries)
    }

    /// 删除低频概念
    pub async fn prune_inactive_concepts(
        &self,
        min_instance_count: u32,
        max_age_ms: i64,
    ) -> Result<usize> {
        self.check_initialized()?;

        let cutoff_time = chrono::Utc::now().timestamp_millis() - max_age_ms;

        let table = self.open_table(table_names::CONCEPT_POOL).await?;
        let predicate = format!(
            "instance_count < {} AND last_used_at < {}",
            min_instance_count, cutoff_time
        );

        // 先查询要删除的数量
        let batches = table.query()
            .only_if(predicate.clone())
            .execute()
            .await
            .map_err(|e| MemoryError::Storage(format!("查询待删除概念失败: {}", e)))?
            .try_collect::<Vec<_>>()
            .await
            .map_err(|e| MemoryError::Storage(format!("收集查询结果失败: {}", e)))?;

        let count: usize = batches.iter().map(|b| b.num_rows()).sum();

        if count > 0 {
            table.delete(&predicate)
                .await
                .map_err(|e| MemoryError::Storage(format!("删除低频概念失败: {}", e)))?;
            log::info!("已删除 {} 个低频概念", count);
        }

        Ok(count)
    }

    // ============================================
    // 自定义记忆标记
    // ============================================

    /// 标记自定义记忆
    pub async fn mark_custom_memory(&self, node_id: &uuid::Uuid) -> Result<()> {
        self.check_initialized()?;

        // 检查是否已标记
        if self.is_custom_memory(node_id).await? {
            return Ok(());
        }

        let record = CustomMemoryRecord {
            node_id: node_id.to_string(),
            marked_at: chrono::Utc::now().timestamp(),
        };

        let batch = custom_memories_to_batch(&[record])?;
        let schema = custom_memories_schema();

        let table = self.open_table(table_names::CUSTOM_MEMORIES).await?;
        let reader = make_batch_reader(batch, schema);
        table.add(reader)
            .execute()
            .await
            .map_err(|e| MemoryError::Storage(format!("标记自定义记忆失败: {}", e)))?;

        log::debug!("已标记自定义记忆: {}", node_id);
        Ok(())
    }

    /// 取消自定义记忆标记
    pub async fn unmark_custom_memory(&self, node_id: &uuid::Uuid) -> Result<()> {
        self.check_initialized()?;

        let table = self.open_table(table_names::CUSTOM_MEMORIES).await?;
        table.delete(&format!("node_id = '{}'", node_id))
            .await
            .map_err(|e| MemoryError::Storage(format!("取消自定义记忆标记失败: {}", e)))?;

        log::debug!("已取消自定义记忆标记: {}", node_id);
        Ok(())
    }

    /// 获取所有自定义标记的记忆 ID
    pub async fn get_custom_memory_ids(&self) -> Result<HashSet<uuid::Uuid>> {
        self.check_initialized()?;

        let table = self.open_table(table_names::CUSTOM_MEMORIES).await?;

        let batches = table.query()
            .execute()
            .await
            .map_err(|e| MemoryError::Storage(format!("查询自定义记忆标记失败: {}", e)))?
            .try_collect::<Vec<_>>()
            .await
            .map_err(|e| MemoryError::Storage(format!("收集查询结果失败: {}", e)))?;

        let mut ids = HashSet::new();
        for batch in &batches {
            let id_strings = batch_to_custom_memory_ids(batch)?;
            for id_str in id_strings {
                match uuid::Uuid::parse_str(&id_str) {
                    Ok(id) => { ids.insert(id); }
                    Err(e) => log::error!("解析自定义记忆 UUID 失败: {} - {}", id_str, e),
                }
            }
        }

        Ok(ids)
    }

    /// 检查节点是否被标记为自定义记忆
    pub async fn is_custom_memory(&self, node_id: &uuid::Uuid) -> Result<bool> {
        self.check_initialized()?;

        let table = self.open_table(table_names::CUSTOM_MEMORIES).await?;
        let filter = format!("node_id = '{}'", node_id);

        let batches = table.query()
            .only_if(filter)
            .limit(1)
            .execute()
            .await
            .map_err(|e| MemoryError::Storage(format!("查询自定义记忆标记失败: {}", e)))?
            .try_collect::<Vec<_>>()
            .await
            .map_err(|e| MemoryError::Storage(format!("收集查询结果失败: {}", e)))?;

        Ok(batches.iter().any(|b| b.num_rows() > 0))
    }

    // ============================================
    // 同步元数据
    // ============================================

    /// 获取最后同步时间
    pub async fn get_last_sync_time(&self) -> Result<i64> {
        self.check_initialized()?;

        let table = self.open_table(table_names::SYNC_METADATA).await?;

        let batches = table.query()
            .limit(1)
            .execute()
            .await
            .map_err(|e| MemoryError::Storage(format!("查询同步元数据失败: {}", e)))?
            .try_collect::<Vec<_>>()
            .await
            .map_err(|e| MemoryError::Storage(format!("收集查询结果失败: {}", e)))?;

        for batch in &batches {
            if batch.num_rows() > 0 {
                if let Some(col) = batch.column_by_name("last_sync_at") {
                    if let Some(arr) = col.as_any().downcast_ref::<arrow_array::Int64Array>() {
                        return Ok(arr.value(0));
                    }
                }
            }
        }

        Ok(0) // 从未同步过
    }

    /// 更新同步元数据
    pub async fn update_sync_metadata(&self, timestamp: i64, version: &str) -> Result<()> {
        self.check_initialized()?;

        // 清空旧记录
        let table = self.open_table(table_names::SYNC_METADATA).await?;
        // 尝试删除所有记录（忽略空表错误）
        let _ = table.delete("last_sync_at >= 0").await;

        // 插入新记录
        let schema = sync_metadata_schema();
        let columns: Vec<Arc<dyn arrow_array::Array>> = vec![
            Arc::new(arrow_array::Int64Array::from(vec![timestamp])),
            Arc::new(arrow_array::StringArray::from(vec![version])),
        ];
        let batch = arrow_array::RecordBatch::try_new(schema.clone(), columns)
            .map_err(|e| MemoryError::Storage(format!("创建同步元数据 RecordBatch 失败: {}", e)))?;

        let reader = make_batch_reader(batch, schema);
        table.add(reader)
            .execute()
            .await
            .map_err(|e| MemoryError::Storage(format!("更新同步元数据失败: {}", e)))?;

        log::debug!("已更新同步元数据: {} @ {}", version, timestamp);
        Ok(())
    }

    // ============================================
    // 内部方法
    // ============================================

    fn check_initialized(&self) -> Result<()> {
        if !self.initialized {
            return Err(MemoryError::NotInitialized);
        }
        Ok(())
    }
}

/// 向量检索过滤器
#[derive(Debug, Clone, Default)]
pub struct VectorSearchFilter {
    pub node_type: Option<NodeType>,
}
