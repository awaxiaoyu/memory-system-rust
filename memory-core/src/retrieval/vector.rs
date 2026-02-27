//! 向量检索
//!
//! 提供基于向量相似度的检索功能

// use crate::error::Result;
use crate::types::MemoryNode;

/// 向量相似度算法
#[derive(Debug, Clone, Copy, Default)]
pub enum SimilarityMetric {
    /// 余弦相似度（默认）
    #[default]
    Cosine,
    /// 欧氏距离
    Euclidean,
    /// 点积
    DotProduct,
}

/// 向量检索结果
#[derive(Debug, Clone)]
pub struct VectorSearchResult {
    /// 节点 ID
    pub node_id: uuid::Uuid,
    /// 相似度分数
    pub score: f32,
    /// 节点内容
    pub content: String,
    /// 节点类型
    pub node_type: crate::types::NodeType,
}

/// 计算余弦相似度
/// 
/// # Arguments
/// * `a` - 向量 A
/// * `b` - 向量 B
/// 
/// # Returns
/// 相似度分数 [-1, 1]
pub fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
    if a.len() != b.len() || a.is_empty() {
        return 0.0;
    }

    // 使用迭代器优化，支持SIMD自动向量化
    let (dot_product, norm_a, norm_b) = a.iter()
        .zip(b.iter())
        .fold((0.0f32, 0.0f32, 0.0f32), |(dot, na, nb), (x, y)| {
            (dot + x * y, na + x * x, nb + y * y)
        });

    let norm_a = norm_a.sqrt();
    let norm_b = norm_b.sqrt();

    if norm_a == 0.0 || norm_b == 0.0 {
        return 0.0;
    }

    dot_product / (norm_a * norm_b)
}

/// 计算欧氏距离
pub fn euclidean_distance(a: &[f32], b: &[f32]) -> f32 {
    if a.len() != b.len() {
        return f32::MAX;
    }

    // 使用迭代器优化，支持SIMD自动向量化
    a.iter()
        .zip(b.iter())
        .map(|(x, y)| {
            let diff = x - y;
            diff * diff
        })
        .sum::<f32>()
        .sqrt()
}

/// 计算点积
pub fn dot_product(a: &[f32], b: &[f32]) -> f32 {
    if a.len() != b.len() {
        return 0.0;
    }

    a.iter().zip(b.iter()).map(|(x, y)| x * y).sum()
}

/// 计算相似度（根据指定的度量方式）
pub fn calculate_similarity(a: &[f32], b: &[f32], metric: SimilarityMetric) -> f32 {
    match metric {
        SimilarityMetric::Cosine => cosine_similarity(a, b),
        SimilarityMetric::Euclidean => {
            // 将欧氏距离转换为相似度 [0, 1]
            let distance = euclidean_distance(a, b);
            1.0 / (1.0 + distance)
        }
        SimilarityMetric::DotProduct => dot_product(a, b),
    }
}

/// 向量归一化
pub fn normalize_vector(v: &mut [f32]) {
    let norm: f32 = v.iter().map(|x| x * x).sum::<f32>().sqrt();
    if norm > 0.0 {
        for x in v.iter_mut() {
            *x /= norm;
        }
    }
}

/// 向量加权平均
pub fn weighted_average(vectors: &[(&[f32], f32)]) -> Vec<f32> {
    if vectors.is_empty() {
        return Vec::new();
    }

    let dim = vectors[0].0.len();
    let mut result = vec![0.0f32; dim];
    let mut total_weight = 0.0f32;

    for (vec, weight) in vectors {
        if vec.len() != dim {
            continue;
        }
        for (i, &v) in vec.iter().enumerate() {
            result[i] += v * weight;
        }
        total_weight += weight;
    }

    if total_weight > 0.0 {
        for x in result.iter_mut() {
            *x /= total_weight;
        }
    }

    result
}

/// 向量检索配置
#[derive(Debug, Clone)]
pub struct VectorSearchConfig {
    /// 返回结果数量
    pub top_k: usize,
    /// 相似度阈值（低于此阈值的结果将被过滤）
    pub threshold: f32,
    /// 相似度度量方式
    pub metric: SimilarityMetric,
    /// 是否进行归一化
    pub normalize: bool,
}

impl Default for VectorSearchConfig {
    fn default() -> Self {
        Self {
            top_k: 10,
            threshold: 0.0,
            metric: SimilarityMetric::Cosine,
            normalize: true,
        }
    }
}

/// 在节点列表中进行向量检索
/// 
/// 这是一个简单的内存检索实现，用于小规模数据
/// 大规模数据应使用 LanceDB 的向量索引
pub fn vector_search_in_memory(
    query: &[f32],
    nodes: &[MemoryNode],
    config: &VectorSearchConfig,
) -> Vec<VectorSearchResult> {
    let mut results: Vec<VectorSearchResult> = nodes
        .iter()
        .filter(|n| !n.embedding.is_empty())
        .map(|node| {
            let score = calculate_similarity(query, &node.embedding, config.metric);
            VectorSearchResult {
                node_id: node.id,
                score,
                content: node.content.clone(),
                node_type: node.node_type(),
            }
        })
        .filter(|r| r.score >= config.threshold)
        .collect();

    // 按分数降序排序
    results.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap_or(std::cmp::Ordering::Equal));

    // 截取 top_k
    results.truncate(config.top_k);

    results
}

/// 批量计算查询向量与多个节点的相似度
pub fn batch_similarity(
    query: &[f32],
    embeddings: &[Vec<f32>],
    metric: SimilarityMetric,
) -> Vec<f32> {
    embeddings
        .iter()
        .map(|emb| calculate_similarity(query, emb, metric))
        .collect()
}

/// 查询扩展：将多个查询向量合并
/// 
/// 用于 HippoRAG 中的多查询融合
pub fn expand_query(
    primary_query: &[f32],
    context_embeddings: &[Vec<f32>],
    context_weight: f32,
) -> Vec<f32> {
    if context_embeddings.is_empty() {
        return primary_query.to_vec();
    }

    // 计算上下文的平均向量
    let context_avg = weighted_average(
        &context_embeddings
            .iter()
            .map(|e| (e.as_slice(), 1.0))
            .collect::<Vec<_>>(),
    );

    // 加权组合
    let primary_weight = 1.0 - context_weight;
    let mut result: Vec<f32> = primary_query
        .iter()
        .zip(context_avg.iter())
        .map(|(&p, &c)| p * primary_weight + c * context_weight)
        .collect();

    // 归一化
    normalize_vector(&mut result);

    result
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cosine_similarity() {
        let a = vec![1.0, 0.0, 0.0];
        let b = vec![1.0, 0.0, 0.0];
        assert!((cosine_similarity(&a, &b) - 1.0).abs() < 1e-6);

        let c = vec![0.0, 1.0, 0.0];
        assert!((cosine_similarity(&a, &c) - 0.0).abs() < 1e-6);

        let d = vec![-1.0, 0.0, 0.0];
        assert!((cosine_similarity(&a, &d) - (-1.0)).abs() < 1e-6);
    }

    #[test]
    fn test_euclidean_distance() {
        let a = vec![0.0, 0.0, 0.0];
        let b = vec![3.0, 4.0, 0.0];
        assert!((euclidean_distance(&a, &b) - 5.0).abs() < 1e-6);
    }

    #[test]
    fn test_normalize_vector() {
        let mut v = vec![3.0, 4.0];
        normalize_vector(&mut v);
        assert!((v[0] - 0.6).abs() < 1e-6);
        assert!((v[1] - 0.8).abs() < 1e-6);
    }

    #[test]
    fn test_weighted_average() {
        let v1 = vec![1.0, 0.0];
        let v2 = vec![0.0, 1.0];
        let result = weighted_average(&[(&v1, 1.0), (&v2, 1.0)]);
        assert!((result[0] - 0.5).abs() < 1e-6);
        assert!((result[1] - 0.5).abs() < 1e-6);
    }
}
