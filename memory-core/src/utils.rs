//! 通用工具函数
//!
//! 提供时间计算等共享工具，消除模块间的重复代码

/// 计算距今时间的友好显示
///
/// # Arguments
/// * `event_time` - 时间字符串，格式：YYYY-MM-DD-HH-MM
///
/// # Returns
/// 友好的时间差描述，如"3小时前"、"2天前"
pub fn calculate_time_ago(event_time: &str) -> String {
    let parts: Vec<&str> = event_time.split('-').collect();
    if parts.len() != 5 {
        return "未知时间".to_string();
    }

    let year: i32 = parts[0].parse().unwrap_or(2026);
    let month: u32 = parts[1].parse().unwrap_or(1);
    let day: u32 = parts[2].parse().unwrap_or(1);
    let hour: u32 = parts[3].parse().unwrap_or(0);
    let minute: u32 = parts[4].parse().unwrap_or(0);

    let event_date = chrono::NaiveDate::from_ymd_opt(year, month, day)
        .and_then(|d| d.and_hms_opt(hour, minute, 0))
        .map(|dt| chrono::DateTime::<chrono::Utc>::from_naive_utc_and_offset(dt, chrono::Utc));

    let event_date = match event_date {
        Some(d) => d,
        None => return "未知时间".to_string(),
    };

    let now = chrono::Utc::now();
    let diff = now.signed_duration_since(event_date);

    if diff.num_seconds() < 0 {
        return "刚刚".to_string();
    }

    let diff_minutes = diff.num_minutes();
    let diff_hours = diff.num_hours();
    let diff_days = diff.num_days();
    let diff_weeks = diff_days / 7;
    let diff_months = diff_days / 30;

    if diff_minutes < 1 {
        "刚刚".to_string()
    } else if diff_minutes < 60 {
        format!("{}分钟前", diff_minutes)
    } else if diff_hours < 24 {
        format!("{}小时前", diff_hours)
    } else if diff_days < 7 {
        format!("{}天前", diff_days)
    } else if diff_weeks < 4 {
        format!("{}周前", diff_weeks)
    } else {
        format!("{}个月前", diff_months)
    }
}

/// 计算时间新鲜度（用于权重计算）
///
/// # Arguments
/// * `timestamp` - Unix 时间戳（秒）
///
/// # Returns
/// 0.0 到 1.0 之间的值，越新越接近 1.0
pub fn calculate_recency(timestamp: i64) -> f32 {
    let now = chrono::Utc::now().timestamp();
    let diff_secs = (now - timestamp).max(0);

    let one_week_secs = 7 * 24 * 60 * 60;
    let thirty_days_secs = 30 * 24 * 60 * 60;

    if diff_secs <= one_week_secs {
        1.0
    } else if diff_secs <= thirty_days_secs {
        let decay_range = (thirty_days_secs - one_week_secs) as f32;
        let decay_progress = (diff_secs - one_week_secs) as f32 / decay_range;
        1.0 - 0.9 * decay_progress
    } else {
        0.1
    }
}

/// 格式化事件时间为日期字符串
///
/// # Arguments
/// * `event_time` - 时间字符串，格式：YYYY-MM-DD-HH-MM
///
/// # Returns
/// 日期部分 "YYYY-MM-DD"
pub fn format_date(event_time: &str) -> String {
    let parts: Vec<&str> = event_time.split('-').collect();
    if parts.len() >= 3 {
        format!("{}-{}-{}", parts[0], parts[1], parts[2])
    } else {
        "未知日期".to_string()
    }
}

/// 获取当前时间的事件时间格式字符串
///
/// # Returns
/// 格式：YYYY-MM-DD-HH-MM
pub fn now_event_time() -> String {
    chrono::Utc::now().format("%Y-%m-%d-%H-%M").to_string()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_format_date() {
        assert_eq!(format_date("2026-01-15-10-30"), "2026-01-15");
        assert_eq!(format_date("invalid"), "未知日期");
    }

    #[test]
    fn test_calculate_recency() {
        let now = chrono::Utc::now().timestamp();
        assert!((calculate_recency(now) - 1.0).abs() < 0.01);

        let old = now - 60 * 24 * 60 * 60; // 60 days ago
        assert!(calculate_recency(old) < 0.2);
    }
}
