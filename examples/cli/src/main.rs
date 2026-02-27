//! 记忆系统 CLI 示例
//!
//! 展示如何使用 memory-core 库进行记忆存储和检索

use clap::{Parser, Subcommand};
use colored::Colorize;
use memory_core::{MemorySystem, Message, QueryParams};
use std::io::{self, Write};

#[derive(Parser)]
#[command(name = "memory-cli")]
#[command(about = "记忆系统命令行工具")]
#[command(version)]
struct Cli {
    #[command(subcommand)]
    command: Commands,

    /// 数据库路径
    #[arg(short, long, default_value = "./memory_db")]
    db_path: String,
}

#[derive(Subcommand)]
enum Commands {
    /// 初始化记忆系统
    Init,

    /// 保存对话到记忆
    Save {
        /// 对话内容（JSON格式）
        #[arg(short, long)]
        content: Option<String>,

        /// 从文件读取对话
        #[arg(short, long)]
        file: Option<String>,
    },

    /// 查询相关记忆
    Query {
        /// 查询消息
        message: String,

        /// 返回结果数量
        #[arg(short, long, default_value = "5")]
        top_k: usize,
    },

    /// 交互式对话模式
    Chat,

    /// 设置认证Token
    SetToken {
        /// 认证Token
        token: String,
    },

    /// 设置服务端URL
    SetUrl {
        /// 服务端URL
        url: String,
    },
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let cli = Cli::parse();

    match cli.command {
        Commands::Init => {
            println!("{}", "正在初始化记忆系统...".cyan());
            let mut system = MemorySystem::new(Some(&cli.db_path))?;
            system.initialize().await?;
            println!("{}", "✓ 记忆系统初始化完成".green());
        }

        Commands::Save { content, file } => {
            let messages = if let Some(file_path) = file {
                let content = std::fs::read_to_string(file_path)?;
                parse_messages(&content)?
            } else if let Some(content) = content {
                parse_messages(&content)?
            } else {
                println!("{}", "错误: 请提供 --content 或 --file 参数".red());
                return Ok(());
            };

            let system = MemorySystem::new(Some(&cli.db_path))?;
            system.save(&messages).await?;
            println!("{} 保存了 {} 条消息", "✓".green(), messages.len());
        }

        Commands::Query { message, top_k } => {
            let system = MemorySystem::new(Some(&cli.db_path))?;

            let params = QueryParams {
                user_message: message.clone(),
                recent_messages: None,
                top_k: Some(top_k),
                include_raw: Some(true),
            };

            println!("{}: {}", "查询".cyan(), message);
            println!("{}", "-".repeat(50));

            let result = system.query(params).await?;

            if result.count == 0 {
                println!("{}", "未找到相关记忆".yellow());
            } else {
                println!("找到 {} 条相关记忆:\n", result.count);
                println!("{}", result.formatted_context);

                if let Some(raw) = result.raw {
                    println!("\n{}", "详细结果:".cyan());
                    for (i, memory) in raw.iter().enumerate() {
                        println!(
                            "  {}. [{}] {} (相关度: {:.2})",
                            i + 1,
                            format!("{:?}", memory.memory_type).to_lowercase(),
                            memory.content.chars().take(50).collect::<String>(),
                            memory.relevance
                        );
                    }
                }
            }
        }

        Commands::Chat => {
            run_interactive_chat(&cli.db_path).await?;
        }

        Commands::SetToken { token } => {
            let system = MemorySystem::new(Some(&cli.db_path))?;
            system.set_auth_token(token).await;
            println!("{}", "✓ 认证Token已设置".green());
        }

        Commands::SetUrl { url } => {
            let system = MemorySystem::new(Some(&cli.db_path))?;
            system.set_server_url(url).await;
            println!("{}", "✓ 服务端URL已设置".green());
        }
    }

    Ok(())
}

/// 解析消息JSON
fn parse_messages(content: &str) -> anyhow::Result<Vec<Message>> {
    let messages: Vec<Message> = serde_json::from_str(content)?;
    Ok(messages)
}

/// 运行交互式对话模式
async fn run_interactive_chat(db_path: &str) -> anyhow::Result<()> {
    println!("{}", "记忆系统交互式对话模式".cyan().bold());
    println!("{}", "输入 'quit' 或 'exit' 退出\n".dimmed());

    let system = MemorySystem::new(Some(db_path))?;
    let mut conversation_history: Vec<Message> = Vec::new();

    loop {
        print!("{}", "你: ".cyan());
        io::stdout().flush()?;

        let mut input = String::new();
        io::stdin().read_line(&mut input)?;
        let input = input.trim();

        if input.eq_ignore_ascii_case("quit") || input.eq_ignore_ascii_case("exit") {
            println!("{}", "再见!".green());
            break;
        }

        if input.is_empty() {
            continue;
        }

        // 创建用户消息
        let user_message = Message {
            role: "user".to_string(),
            content: input.to_string(),
            timestamp: Some(chrono::Utc::now().timestamp()),
        };

        // 查询相关记忆
        let params = QueryParams {
            user_message: input.to_string(),
            recent_messages: if conversation_history.is_empty() {
                None
            } else {
                Some(conversation_history.clone())
            },
            top_k: Some(3),
            include_raw: None,
        };

        match system.query(params).await {
            Ok(result) => {
                if !result.formatted_context.is_empty() {
                    println!("\n{}", "相关记忆:".yellow());
                    println!("{}", result.formatted_context);
                    println!();
                }
            }
            Err(e) => {
                eprintln!("{} 查询失败: {}", "警告:".yellow(), e);
            }
        }

        // 模拟助手回复
        let assistant_reply = format!("收到你的消息: {}", input);
        println!("{} {}\n", "助手:".green(), assistant_reply);

        // 创建助手消息
        let assistant_message = Message {
            role: "assistant".to_string(),
            content: assistant_reply,
            timestamp: Some(chrono::Utc::now().timestamp()),
        };

        // 保存对话
        conversation_history.push(user_message);
        conversation_history.push(assistant_message);

        // 每5轮对话保存一次到记忆
        if conversation_history.len() >= 10 {
            if let Err(e) = system.save(&conversation_history).await {
                eprintln!("{} 保存记忆失败: {}", "错误:".red(), e);
            } else {
                println!("{}", "[对话已保存到记忆]".dimmed());
            }
            conversation_history.clear();
        }
    }

    // 保存剩余的对话
    if !conversation_history.is_empty() {
        if let Err(e) = system.save(&conversation_history).await {
            eprintln!("{} 保存记忆失败: {}", "错误:".red(), e);
        }
    }

    Ok(())
}
