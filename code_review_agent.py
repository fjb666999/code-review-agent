#!/usr/bin/env python3
"""
增强版多 Agent 代码审查系统
- 并行执行无依赖 Agent
- 动态条件执行
- 自动重试与错误恢复
"""

import os
import sys
import time
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, Any, List, Callable, Optional
from openai import OpenAI

# ---------- 配置 ----------
logging.basicConfig(level=logging.INFO, format='[%(asctime)s] %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

DEEPSEEK_API_KEY = os.environ.get("DEEPSEEK_API_KEY")
if not DEEPSEEK_API_KEY:
    logger.error("请设置环境变量 DEEPSEEK_API_KEY")
    sys.exit(1)

client = OpenAI(api_key=DEEPSEEK_API_KEY, base_url="https://api.deepseek.com/v1")
MODEL_NAME = "deepseek-chat"

# ---------- 增强版 API 调用（带重试） ----------
def call_deepseek_with_retry(system_prompt: str, user_prompt: str, temperature: float = 0.3,
                             max_retries: int = 3, base_delay: float = 1.0) -> str:
    """带指数退避重试的 API 调用"""
    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model=MODEL_NAME,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=temperature,
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            logger.warning(f"API 调用失败 (尝试 {attempt+1}/{max_retries}): {e}")
            if attempt == max_retries - 1:
                return f"API 错误（重试 {max_retries} 次后放弃）: {str(e)}"
            time.sleep(base_delay * (2 ** attempt))  # 指数退避
    return "未知错误"  # 不会执行到这里

# ---------- Agent 基类（增强版） ----------
class BaseAgent:
    """Agent 基类，支持条件执行和结果合并"""
    def __init__(self, name: str, depends_on: List[str] = None, condition: Callable[[Dict], bool] = None):
        self.name = name
        self.depends_on = depends_on or []      # 依赖的 Agent 名称列表
        self.condition = condition or (lambda ctx: True)  # 运行条件

    def should_run(self, context: Dict[str, Any]) -> bool:
        """根据上下文判断是否应该运行本 Agent"""
        return self.condition(context)

    def run(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """执行 Agent 主要逻辑，必须更新 context 并返回"""
        raise NotImplementedError

# ---------- 具体 Agent 实现 ----------
class StyleAgent(BaseAgent):
    def __init__(self):
        super().__init__(
            name="StyleAgent",
            depends_on=[],   # 无依赖，可第一时间执行
            condition=lambda ctx: bool(ctx.get("code"))
        )

    def run(self, context: Dict[str, Any]) -> Dict[str, Any]:
        code = context["code"]
        logger.info(f"[{self.name}] 开始风格检查...")
        system_prompt = """你是一位严格的代码风格审查专家，遵循 PEP8 规范。请检查以下 Python 代码，列出所有风格问题。
输出格式：
- 每条问题指出具体行号和违反规范。
- 如果完全符合，回复“未发现风格问题”。"""
        user_prompt = f"```python\n{code}\n```"
        result = call_deepseek_with_retry(system_prompt, user_prompt)
        context["style_issues"] = result
        logger.info(f"[{self.name}] 完成")
        return context

class ReviewAgent(BaseAgent):
    def __init__(self):
        super().__init__(
            name="ReviewAgent",
            depends_on=[],   # 不依赖 StyleAgent，可并行
            condition=lambda ctx: bool(ctx.get("code"))
        )

    def run(self, context: Dict[str, Any]) -> Dict[str, Any]:
        code = context["code"]
        logger.info(f"[{self.name}] 开始深度审查...")
        system_prompt = """你是一位资深代码审查专家。请分析以下 Python 代码，指出逻辑、性能、安全、可维护性问题。
每条包含：问题类型、行号（近似）、修复建议。"""
        user_prompt = f"```python\n{code}\n```"
        result = call_deepseek_with_retry(system_prompt, user_prompt, temperature=0.4)
        context["review_comments"] = result
        logger.info(f"[{self.name}] 完成")
        return context

class PRSummaryAgent(BaseAgent):
    def __init__(self):
        # 依赖前两个 Agent 的结果，只有两者都完成后才运行
        super().__init__(
            name="PRSummaryAgent",
            depends_on=["StyleAgent", "ReviewAgent"],
            condition=lambda ctx: "style_issues" in ctx and "review_comments" in ctx
        )

    def run(self, context: Dict[str, Any]) -> Dict[str, Any]:
        logger.info(f"[{self.name}] 开始生成 PR 建议...")
        system_prompt = """你是一位技术团队 PR 管理员。请根据风格问题和深度审查结果，生成专业、可操作的 PR 评论。
使用 Markdown 格式，包含：总体评价、风格问题列表、深度审查问题（按优先级）、推荐改进步骤。"""
        user_prompt = f"""
风格问题：
{context['style_issues']}

深度审查意见：
{context['review_comments']}

原始代码：
```python
{context['code']}
