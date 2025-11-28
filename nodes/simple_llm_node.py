import json
import time
from typing import Tuple

import requests


class SimpleLLMNode:
    """
    ComfyUI node that performs a single round-trip LLM request against a configurable API.
    """

    def __init__(self):
        pass

    @classmethod
    def IS_CHANGED(cls, **kwargs):
        # Force execution even when inputs stay the same so each run triggers a fresh API call.
        return str(time.time_ns())

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "api_key": (
                    "STRING",
                    {
                        "default": "",
                        "multiline": False,
                        "placeholder": "输入你的API密钥",
                    },
                ),
                "model": (
                    "STRING",
                    {
                        "default": "Qwen/Qwen2.5-72B-Instruct",
                        "multiline": False,
                        "placeholder": "例如: Qwen/Qwen2.5-72B-Instruct",
                    },
                ),
                "user_prompt": (
                    "STRING",
                    {
                        "forceInput": True,
                    },
                ),
                "temperature": (
                    "FLOAT",
                    {
                        "default": 0.7,
                        "min": 0.0,
                        "max": 2.0,
                        "step": 0.1,
                        "display": "slider",
                    },
                ),
                "max_tokens": (
                    "INT",
                    {
                        "default": 1024,
                        "min": 1,
                        "max": 4096,
                        "step": 1,
                    },
                ),
            },
            "optional": {
                "system_prompt": (
                    "STRING",
                    {
                        "forceInput": True,
                    },
                ),
                "api_url": (
                    "STRING",
                    {
                        "default": "https://api.siliconflow.cn/v1/chat/completions",
                        "multiline": False,
                        "placeholder": "API端点URL",
                    },
                ),
                "top_p": (
                    "FLOAT",
                    {
                        "default": 0.9,
                        "min": 0.0,
                        "max": 1.0,
                        "step": 0.1,
                        "display": "slider",
                    },
                ),
                "top_k": (
                    "INT",
                    {
                        "default": 50,
                        "min": 1,
                        "max": 100,
                        "step": 1,
                    },
                ),
                "frequency_penalty": (
                    "FLOAT",
                    {
                        "default": 0.0,
                        "min": 0.0,
                        "max": 2.0,
                        "step": 0.1,
                        "display": "slider",
                    },
                ),
                "presence_penalty": (
                    "FLOAT",
                    {
                        "default": 0.0,
                        "min": 0.0,
                        "max": 2.0,
                        "step": 0.1,
                        "display": "slider",
                    },
                ),
            },
        }

    RETURN_TYPES = ("STRING", "STRING", "INT")
    RETURN_NAMES = ("response", "full_json", "tokens_used")
    FUNCTION = "call_llm"
    CATEGORY = "LLM"
    DESCRIPTION = "调用LLM API进行单轮对话"

    def call_llm(
        self,
        api_key: str,
        model: str,
        user_prompt: str,
        temperature: float,
        max_tokens: int,
        system_prompt: str = "",
        api_url: str = "https://api.siliconflow.cn/v1/chat/completions",
        top_p: float = 0.9,
        top_k: int = 50,
        frequency_penalty: float = 0.0,
        presence_penalty: float = 0.0,
    ) -> Tuple[str, str, int]:
        if not api_key.strip():
            return ("错误：请输入API密钥", json.dumps({"error": "API密钥为空"}), 0)

        if not user_prompt.strip():
            return ("错误：请输入用户提示", json.dumps({"error": "用户提示为空"}), 0)

        try:
            messages = []
            if system_prompt.strip():
                messages.append({"role": "system", "content": system_prompt.strip()})

            messages.append({"role": "user", "content": user_prompt.strip()})

            request_data = {
                "model": model,
                "messages": messages,
                "temperature": temperature,
                "max_tokens": max_tokens,
                "top_p": top_p,
                "top_k": top_k,
                "frequency_penalty": frequency_penalty,
                "presence_penalty": presence_penalty,
                "stream": False,
            }

            headers = {
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            }

            print(f"正在调用LLM API: {model}")
            print(f"用户提示: {user_prompt[:100]}{'...' if len(user_prompt) > 100 else ''}")

            start_time = time.time()
            response = requests.post(
                api_url,
                headers=headers,
                json=request_data,
                timeout=120,
            )

            response_time = time.time() - start_time

            if response.status_code != 200:
                error_msg = f"API调用失败 (状态码: {response.status_code}): {response.text}"
                print(error_msg)
                return (error_msg, json.dumps({"error": error_msg, "status_code": response.status_code}), 0)

            try:
                response_json = response.json()
            except json.JSONDecodeError as exc:
                error_msg = f"响应JSON解析失败: {str(exc)}"
                print(error_msg)
                return (error_msg, json.dumps({"error": error_msg}), 0)

            if "choices" in response_json and len(response_json["choices"]) > 0:
                llm_response = response_json["choices"][0]["message"]["content"]
            else:
                error_msg = "API响应中没有找到有效内容"
                print(error_msg)
                return (error_msg, json.dumps(response_json), 0)

            tokens_used = 0
            if "usage" in response_json:
                tokens_used = response_json["usage"].get("total_tokens", 0)

            full_response = json.dumps(response_json, indent=2, ensure_ascii=False)

            print("✅ API调用成功！")
            print(f"📊 响应时间: {response_time:.2f}秒")
            print(f"🔢 Token使用: {tokens_used}")
            print(f"📝 响应长度: {len(llm_response)}字符")

            return (llm_response, full_response, tokens_used)

        except requests.exceptions.Timeout:
            error_msg = "请求超时，请稍后重试"
            print(error_msg)
            return (error_msg, json.dumps({"error": error_msg}), 0)

        except requests.exceptions.ConnectionError:
            error_msg = "网络连接错误，请检查网络连接"
            print(error_msg)
            return (error_msg, json.dumps({"error": error_msg}), 0)

        except requests.exceptions.RequestException as exc:
            error_msg = f"请求错误: {str(exc)}"
            print(error_msg)
            return (error_msg, json.dumps({"error": error_msg}), 0)

        except Exception as exc:
            error_msg = f"未知错误: {str(exc)}"
            print(error_msg)
            return (error_msg, json.dumps({"error": error_msg}), 0)


__all__ = ["SimpleLLMNode"]
