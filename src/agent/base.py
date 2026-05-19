"""Base agent implementation for ARAG - src3 version."""

import json
import logging
from typing import Any, Dict, List, Optional

import tiktoken

from ..core.context import AgentContext
from ..core.llm import LLMClient
from ..tools.registry import ToolRegistry

logger = logging.getLogger(__name__)


class BaseAgent:
    """Base agent with tool calling capabilities."""
    
    def __init__(
        self,
        llm_client: LLMClient,
        tools: ToolRegistry,
        system_prompt: str = None,
        max_loops: int = 10,
        max_token_budget: int = 128000,
        verbose: bool = False,
    ):
        self.llm = llm_client
        self.tools = tools
        self.system_prompt = system_prompt
        self.max_loops = max_loops
        self.max_token_budget = max_token_budget
        self.verbose = verbose
        self.tokenizer = tiktoken.encoding_for_model("gpt-4o")
    
    def _calculate_message_tokens(self, messages: List[Dict[str, Any]]) -> int:
        total = len(self.tokenizer.encode(self.system_prompt))
        for msg in messages:
            content = msg.get("content", "")
            if content:
                total += len(self.tokenizer.encode(str(content)))
        return total

    def _force_final_answer(self, messages: List[Dict[str, Any]], context: AgentContext,
                           reason: str) -> tuple:
        """Force the model to give a final answer when limits are reached."""
        force_prompt = (
            "You have reached the limit. "
            "You MUST now provide a final answer based on the information you have gathered so far. "
            "Do NOT call any more tools. Synthesize the available information and respond directly."
        )
        
        messages.append({"role": "user", "content": force_prompt})
        
        answered = True
        try:
            response = self.llm.chat(messages=messages, tools=None, temperature=0.0)
            final_answer = response["message"].get("content", "")
            
            # if self.verbose:
            #     logger.info(f"Forced answer: {final_answer}")
        except Exception as e:
            # if self.verbose:
            #     logger.info(f"Error getting forced answer: {e}")
            final_answer = f"Error: {reason} and failed to generate final answer."
            answered = False
        
        return final_answer, answered
    
    def run(self, query: str) -> Dict[str, Any]:
        context = AgentContext()
        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": query},
        ]
        
        trajectory = []
        loop_count = 0
        tool_schemas = self.tools.get_all_schemas()
        
        # if self.verbose:
        #     logger.info(f"\n{'='*60}")
        #     logger.info(f"Question: {query}")
        #     logger.info(f"{'='*60}\n")
        
        for loop_idx in range(self.max_loops):
            loop_count = loop_idx + 1
            
            current_tokens = self._calculate_message_tokens(messages)
            if current_tokens > self.max_token_budget:
                # if self.verbose:
                #     logger.info(f"Token budget exceeded ({current_tokens} > {self.max_token_budget}), forcing answer...")
                
                final_answer, answered = self._force_final_answer(
                    messages, context, "Token budget exceeded"
                )
                
                return {
                    "answer": final_answer,
                    "trajectory": trajectory,
                    "messages": messages,
                    "loops": loop_count,
                    "token_budget_exceeded": True,
                    "answered": answered,
                    **context.get_summary()
                }
            
            # if self.verbose:
            #     logger.info(f"Loop {loop_count}/{self.max_loops} (Tokens: {current_tokens}/{self.max_token_budget})")
            
            try:
                response = self.llm.chat(messages=messages, tools=tool_schemas)
            except Exception as e:
                # if self.verbose:
                #     logger.info(f"LLM error: {e}")
                return {
                    "answer": f"Error: LLM调用失败 - {str(e)}",
                    "trajectory": trajectory,
                    "messages": messages,
                    "loops": loop_count,
                    "answered": False,
                    **context.get_summary()
                }
            
            message = response["message"]
            messages.append(message)
            
            # if self.verbose and message.get("content"):
            #     logger.info(f"Assistant: {message['content']}")
            
            tool_calls = message.get("tool_calls")
            if not tool_calls:
                # No tool calls - agent is done
                final_answer = message.get("content", "")
                return {
                    "answer": final_answer,
                    "trajectory": trajectory,
                    "messages": messages,
                    "loops": loop_count,
                    "answered": True,
                    **context.get_summary()
                }
            
            # Execute tool calls
            for tc in tool_calls:
                func_name = tc["function"]["name"]
                try:
                    func_args = json.loads(tc["function"]["arguments"])
                except json.JSONDecodeError:
                    func_args = {}
                
                # if self.verbose:
                #     logger.info(f"Tool: {func_name}")
                #     logger.info(f"  Args: {func_args}")
                
                try:
                    tool_result, tool_log = self.tools.execute(func_name, context, **func_args)
                except Exception as e:
                    tool_result = f"Error executing tool: {str(e)}"
                    tool_log = {"retrieved_tokens": 0, "error": str(e)}
                
                # if self.verbose:
                #     logger.info(f"  Result: {tool_result}")
                #     if tool_log.get("retrieved_tokens", 0) > 0:
                #         logger.info(f"  Tokens: {tool_log['retrieved_tokens']}")
                #     logger.info()
                
                messages.append({
                    "role": "tool",
                    "tool_call_id": tc["id"],
                    "content": tool_result,
                })
                
                # Record trajectory
                traj_entry = {
                    "loop": loop_count,
                    "tool_name": func_name,
                    "arguments": func_args,
                    "tool_result": tool_result,
                    **tool_log
                }
                trajectory.append(traj_entry)
        
        # Max loops reached - force final answer
        # if self.verbose:
        #     logger.info(f"Max loops reached ({self.max_loops}), forcing answer...")
        
        final_answer, answered = self._force_final_answer(
            messages, context, "Maximum loops exceeded"
        )

        return {
            "answer": final_answer,
            "trajectory": trajectory,
            "messages": messages,
            "loops": loop_count,
            "max_loops_exceeded": True,
            "answered": answered,
            **context.get_summary()
        }
