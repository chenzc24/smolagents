#!/usr/bin/env python
# coding=utf-8
# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import ast
import json
import os
import re
import shutil
from datetime import datetime
from pathlib import Path
from typing import Generator

from smolagents.agent_types import AgentAudio, AgentImage, AgentText
from smolagents.agents import MultiStepAgent, PlanningStep
from smolagents.memory import ActionStep, FinalAnswerStep
from smolagents.models import ChatMessageStreamDelta, MessageRole, agglomerate_stream_deltas
from smolagents.utils import _is_package_available


def get_step_footnote_content(step_log: ActionStep | PlanningStep, step_name: str) -> str:
    """Get a footnote string for a step log with duration and token information"""
    step_footnote = f"**{step_name}**"
    if step_log.token_usage is not None:
        step_footnote += f" | Input tokens: {step_log.token_usage.input_tokens:,} | Output tokens: {step_log.token_usage.output_tokens:,}"
    step_footnote += f" | Duration: {round(float(step_log.timing.duration), 2)}s" if step_log.timing.duration else ""
    step_footnote_content = f"""<span style="color: #bbbbc2; font-size: 12px;">{step_footnote}</span> """
    return step_footnote_content


def _clean_model_output(model_output: str) -> str:
    """
    Clean up model output by removing trailing tags and extra backticks.

    Args:
        model_output (`str`): Raw model output.

    Returns:
        `str`: Cleaned model output.
    """
    if not model_output:
        return ""
    model_output = model_output.strip()
    # Remove any trailing <end_code> and extra backticks, handling multiple possible formats
    model_output = re.sub(r"```\s*<end_code>", "```", model_output)  # handles ```<end_code>
    model_output = re.sub(r"<end_code>\s*```", "```", model_output)  # handles <end_code>```
    model_output = re.sub(r"```\s*\n\s*<end_code>", "```", model_output)  # handles ```\n<end_code>
    return model_output.strip()


_DICT_BLOCK_PATTERN = re.compile(r"\{[^{}]*\}", re.DOTALL)


def _format_code_content(content: str) -> str:
    """
    Format code content as Python code block if it's not already formatted.

    Args:
        content (`str`): Code content to format.

    Returns:
        `str`: Code content formatted as a Python code block.
    """
    content = content.strip()
    # Remove existing code blocks and end_code tags
    content = re.sub(r"```.*?\n", "", content)
    content = re.sub(r"\s*<end_code>\s*", "", content)
    content = content.strip()
    # Add Python code block formatting if not already present
    if not content.startswith("```python"):
        content = f"```python\n{content}\n```"
    return content


def _as_stream_payload(execution_log: str | None, full_log: str | None) -> dict[str, str]:
    return {
        "execution_log": (execution_log or "").strip(),
        "full_log": full_log or "",
    }


def _maybe_dict_from_string(payload) -> dict | None:
    if isinstance(payload, str):
        text = payload.strip()
        candidate = text
        if not (text.startswith("{") and text.endswith("}")):
            start = text.find("{")
            end = text.rfind("}")
            if start == -1 or end == -1 or end <= start:
                candidate = None
            else:
                candidate = text[start : end + 1]
        if candidate:
            try:
                return json.loads(candidate)
            except json.JSONDecodeError:
                try:
                    return ast.literal_eval(candidate)
                except (ValueError, SyntaxError):
                    return None
    if isinstance(payload, dict):
        return payload
    return None


def _extract_execution_text(block: str) -> str:
    """Extract execution_log text from mixed observation strings containing dict blocks."""
    if not block:
        return ""
    text = block.strip()
    if not text:
        return ""

    direct_dict = _maybe_dict_from_string(text)
    if direct_dict and "execution_log" in direct_dict:
        exec_text = str(direct_dict.get("execution_log") or "").strip()
        if exec_text:
            return exec_text

    pieces: list[str] = []
    saw_dict = False
    cursor = 0
    for match in _DICT_BLOCK_PATTERN.finditer(text):
        saw_dict = True
        prefix = text[cursor:match.start()].strip()
        data = _maybe_dict_from_string(match.group(0))
        if data and "execution_log" in data:
            exec_text = str(data.get("execution_log") or "").strip()
            if exec_text:
                if prefix:
                    normalized = prefix.rstrip()
                    suffix = ":" if normalized.endswith(":") else ""
                    normalized = normalized[:-1].rstrip() if suffix else normalized
                    if normalized:
                        pieces.append(f"{normalized}:{exec_text}")
                    else:
                        pieces.append(exec_text)
                else:
                    pieces.append(exec_text)
        cursor = match.end()

    if pieces:
        return "\n\n".join(pieces).strip()

    if saw_dict:
        stripped = _DICT_BLOCK_PATTERN.sub("", text)
        stripped = re.sub(r"^Execution logs:\s*", "", stripped, flags=re.IGNORECASE).strip()
        return stripped

    cleaned = re.sub(r"^Execution logs:\s*", "", text, flags=re.IGNORECASE).strip()
    return cleaned


def _extract_tool_logs(action_output) -> tuple[str, str]:
    maybe_dict = _maybe_dict_from_string(action_output)
    if maybe_dict is not None and {"execution_log", "full_log"}.issubset(maybe_dict.keys()):
        exec_log = maybe_dict.get("execution_log") or ""
        full_log = maybe_dict.get("full_log") or ""
        return str(exec_log), str(full_log)
    if isinstance(action_output, dict) and {"execution_log", "full_log"}.issubset(action_output.keys()):
        exec_log = action_output.get("execution_log") or ""
        full_log = action_output.get("full_log") or ""
        return str(exec_log), str(full_log)
    if action_output is None:
        return "", ""
    return str(action_output), ""


def _render_final_answer_logs(step_log: FinalAnswerStep) -> tuple[str, str]:
    final_answer = step_log.output
    if isinstance(final_answer, AgentText):
        text = final_answer.to_string()
    elif isinstance(final_answer, (AgentImage, AgentAudio)):
        text = final_answer.to_string()
    else:
        text = str(final_answer)
    formatted = f"**Final answer:**\n{text}" if text else "Final answer provided."
    return formatted, formatted


def _step_to_full_markdown(step_log: ActionStep | PlanningStep | FinalAnswerStep) -> str:
    parts: list[str] = []

    if isinstance(step_log, ActionStep):
        step_number = f"Step {step_log.step_number}"
        parts.append(f"**{step_number}**")

        model_output = getattr(step_log, "model_output", "")
        if model_output:
            parts.append(_clean_model_output(model_output))

        tool_calls = getattr(step_log, "tool_calls", []) or []
        if tool_calls:
            first_tool_call = tool_calls[0]
            args = first_tool_call.arguments
            if isinstance(args, dict):
                content = str(args.get("answer", args))
            else:
                content = str(args).strip()

            if first_tool_call.name == "python_interpreter":
                content = _format_code_content(content)

            parts.append(f"**🛠️ Used tool `{first_tool_call.name}`**\n\n{content}")

        observations = getattr(step_log, "observations", "")
        if observations and observations.strip():
            log_content = re.sub(r"^Execution logs:\s*", "", observations.strip())
            parts.append(f"```bash\n{log_content}\n```")

        images = getattr(step_log, "observations_images", []) or []
        for image in images:
            path_image = AgentImage(image).to_string()
            parts.append(f"![Output image]({path_image})")

        if getattr(step_log, "error", None):
            parts.append(f"**Error:** {step_log.error}")

        parts.append(get_step_footnote_content(step_log, step_number))
        parts.append("-----")
        return "\n\n".join(part for part in parts if part).strip()

    if isinstance(step_log, PlanningStep):
        parts.append("**Planning step**")
        if step_log.plan:
            parts.append(step_log.plan)
        parts.append(get_step_footnote_content(step_log, "Planning step"))
        parts.append("-----")
        return "\n\n".join(part for part in parts if part).strip()

    if isinstance(step_log, FinalAnswerStep):
        _, full_log = _render_final_answer_logs(step_log)
        return full_log

    return ""


def _step_to_stream_payload(step_log: ActionStep | PlanningStep | FinalAnswerStep) -> dict[str, str]:
    full_markdown = _step_to_full_markdown(step_log)
    if isinstance(step_log, ActionStep):
        exec_log, tool_full = _extract_tool_logs(step_log.action_output)
        full_log_parts: list[str] = []
        if tool_full:
            full_log_parts.append(tool_full)
        if step_log.observations:
            full_log_parts.append(step_log.observations)
        if step_log.error:
            full_log_parts.append(f"Error: {step_log.error}")
        if not exec_log and step_log.observations:
            observations = step_log.observations.strip()
            execution_text = _extract_execution_text(observations)
            if execution_text:
                exec_log = execution_text
        if not exec_log and step_log.error:
            exec_log = f"Error: {step_log.error}"
        if not exec_log:
            exec_log = "Action step completed."
        full_log = "\n\n".join(part for part in full_log_parts if part)
        if not full_log:
            full_log = exec_log
        return _as_stream_payload(exec_log, full_markdown or full_log)
    if isinstance(step_log, PlanningStep):
        return _as_stream_payload("", full_markdown or step_log.plan)
    if isinstance(step_log, FinalAnswerStep):
        _, full_log = _render_final_answer_logs(step_log)
        return _as_stream_payload("", full_markdown or full_log)
    return _as_stream_payload("", "")


def _process_action_step(step_log: ActionStep, skip_model_outputs: bool = False) -> Generator:
    """
    Process an [`ActionStep`] and yield appropriate Gradio ChatMessage objects.

    Args:
        step_log ([`ActionStep`]): ActionStep to process.
        skip_model_outputs (`bool`): Whether to skip model outputs.

    Yields:
        `gradio.ChatMessage`: Gradio ChatMessages representing the action step.
    """
    import gradio as gr

    # Output the step number
    step_number = f"Step {step_log.step_number}"
    if not skip_model_outputs:
        yield gr.ChatMessage(role=MessageRole.ASSISTANT, content=f"**{step_number}**", metadata={"status": "done"})

    # First yield the thought/reasoning from the LLM
    if not skip_model_outputs and getattr(step_log, "model_output", ""):
        model_output = _clean_model_output(step_log.model_output)
        yield gr.ChatMessage(role=MessageRole.ASSISTANT, content=model_output, metadata={"status": "done"})

    # For tool calls, create a parent message
    if getattr(step_log, "tool_calls", []):
        first_tool_call = step_log.tool_calls[0]
        used_code = first_tool_call.name == "python_interpreter"

        # Process arguments based on type
        args = first_tool_call.arguments
        if isinstance(args, dict):
            content = str(args.get("answer", str(args)))
        else:
            content = str(args).strip()

        # Format code content if needed
        if used_code:
            content = _format_code_content(content)

        # Create the tool call message
        parent_message_tool = gr.ChatMessage(
            role=MessageRole.ASSISTANT,
            content=content,
            metadata={
                "title": f"🛠️ Used tool {first_tool_call.name}",
                "status": "done",
            },
        )
        yield parent_message_tool

    # Display execution logs if they exist
    if getattr(step_log, "observations", "") and step_log.observations.strip():
        log_content = _extract_execution_text(step_log.observations)
        if log_content:
            yield gr.ChatMessage(
                role=MessageRole.ASSISTANT,
                content=f"```bash\n{log_content}\n```",
                metadata={"title": "📝 Execution Logs", "status": "done"},
            )

    # Display any images in observations
    if getattr(step_log, "observations_images", []):
        for image in step_log.observations_images:
            path_image = AgentImage(image).to_string()
            yield gr.ChatMessage(
                role=MessageRole.ASSISTANT,
                content=(path_image, None),
                metadata={"title": "🖼️ Output Image", "status": "done"},
            )

    # Handle errors
    if getattr(step_log, "error", None):
        yield gr.ChatMessage(
            role=MessageRole.ASSISTANT, content=str(step_log.error), metadata={"title": "💥 Error", "status": "done"}
        )

    # Add step footnote and separator
    yield gr.ChatMessage(
        role=MessageRole.ASSISTANT,
        content=get_step_footnote_content(step_log, step_number),
        metadata={"status": "done"},
    )
    yield gr.ChatMessage(role=MessageRole.ASSISTANT, content="-----", metadata={"status": "done"})


def _process_planning_step(step_log: PlanningStep, skip_model_outputs: bool = False) -> Generator:
    """
    Process a [`PlanningStep`] and yield appropriate gradio.ChatMessage objects.

    Args:
        step_log ([`PlanningStep`]): PlanningStep to process.

    Yields:
        `gradio.ChatMessage`: Gradio ChatMessages representing the planning step.
    """
    import gradio as gr

    if not skip_model_outputs:
        yield gr.ChatMessage(role=MessageRole.ASSISTANT, content="**Planning step**", metadata={"status": "done"})
        yield gr.ChatMessage(role=MessageRole.ASSISTANT, content=step_log.plan, metadata={"status": "done"})
    yield gr.ChatMessage(
        role=MessageRole.ASSISTANT,
        content=get_step_footnote_content(step_log, "Planning step"),
        metadata={"status": "done"},
    )
    yield gr.ChatMessage(role=MessageRole.ASSISTANT, content="-----", metadata={"status": "done"})


def _process_final_answer_step(step_log: FinalAnswerStep) -> Generator:
    """
    Process a [`FinalAnswerStep`] and yield appropriate gradio.ChatMessage objects.

    Args:
        step_log ([`FinalAnswerStep`]): FinalAnswerStep to process.

    Yields:
        `gradio.ChatMessage`: Gradio ChatMessages representing the final answer.
    """
    import gradio as gr

    final_answer = step_log.output
    if isinstance(final_answer, AgentText):
        yield gr.ChatMessage(
            role=MessageRole.ASSISTANT,
            content=f"**Final answer:**\n{final_answer.to_string()}\n",
            metadata={"status": "done"},
        )
    elif isinstance(final_answer, AgentImage):
        yield gr.ChatMessage(
            role=MessageRole.ASSISTANT,
            content=(final_answer.to_string(), None),
            metadata={"status": "done"},
        )
    elif isinstance(final_answer, AgentAudio):
        yield gr.ChatMessage(
            role=MessageRole.ASSISTANT,
            content=(final_answer.to_string(), None),
            metadata={"status": "done"},
        )
    else:
        yield gr.ChatMessage(
            role=MessageRole.ASSISTANT, content=f"**Final answer:** {str(final_answer)}", metadata={"status": "done"}
        )


def pull_messages_from_step(step_log: ActionStep | PlanningStep | FinalAnswerStep, skip_model_outputs: bool = False):
    """Extract Gradio ChatMessage objects from agent steps with proper nesting.

    Args:
        step_log: The step log to display as gr.ChatMessage objects.
        skip_model_outputs: If True, skip the model outputs when creating the gr.ChatMessage objects:
            This is used for instance when streaming model outputs have already been displayed.
    """
    if not _is_package_available("gradio"):
        raise ModuleNotFoundError(
            "Please install 'gradio' extra to use the GradioUI: `pip install 'smolagents[gradio]'`"
        )
    if isinstance(step_log, ActionStep):
        yield from _process_action_step(step_log, skip_model_outputs)
    elif isinstance(step_log, PlanningStep):
        yield from _process_planning_step(step_log, skip_model_outputs)
    elif isinstance(step_log, FinalAnswerStep):
        yield from _process_final_answer_step(step_log)
    else:
        raise ValueError(f"Unsupported step type: {type(step_log)}")


def stream_to_gradio(
    agent,
    task: str,
    task_images: list | None = None,
    reset_agent_memory: bool = False,
    additional_args: dict | None = None,
) -> Generator:
    """Runs an agent with the given task and streams the messages from the agent as gradio ChatMessages."""

    if not _is_package_available("gradio"):
        raise ModuleNotFoundError(
            "Please install 'gradio' extra to use the GradioUI: `pip install 'smolagents[gradio]'`"
        )
    accumulated_events: list[ChatMessageStreamDelta] = []
    last_stream_text = ""
    for event in agent.run(
        task, images=task_images, stream=True, reset=reset_agent_memory, additional_args=additional_args
    ):
        if isinstance(event, (ActionStep, PlanningStep, FinalAnswerStep)):
            yield _step_to_stream_payload(event)
            accumulated_events = []
            last_stream_text = ""
        elif isinstance(event, ChatMessageStreamDelta):
            accumulated_events.append(event)
            text = agglomerate_stream_deltas(accumulated_events).render_as_markdown()
            if not text:
                continue
            if last_stream_text and text.startswith(last_stream_text):
                new_piece = text[len(last_stream_text) :]
            else:
                new_piece = text
            if not new_piece:
                continue
            last_stream_text = text
            yield _as_stream_payload("", new_piece)


class GradioUI:
    """
    Gradio interface for interacting with a [`MultiStepAgent`].

    This class provides a web interface to interact with the agent in real-time, allowing users to submit prompts, upload files, and receive responses in a chat-like format.
    It  can reset the agent's memory at the start of each interaction if desired.
    It supports file uploads, which are saved to a specified folder.
    It uses the [`gradio.Chatbot`] component to display the conversation history.
    This class requires the `gradio` extra to be installed: `pip install 'smolagents[gradio]'`.

    Args:
        agent ([`MultiStepAgent`]): The agent to interact with.
        file_upload_folder (`str`, *optional*): The folder where uploaded files will be saved.
            If not provided, inline uploads are disabled.
        reset_agent_memory (`bool`, *optional*, defaults to `False`): Whether to reset the agent's memory at the start of each interaction.
            If `True`, the agent will not remember previous interactions.
        allowed_file_types (`list[str]`, *optional*): Allowed file extensions for inline uploads. Defaults to `[".pdf", ".docx", ".txt"]`.
        output_base_folders (`list[str]`, *optional*): Local folders that should be initialized alongside the upload folder. Useful for surfacing generated outputs.

    Raises:
        ModuleNotFoundError: If the `gradio` extra is not installed.

    Example:
        ```python
        from smolagents import CodeAgent, GradioUI, InferenceClientModel

        model = InferenceClientModel(model_id="meta-llama/Meta-Llama-3.1-8B-Instruct")
        agent = CodeAgent(tools=[], model=model)
        gradio_ui = GradioUI(agent, file_upload_folder="uploads", reset_agent_memory=True)
        gradio_ui.launch()
        ```
    """

    _IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".gif", ".bmp", ".tiff"}
    _TEXT_EXTENSIONS = {".txt", ".md", ".log", ".csv"}
    _JSON_EXTENSIONS = {".json"}
    _PDF_EXTENSIONS = {".pdf"}

    def __init__(
        self,
        agent: MultiStepAgent,
        file_upload_folder: str | None = None,
        reset_agent_memory: bool = False,
        allowed_file_types: list[str] | None = None,
        output_base_folders: list[str] | None = None,
    ):
        if not _is_package_available("gradio"):
            raise ModuleNotFoundError(
                "Please install 'gradio' extra to use the GradioUI: `pip install 'smolagents[gradio]'`"
            )
        self.agent = agent
        self.file_upload_folder = Path(file_upload_folder) if file_upload_folder is not None else None
        self.reset_agent_memory = reset_agent_memory
        self.name = getattr(agent, "name") or "Agent interface"
        self.description = getattr(agent, "description", None)
        self.allowed_file_types = (
            [ft.lower() if ft.startswith(".") else f".{ft.lower().lstrip('.')}" for ft in allowed_file_types]
            if allowed_file_types
            else [".pdf", ".docx", ".txt"]
        )
        base_folders = output_base_folders if output_base_folders else ["output"]
        self.output_base_folders = [Path(folder) for folder in base_folders]
        self._chatbot_component = None

        if self.file_upload_folder is not None:
            if not self.file_upload_folder.exists():
                self.file_upload_folder.mkdir(parents=True, exist_ok=True)
        for folder in self.output_base_folders:
            folder.mkdir(parents=True, exist_ok=True)

    def _sanitize_filename(self, filename: str) -> str:
        sanitized_name = re.sub(r"[^\w\-.]", "_", os.path.basename(filename))
        return sanitized_name or "uploaded_file"

    def _extension_is_allowed(self, file_path: Path) -> bool:
        if not self.allowed_file_types:
            return True
        return file_path.suffix.lower() in self.allowed_file_types

    def _resolve_file_path(self, file_like) -> Path | None:
        if file_like is None:
            return None
        candidate = None
        if isinstance(file_like, (str, Path)):
            candidate = Path(file_like)
        elif isinstance(file_like, dict):
            for key in ("path", "name", "file"):
                value = file_like.get(key)
                if value:
                    candidate = Path(value)
                    break
        elif hasattr(file_like, "name") and getattr(file_like, "name"):
            candidate = Path(file_like.name)

        if candidate is None:
            return None
        return candidate if candidate.exists() else None

    def _copy_into_uploads(self, source_path: Path, session_folder_name: str | None = None) -> str:
        if self.file_upload_folder is None:
            raise ValueError("File uploads are disabled because no upload folder was provided.")
        sanitized_name = self._sanitize_filename(source_path.name)

        if session_folder_name is None:
            session_folder_name = datetime.now().strftime("%Y%m%d_%H%M%S")

        target_dir = self.file_upload_folder / session_folder_name
        if not target_dir.exists():
            target_dir.mkdir(parents=True, exist_ok=True)

        destination = target_dir / sanitized_name
        counter = 1
        while destination.exists():
            destination = target_dir / f"{destination.stem}_{counter}{destination.suffix}"
            counter += 1
        shutil.copy(source_path, destination)
        return str(destination)

    def _chatbot_image_content(self, file_path: str):
        """Return an image tuple pointing to a backend-served path (keeps original preview)."""
        served_path = file_path
        try:
            from gradio import processing_utils
        except ModuleNotFoundError:
            return (served_path, None)

        chatbot_component = getattr(self, "_chatbot_component", None)
        if chatbot_component is None:
            return (served_path, None)

        data = {"path": file_path, "meta": {"_type": "gradio.FileData"}}
        try:
            cached_data = processing_utils.move_files_to_cache(data, chatbot_component)
            served_path = cached_data.get("path") or served_path
        except Exception:
            pass
        return (served_path, None)

    def _extract_prompt_and_files(self, prompt_payload) -> tuple[str, list[str], list[str]]:
        if prompt_payload is None:
            return "", [], []

        session_folder = datetime.now().strftime("%Y%m%d_%H%M%S")

        def _store_upload(resolved_path: Path, saved_list: list[str], source_list: list[str]):
            saved_path = self._copy_into_uploads(resolved_path, session_folder_name=session_folder)
            saved_list.append(saved_path)
            source_list.append(str(resolved_path))

        if isinstance(prompt_payload, list):
            text_segments: list[str] = []
            saved_files: list[str] = []
            source_files: list[str] = []
            for item in prompt_payload:
                if not isinstance(item, dict):
                    continue
                if item.get("type") == "text" and item.get("text"):
                    text_segments.append(item["text"].strip())
                elif item.get("type") in {"file", "image", "audio", "video"}:
                    file_payload = (
                        item.get(item["type"])  # structured payload like {"path": ...}
                        or item.get("file")
                        or item.get("path")
                    )
                    resolved_path = self._resolve_file_path(file_payload)
                    if resolved_path is None:
                        continue
                    if not self._extension_is_allowed(resolved_path):
                        raise ValueError(f"File type {resolved_path.suffix} is not allowed.")
                    _store_upload(resolved_path, saved_files, source_files)
            return "\n".join(filter(None, text_segments)).strip(), saved_files, source_files

        if isinstance(prompt_payload, dict):
            text_value = str(prompt_payload.get("text", ""))
            saved_files: list[str] = []
            source_files: list[str] = []

            def _iter_payload_files(payload_dict):
                for key in ("files", "images", "audios", "videos"):
                    entries = payload_dict.get(key)
                    if not entries:
                        continue
                    if not isinstance(entries, list):
                        entries = [entries]
                    for entry in entries:
                        yield entry

            for entry in _iter_payload_files(prompt_payload):
                resolved_path = self._resolve_file_path(entry)
                if resolved_path is None:
                    continue
                if not self._extension_is_allowed(resolved_path):
                    raise ValueError(f"File type {resolved_path.suffix} is not allowed.")
                _store_upload(resolved_path, saved_files, source_files)

            return text_value.strip(), saved_files, source_files

        if isinstance(prompt_payload, str):
            return prompt_payload.strip(), [], []

        resolved_path = self._resolve_file_path(prompt_payload)
        if resolved_path is not None:
            if not self._extension_is_allowed(resolved_path):
                raise ValueError(f"File type {resolved_path.suffix} is not allowed.")
            saved_files: list[str] = []
            source_files: list[str] = []
            _store_upload(resolved_path, saved_files, source_files)
            return "", saved_files, source_files

        return str(prompt_payload).strip(), [], []

    def _detect_output_type(self, file_path: Path) -> str:
        suffix = file_path.suffix.lower()
        if suffix in self._IMAGE_EXTENSIONS:
            return "image"
        if suffix in self._JSON_EXTENSIONS:
            return "json"
        if suffix in self._TEXT_EXTENSIONS:
            return "text"
        if suffix in self._PDF_EXTENSIONS:
            return "pdf"
        return "other"

    def _read_text_preview(self, file_path: Path, limit: int = 2000) -> str:
        try:
            with file_path.open("r", encoding="utf-8", errors="ignore") as handle:
                snippet = handle.read(limit)
            if file_path.stat().st_size > limit:
                snippet = snippet.rstrip() + "\n..."
            return snippet.strip()
        except Exception as exc:  # pragma: no cover - best effort preview only
            return f"无法读取文本文件：{exc}"

    def _read_json_preview(self, file_path: Path):
        try:
            with file_path.open("r", encoding="utf-8", errors="ignore") as handle:
                data = json.load(handle)
            return data
        except Exception as exc:  # pragma: no cover - best effort preview only
            return {"error": f"无法解析 JSON：{exc}", "path": str(file_path)}

    def load_latest_output_files(self, min_timestamp: float = 0) -> list[dict] | None:
        candidate_dirs: list[Path] = []
        for base in self.output_base_folders:
            if not base.exists():
                continue
            for child in base.iterdir():
                if child.is_dir():
                    if child.name.startswith(".") or child.name == "drc" or child.name == "lvs":
                        continue
                    candidate_dirs.append(child)

        if not candidate_dirs:
            return None

        # Filter directories by timestamp (only show those created/modified after session start)
        valid_dirs = [d for d in candidate_dirs if d.stat().st_mtime >= min_timestamp]
        
        if not valid_dirs:
            return None

        # Sort by time descending (newest first)
        valid_dirs.sort(key=lambda folder: folder.stat().st_mtime, reverse=True)
        
        results = []
        for directory in valid_dirs:
            file_entries = []
            for child in sorted(directory.iterdir()):
                if not child.is_file():
                    continue
                if child.name.startswith("."):
                    continue
                file_type = self._detect_output_type(child)
                entry: dict = {
                    "path": str(child),
                    "name": child.name,
                    "type": file_type,
                }
                if file_type == "text":
                    entry["content"] = self._read_text_preview(child)
                elif file_type == "json":
                    entry["json_content"] = self._read_json_preview(child)
                file_entries.append(entry)
            
            if file_entries:
                results.append({"folder": str(directory), "folder_name": directory.name, "files": file_entries})

        return results if results else None



    def interact_with_agent(self, prompt, messages, session_state):
        import gradio as gr

        if "agent" not in session_state:
            session_state["agent"] = self.agent

        attached_files = session_state.get("latest_files", []) or []
        # attached_temp_files = session_state.get("latest_temp_files", []) or []

        # Separate images for native vision support
        task_images = []
        final_prompt = prompt

        # Check if any attached files are images
        if attached_files:
            for file_path in attached_files:
                path_obj = Path(file_path)
                if path_obj.suffix.lower() in self._IMAGE_EXTENSIONS:
                    task_images.append(file_path)
            
            # If we have images, we might want to adjust the prompt or rely on the agent's vision capabilities
            # The prompt already contains <uploaded_files> block from log_user_message

        exec_index: int | None = None
        try:
            # Construct user message with previews
            # Use separate messages for text and images to ensure compatibility
            display_text = prompt or ""
            if attached_files:
                file_paths_text = "\n_Uploaded files:_\n" + "\n".join(attached_files)
                display_text = (display_text + "\n" + file_paths_text).strip()
            
            if not display_text:
                display_text = "(files uploaded; no prompt text)"

            messages.append(gr.ChatMessage(role="user", content=display_text, metadata={"status": "done"}))

            # Add separate chat messages for uploaded images to show previews
            if attached_files:
                for file_path in attached_files:
                    path_obj = Path(file_path)
                    if path_obj.suffix.lower() in self._IMAGE_EXTENSIONS:
                        messages.append(gr.ChatMessage(
                            role="user",
                            content=self._chatbot_image_content(file_path),
                            metadata={"status": "done", "title": "🖼️ Uploaded Image"}
                        ))

            session_state.setdefault("full_buffer", "")
            session_state["full_buffer"] = ""
            session_state.setdefault("exec_buffer", "")
            session_state["exec_buffer"] = ""
            session_state["step"] = 1

            exec_msg = gr.ChatMessage(role="assistant", content="", metadata={"type": "exec", "status": "pending"})
            messages.append(exec_msg)
            exec_index = len(messages) - 1

            def _full_reason_update():
                full_buffer = session_state.get("full_buffer", "")
                content = full_buffer or "_No reasoning yet._"
                return gr.update(value=content)

            yield messages, _full_reason_update()

            # Determine if we should pass images natively based on model config
            # If flatten_messages_as_text is True, the model expects text-only, so we don't pass images natively.
            # The agent can still access images via the file paths injected in the prompt.
            agent_model = getattr(session_state["agent"], "model", None)
            flatten_messages = getattr(agent_model, "flatten_messages_as_text", True)
            should_pass_images = (not flatten_messages) and bool(task_images)

            # Pass task_images to stream_to_gradio
            for msg in stream_to_gradio(
                session_state["agent"], 
                task=final_prompt, 
                task_images=task_images if should_pass_images else None,
                reset_agent_memory=self.reset_agent_memory
            ):
                if isinstance(msg, dict) and {"execution_log", "full_log"}.issubset(msg.keys()):
                    exec_piece = msg.get("execution_log") or ""
                    full_piece = msg.get("full_log") or ""
                    if exec_piece:
                        step = session_state.get("step", 1)
                        append = f"### Step {step}\n{exec_piece}\n\n"
                        session_state["exec_buffer"] += append
                        messages[exec_index].content = session_state["exec_buffer"].rstrip()
                        session_state["step"] = step + 1
                    if full_piece:
                        session_state["full_buffer"] += full_piece
                    yield messages, _full_reason_update()
                    continue

                if isinstance(msg, gr.ChatMessage):
                    messages.append(msg)
                    yield messages, _full_reason_update()
                elif isinstance(msg, str):
                    session_state["full_buffer"] += msg
                    yield messages, _full_reason_update()

            if exec_index is not None:
                messages[exec_index].metadata["status"] = "done"
            yield messages, _full_reason_update()
        except Exception as e:
            if '_full_reason_update' in locals():
                yield messages, _full_reason_update()
            else:
                yield messages, gr.update()
            raise gr.Error(f"Error in interaction: {str(e)}")
        finally:
            session_state["latest_files"] = []
            session_state["latest_temp_files"] = []
            if exec_index is not None and exec_index < len(messages):
                messages[exec_index].metadata["status"] = "done"
            session_state["exec_buffer"] = ""
            session_state["full_buffer"] = ""
            session_state["step"] = 1

    def upload_file(self, file, file_uploads_log, allowed_file_types=None):
        """Upload a file triggered by a classic gr.File component (legacy helper)."""
        import gradio as gr

        if file is None:
            return gr.Textbox(value="No file uploaded", visible=True), file_uploads_log

        resolved_path = self._resolve_file_path(file)
        if resolved_path is None:
            return gr.Textbox(value="No file uploaded", visible=True), file_uploads_log

        allowed_types = (
            [ft.lower() if ft.startswith(".") else f".{ft.lower().lstrip('.')}" for ft in allowed_file_types]
            if allowed_file_types
            else self.allowed_file_types
        )

        if allowed_types and resolved_path.suffix.lower() not in allowed_types:
            return gr.Textbox("File type disallowed", visible=True), file_uploads_log

        try:
            saved_path = self._copy_into_uploads(resolved_path)
        except ValueError as exc:  # Raised when no upload folder is configured
            raise gr.Error(str(exc)) from exc

        return gr.Textbox(f"File uploaded: {saved_path}", visible=True), file_uploads_log + [saved_path]

    def log_user_message(self, prompt_payload, file_uploads_log, session_state):
        import gradio as gr

        try:
            prompt_text, saved_files, temp_files = self._extract_prompt_and_files(prompt_payload)
        except ValueError as exc:
            raise gr.Error(str(exc)) from exc

        if not prompt_text and not saved_files and not temp_files:
            raise gr.Error("Please enter a prompt or attach a supported file.")

        session_state.setdefault("latest_files", [])
        session_state["latest_files"] = saved_files
        session_state["latest_temp_files"] = temp_files

        if temp_files or saved_files:
            attachment_sections: list[str] = []
            # Only inject uploads paths for agent usage, ignoring temp paths to reduce confusion
            if saved_files:
                saved_lines = "\n".join(f"{path}" for path in saved_files)
                attachment_sections.append("<uploaded_files>\n" + saved_lines + "\n</uploaded_files>")
            
            attachments_note = "\n\n".join(attachment_sections)
            prompt_text = f"{prompt_text}\n\n{attachments_note}" if prompt_text else attachments_note

        updated_log = (file_uploads_log or []) + saved_files

        return (
            prompt_text,
            gr.update(value=None, interactive=False),
            updated_log,
        )

    def launch(self, share: bool = True, **kwargs):
        """
        Launch the Gradio app with the agent interface.

        Args:
            share (`bool`, defaults to `True`): Whether to share the app publicly.
            **kwargs: Additional keyword arguments to pass to the Gradio launch method.
        """
        self.create_app().launch(debug=True, share=share, **kwargs)

    def create_app(self):
        import gradio as gr

        def init_session():
            return {"start_time": datetime.now().timestamp()}

        with gr.Blocks(theme="ocean", fill_height=True) as demo:
            # Add session state to store session-specific data
            session_state = gr.State(init_session)
            stored_messages = gr.State("")
            file_uploads_log = gr.State([])
            output_refresh_state = gr.State(0)

            with gr.Sidebar():
                gr.Markdown(
                    f"# {self.name.replace('_', ' ').capitalize()}"
                    "\n> This web ui allows you to interact with a `smolagents` agent that can use tools and execute steps to complete tasks."
                    + (f"\n\n**Agent description:**\n{self.description}" if self.description else "")
                )
                with gr.Accordion("输出文件", open=False):
                    @gr.render(inputs=[session_state, output_refresh_state])
                    def render_output_files(state, _):
                        start_time = state.get("start_time", 0) if state else 0
                        payloads = self.load_latest_output_files(min_timestamp=start_time)
                        
                        if not payloads:
                            gr.Markdown("`output/` 中暂无可供下载的文件（当前会话）。")
                        else:
                            for entry in payloads:
                                folder_name = entry.get("folder_name", "Unknown Folder")
                                files = entry.get("files", [])
                                file_paths = [f["path"] for f in files]
                                if file_paths:
                                    with gr.Accordion(folder_name, open=False):
                                        gr.File(
                                            value=file_paths,
                                            file_count="multiple",
                                            interactive=False,
                                            label=folder_name,
                                            elem_classes=["file-output-component"]
                                        )

                    refresh_outputs = gr.Button("刷新输出文件", variant="secondary")

            with gr.Column(scale=1, elem_classes=["agent-column"]):
                chatbot = gr.Chatbot(
                    label="Agent",
                    type="messages",
                    avatar_images=(
                        None,
                        "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/smolagents/mascot_smol.png",
                    ),
                    resizeable=False,
                    scale=1,
                    latex_delimiters=[
                        {"left": r"$$", "right": r"$$", "display": True},
                        {"left": r"$", "right": r"$", "display": False},
                        {"left": r"\[", "right": r"\]", "display": True},
                        {"left": r"\(", "right": r"\)", "display": False},
                    ],
                )
                self._chatbot_component = chatbot

                with gr.Accordion("🔎 Full reasoning", open=False, elem_classes=["full-reasoning-accordion"]):
                    full_reasoning_md = gr.Markdown("_No reasoning yet._", elem_classes=["full-reasoning-md"])

                supports_inline_uploads = self.file_upload_folder is not None and hasattr(gr, "MultimodalTextbox")
                placeholder = "Enter a prompt. Paste or drop files to save them locally (not shared with the agent)."

                with gr.Group(elem_classes=["input-container"]):
                    if supports_inline_uploads:
                        text_input = gr.MultimodalTextbox(
                            label="Chat Message",
                            show_label=False,
                            placeholder=placeholder,
                            file_types=self.allowed_file_types or None,
                            file_count="multiple",
                        )
                    else:
                        text_input = gr.Textbox(
                            lines=3,
                            label="Chat Message",
                            show_label=False,
                            placeholder=placeholder,
                        )
                    
                    stop_btn = gr.Button("■", elem_classes=["stop-btn"], visible=False)

            gr.HTML(
                """<style>
.full-reasoning-accordion {
    width: 100%;
}
.full-reasoning-md {
    width: 100%;
    max-height: 40vh;
    overflow-y: auto;
    padding-right: 0.5rem;
    word-break: break-word;
    overflow-wrap: anywhere;
    white-space: normal;
}
.full-reasoning-md *:not(pre):not(code) {
    word-break: break-word;
    overflow-wrap: anywhere;
    white-space: normal;
}
.full-reasoning-md pre,
.full-reasoning-md code {
    white-space: pre-wrap;
    word-break: break-word;
    overflow-wrap: anywhere;
}
.input-container {
    position: relative;
}
.stop-btn {
    position: absolute !important;
    right: 10px !important;
    bottom: 10px !important;
    width: 36px !important;
    height: 36px !important;
    min-width: unset !important;
    border-radius: 50% !important;
    background: #333 !important;
    color: white !important;
    border: none !important;
    box-shadow: none !important;
    z-index: 1000 !important;
    padding: 0 !important;
    display: flex !important;
    align-items: center !important;
    justify-content: center !important;
    font-size: 16px !important;
}
.stop-btn:hover {
    background: #555 !important;
}
.file-output-component {
    overflow-x: auto !important;
    scrollbar-width: thin;
}
.file-output-component table {
    width: max-content !important;
    min-width: 100% !important;
    table-layout: auto !important;
}
.file-output-component td,
.file-output-component .file-name,
.file-output-component .file-preview,
.file-output-component span,
.file-output-component a {
    white-space: nowrap !important;
    word-break: keep-all !important;
    overflow-wrap: normal !important;
    text-overflow: clip !important;
    overflow: visible !important;
    max-width: none !important;
}
</style>"""
            )

            def refresh_trigger(count):
                return count + 1

            refresh_outputs.click(refresh_trigger, [output_refresh_state], [output_refresh_state])

            # Helper to reset UI state
            def reset_ui_state():
                return gr.update(visible=False), gr.update(interactive=True)

            # Helper to show stop button
            def show_stop_button():
                return gr.update(visible=True)

            # Chain of events
            # 1. Log message & clear input
            submission = text_input.submit(
                self.log_user_message,
                [text_input, file_uploads_log, session_state],
                [stored_messages, text_input, file_uploads_log],
            )
            
            # 2. Show stop button (fast)
            submission = submission.then(
                show_stop_button, None, stop_btn
            )
            
            # 3. Run agent (slow, cancellable)
            agent_interaction = submission.then(
                self.interact_with_agent,
                [stored_messages, chatbot, session_state],
                [chatbot, full_reasoning_md],
            )
            
            # 4. After agent finishes (normally)
            agent_interaction.then(
                refresh_trigger,
                [output_refresh_state],
                [output_refresh_state],
            ).then(
                reset_ui_state,
                None,
                [stop_btn, text_input],
            )
            
            # 5. Stop button clicked
            stop_btn.click(
                None, None, None, cancels=[agent_interaction]
            ).then(
                reset_ui_state,
                None,
                [stop_btn, text_input],
            )

            chatbot.clear(self.agent.memory.reset)
        return demo


__all__ = ["stream_to_gradio", "GradioUI"]