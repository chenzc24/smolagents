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
        exec_log, full_log = _render_final_answer_logs(step_log)
        return _as_stream_payload(exec_log, full_markdown or full_log)
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
                content=f"```bash\n{log_content}\n",
                metadata={"title": "📝 Execution Logs", "status": "done"},
            )

    # Display any images in observations
    if getattr(step_log, "observations_images", []):
        for image in step_log.observations_images:
            path_image = AgentImage(image).to_string()
            yield gr.ChatMessage(
                role=MessageRole.ASSISTANT,
                content={"path": path_image, "mime_type": f"image/{path_image.split('.')[-1]}"},
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
            content={"path": final_answer.to_string(), "mime_type": "image/png"},
            metadata={"status": "done"},
        )
    elif isinstance(final_answer, AgentAudio):
        yield gr.ChatMessage(
            role=MessageRole.ASSISTANT,
            content={"path": final_answer.to_string(), "mime_type": "audio/wav"},
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

    def _copy_into_uploads(self, source_path: Path) -> str:
        if self.file_upload_folder is None:
            raise ValueError("File uploads are disabled because no upload folder was provided.")
        sanitized_name = self._sanitize_filename(source_path.name)
        destination = self.file_upload_folder / sanitized_name
        counter = 1
        while destination.exists():
            destination = self.file_upload_folder / f"{destination.stem}_{counter}{destination.suffix}"
            counter += 1
        shutil.copy(source_path, destination)
        return str(destination)

    def _extract_prompt_and_files(self, prompt_payload) -> tuple[str, list[str]]:
        if prompt_payload is None:
            return "", []

        if isinstance(prompt_payload, list):
            text_segments: list[str] = []
            saved_files: list[str] = []
            for item in prompt_payload:
                if not isinstance(item, dict):
                    continue
                if item.get("type") == "text" and item.get("text"):
                    text_segments.append(item["text"].strip())
                elif item.get("type") == "file":
                    file_payload = item.get("file") or item.get("path")
                    resolved_path = self._resolve_file_path(file_payload)
                    if resolved_path is None:
                        continue
                    if not self._extension_is_allowed(resolved_path):
                        raise ValueError(f"File type {resolved_path.suffix} is not allowed.")
                    saved_files.append(self._copy_into_uploads(resolved_path))
            return "\n".join(filter(None, text_segments)).strip(), saved_files

        if isinstance(prompt_payload, str):
            return prompt_payload.strip(), []

        resolved_path = self._resolve_file_path(prompt_payload)
        if resolved_path is not None:
            if not self._extension_is_allowed(resolved_path):
                raise ValueError(f"File type {resolved_path.suffix} is not allowed.")
            saved_file = self._copy_into_uploads(resolved_path)
            return "", [saved_file]

        return str(prompt_payload).strip(), []

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

    def load_latest_output_files(self) -> dict | None:
        candidate_dirs: list[Path] = []
        for base in self.output_base_folders:
            if not base.exists():
                continue
            for child in base.iterdir():
                if child.is_dir():
                    candidate_dirs.append(child)

        if not candidate_dirs:
            return None

        latest_dir = max(candidate_dirs, key=lambda folder: folder.stat().st_mtime)
        file_entries = []
        for child in sorted(latest_dir.iterdir()):
            if not child.is_file():
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

        return {"folder": str(latest_dir), "files": file_entries}

    def update_output_box(self):
        import gradio as gr

        payload = self.load_latest_output_files()
        files = payload.get("files", []) if payload else []
        if not files:
            return (
                gr.update(value="`output/` 中暂无可供下载的文件。", visible=True),
                gr.update(value=None, visible=False),
            )

        downloadable_files = [entry["path"] for entry in files]

        return (
            gr.update(value="", visible=False),
            gr.update(value=downloadable_files, visible=True, file_count="multiple"),
        )

    def interact_with_agent(self, prompt, messages, session_state):
        import gradio as gr

        if "agent" not in session_state:
            session_state["agent"] = self.agent

        attached_files = session_state.get("latest_files", []) or []

        exec_index: int | None = None
        try:
            display_content = prompt or ""
            if attached_files:
                attachment_lines = "\n".join(f"- {Path(path).name}" for path in attached_files)
                attachment_note = (
                    "\n\n_Attachments saved locally (not sent to the agent):_\n" + attachment_lines
                )
                display_content = (display_content + attachment_note).strip()

            if not display_content:
                display_content = "(files uploaded; no prompt text)"

            messages.append(gr.ChatMessage(role="user", content=display_content, metadata={"status": "done"}))
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

            for msg in stream_to_gradio(
                session_state["agent"], task=prompt, reset_agent_memory=self.reset_agent_memory
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
            yield messages, _full_reason_update()
            raise gr.Error(f"Error in interaction: {str(e)}")
        finally:
            session_state["latest_files"] = []
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
            prompt_text, saved_files = self._extract_prompt_and_files(prompt_payload)
        except ValueError as exc:
            raise gr.Error(str(exc)) from exc

        if not prompt_text and not saved_files:
            raise gr.Error("Please enter a prompt or attach a supported file.")

        session_state.setdefault("latest_files", [])
        session_state["latest_files"] = saved_files

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

        with gr.Blocks(theme="ocean", fill_height=True) as demo:
            # Add session state to store session-specific data
            session_state = gr.State({})
            stored_messages = gr.State("")
            file_uploads_log = gr.State([])

            with gr.Sidebar():
                gr.Markdown(
                    f"# {self.name.replace('_', ' ').capitalize()}"
                    "\n> This web ui allows you to interact with a `smolagents` agent that can use tools and execute steps to complete tasks."
                    + (f"\n\n**Agent description:**\n{self.description}" if self.description else "")
                )
                with gr.Accordion("输出文件", open=False):
                    output_status = gr.Markdown("`output/` 中暂无可供下载的文件。")
                    output_downloads = gr.Files(
                        label="全部文件下载",
                        file_count="multiple",
                        interactive=False,
                        visible=False,
                        elem_classes=["output-downloads"],
                    )
                    refresh_outputs = gr.Button("刷新输出文件", variant="secondary")
                    gr.HTML(
                        """<style>
.output-downloads {
    width: 100%;
    max-width: 100%;
}

.output-downloads .file-preview {
    display: grid;
    grid-template-columns: 1fr auto;
    gap: 0.5rem;
    align-items: center;
    width: 100%;
    overflow: visible;
}

.output-downloads .file-preview li {
    width: 100%;
    display: contents;
}

.output-downloads .file-preview span,
.output-downloads .file-preview a,
.output-downloads .file-preview button {
    white-space: normal;
    word-break: break-word;
    overflow-wrap: anywhere;
}

.output-downloads .file-preview button {
    width: auto;
    justify-self: end;
}
</style>"""
                    )

            with gr.Column(scale=1, elem_classes=["agent-column"]):
                chatbot = gr.Chatbot(
                    label="Agent",
                    type="messages",
                    avatar_images=(
                        None,
                        "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/smolagents/mascot_smol.png",
                    ),
                    resizeable=True,
                    scale=1,
                    latex_delimiters=[
                        {"left": r"$$", "right": r"$$", "display": True},
                        {"left": r"$", "right": r"$", "display": False},
                        {"left": r"\[", "right": r"\]", "display": True},
                        {"left": r"\(", "right": r"\)", "display": False},
                    ],
                )

                with gr.Accordion("🔎 Full reasoning", open=False, elem_classes=["full-reasoning-accordion"]):
                    full_reasoning_md = gr.Markdown("_No reasoning yet._", elem_classes=["full-reasoning-md"])

                supports_inline_uploads = self.file_upload_folder is not None and hasattr(gr, "MultimodalTextbox")
                placeholder = "Enter a prompt. Paste or drop files to save them locally (not shared with the agent)."

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

            gr.HTML(
                """<style>
.agent-column {
    min-height: 60vh;
}
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
</style>"""
            )

            output_components = [
                output_status,
                output_downloads,
            ]

            demo.load(self.update_output_box, None, output_components)
            refresh_outputs.click(self.update_output_box, None, output_components)

            text_input.submit(
                self.log_user_message,
                [text_input, file_uploads_log, session_state],
                [stored_messages, text_input, file_uploads_log],
            ).then(
                self.interact_with_agent,
                [stored_messages, chatbot, session_state],
                [chatbot, full_reasoning_md],
            ).then(
                self.update_output_box,
                None,
                output_components,
            ).then(
                lambda: gr.update(interactive=True),
                None,
                [text_input],
            )

            chatbot.clear(self.agent.memory.reset)
        return demo


__all__ = ["stream_to_gradio", "GradioUI"]