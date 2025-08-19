from __future__ import annotations

import base64
import mimetypes
import os
from typing import List, Tuple

import fitz  # PyMuPDF
from docx import Document  # python-docx
from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI


def _read_image_as_data_url(file_path: str) -> str:
    """Reads an image file and returns a data URL suitable for GPT-4o image input."""
    mime_type, _ = mimetypes.guess_type(file_path)
    if not mime_type:
        mime_type = "image/png"
    with open(file_path, "rb") as f:
        image_bytes = f.read()
    b64 = base64.b64encode(image_bytes).decode("utf-8")
    return f"data:{mime_type};base64,{b64}"


def _extract_pdf_text(file_path: str, max_pages: int = 5, max_chars: int = 12000) -> str:
    """Extracts text from a PDF file, limiting pages and characters to control token usage."""
    text_parts: List[str] = []
    with fitz.open(file_path) as doc:
        for page_index, page in enumerate(doc):
            if page_index >= max_pages:
                break
            text_parts.append(page.get_text())
            if sum(len(p) for p in text_parts) >= max_chars:
                break
    text = "\n".join(text_parts)
    if len(text) > max_chars:
        text = text[:max_chars] + "\n... [truncated]"
    return text.strip()


def _extract_docx_text(file_path: str, max_chars: int = 12000) -> str:
    """Extracts text from a DOCX file with a character limit."""
    doc = Document(file_path)
    parts: List[str] = []
    for para in doc.paragraphs:
        if para.text:
            parts.append(para.text)
            if sum(len(p) for p in parts) >= max_chars:
                break
    text = "\n".join(parts)
    if len(text) > max_chars:
        text = text[:max_chars] + "\n... [truncated]"
    return text.strip()


def _build_multimodal_content(file_paths: List[str]) -> Tuple[List[dict], List[str]]:
    """
    Builds a list of content parts for a multimodal ChatOpenAI call and a list of warnings.
    Returns: (content_parts, warnings)
    """
    content: List[dict] = []
    warnings: List[str] = []

    for idx, path in enumerate(file_paths, start=1):
        if not os.path.exists(path):
            warnings.append(f"File not found: {path}")
            continue

        mime_type, _ = mimetypes.guess_type(path)
        title_text = f"[File {idx}] {os.path.basename(path)}"
        content.append({"type": "text", "text": title_text})

        try:
            if mime_type and mime_type.startswith("image/"):
                data_url = _read_image_as_data_url(path)
                content.append({"type": "image_url", "image_url": {"url": data_url}})
            elif mime_type == "application/pdf":
                pdf_text = _extract_pdf_text(path)
                content.append({"type": "text", "text": pdf_text or "[No extractable text]"})
            elif mime_type in ("application/vnd.openxmlformats-officedocument.wordprocessingml.document",):
                docx_text = _extract_docx_text(path)
                content.append({"type": "text", "text": docx_text or "[No extractable text]"})
            else:
                # Fallback: try to read as text
                try:
                    with open(path, "r", encoding="utf-8", errors="replace") as f:
                        text = f.read()
                    content.append({"type": "text", "text": text[:12000]})
                except Exception:
                    warnings.append(f"Unsupported file type (skipped content): {path}")
        except Exception as e:
            warnings.append(f"Error reading {path}: {e}")

    return content, warnings


essential_rules_en = (
    "Do not diagnose. Summarize findings, explain simply, and advise follow‑up with a doctor."
)
essential_rules_ar = (
    "قدّم تشخيصًا مبدئي واقترح بعض الأدوية والعلاج . لخّص النتائج ببساطة، واشرحها للمريض، "
)


def interpret_reports_with_gpt4o(file_paths: List[str], lang: str) -> str:
    """
    Uses a dedicated multimodal agent (GPT-4o) to interpret one or more medical reports
    including images (JPG/PNG), PDFs, and DOCX files.
    Returns a patient-friendly explanation in the requested language.
    """
    system_prompt_en = (
        "You are an empathetic medical reports interpreter. Summarize findings clearly for a patient, "
        "avoid diagnosis, and highlight any red flags that require a doctor's follow‑up. "
        "Be concise, structured, and reassuring. "
        + essential_rules_en
    )
    system_prompt_ar = (
        "أنت وكيل ذكي لشرح التقارير الطبية للمريض بأسلوب واضح ومطمئن دون تشخيص. "
        "لخّص النتائج ونقاط القلق إن وجدت، . "
        + essential_rules_ar
    )

    preface_en = (
        "Please review the following files (images, PDFs, DOCX). "
        "Extract key findings, explain them simply, and recommend next steps."
    )
    preface_ar = (
        "من فضلك راجع الملفات التالية (صور، PDF، DOCX). "
        "استخرج أهم النقاط واشرحها ببساطة وقدّم نصائح للخطوات القادمة."
    )

    content_parts, warnings = _build_multimodal_content(file_paths)
    content_parts.insert(0, {"type": "text", "text": preface_ar if lang == "ar" else preface_en})

    llm = ChatOpenAI(model="gpt-4.1-mini", temperature=0.2)
    response = llm.invoke([
        ("system", system_prompt_ar if lang == "ar" else system_prompt_en),
        HumanMessage(content=content_parts),
    ])

    output = response.content.strip()
    if warnings:
        note = ("\n\n[Note: Some files could not be read/parsed: " + "; ".join(warnings) + "]")
        output += note
    return output


def build_reports_text_context(file_paths: List[str], max_total_chars: int = 16000) -> str:
    """
    Builds a textual context from the uploaded reports for follow-up Q&A turns.
    This avoids repeatedly sending large binary data and is suitable for chat prompts.
    """
    parts: List[str] = []
    total = 0
    for idx, path in enumerate(file_paths, start=1):
        if not os.path.exists(path):
            continue
        mime_type, _ = mimetypes.guess_type(path)
        header = f"\n---\n[Report {idx}] {os.path.basename(path)} ({mime_type or 'unknown'})\n"
        body = ""
        try:
            if mime_type and mime_type.startswith("image/"):
                body = "[Image uploaded. Content referenced during initial interpretation.]"
            elif mime_type == "application/pdf":
                body = _extract_pdf_text(path)
            elif mime_type in ("application/vnd.openxmlformats-officedocument.wordprocessingml.document",):
                body = _extract_docx_text(path)
            else:
                try:
                    with open(path, "r", encoding="utf-8", errors="replace") as f:
                        body = f.read()
                except Exception:
                    body = "[Unsupported file type for text extraction]"
        except Exception as e:
            body = f"[Error reading file: {e}]"

        snippet = header + (body or "[No extractable text]")
        parts.append(snippet)
        total += len(snippet)
        if total >= max_total_chars:
            break

    context = "\n".join(parts).strip()
    if not context:
        context = "[No textual context available from uploaded reports]"
    return context
