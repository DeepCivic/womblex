"""Verification engine for document extraction quality checks.

Two-pass verification:
1. Structural: Schema validation, uniqueness, type constraints
2. Weak-signal scan: Flag documents with potential quality issues
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pandas as pd


@dataclass
class VerificationConfig:
    """Configuration for verification checks."""
    
    # Structural - match actual output schema
    required_columns: list[str] = field(default_factory=lambda: [
        "document_id", "source_path", "text"
    ])
    
    # Weak signals
    min_confidence: float = 0.7
    max_garbled_ratio: float = 0.3
    max_garbled_redaction_ratio: float = 0.05
    max_page_count: int = 200
    
    # Thresholds
    fail_on_flagged_ratio: float = 0.15


@dataclass
class FlaggedDocument:
    """A document flagged by weak-signal scan."""
    
    document_id: str
    signals: list[str]
    snippet: str = ""


@dataclass
class VerificationResult:
    """Result of verification run."""
    
    overall_status: str = "passed"  # passed, warning, failed
    structural_passed: bool = True
    structural_errors: list[str] = field(default_factory=list)
    flagged_count: int = 0
    flagged_docs: list[FlaggedDocument] = field(default_factory=list)
    total_docs: int = 0


# Valid redaction pattern: 3+ identical chars from set
REDACTION_CHARS = set("*x_#=-")
VALID_REDACTION_PATTERN = re.compile(r"([*x_#=-])\1{2,}")


def is_valid_redaction(segment: str) -> bool:
    """Check if a segment is a valid redaction pattern."""
    if len(segment) < 3:
        return False
    return bool(VALID_REDACTION_PATTERN.fullmatch(segment))


def compute_garbled_ratio(text: str) -> float:
    """Compute ratio of non-alphanumeric chars excluding valid redactions and common punctuation."""
    if not text:
        return 0.0
    
    # Remove valid redaction patterns
    cleaned = VALID_REDACTION_PATTERN.sub("", text)
    
    # Common punctuation to allow
    allowed_punct = set(".,;:!?'\"()-/\n\t ")
    
    total = len(cleaned)
    if total == 0:
        return 0.0
    
    garbled = sum(1 for c in cleaned if not c.isalnum() and c not in allowed_punct)
    return garbled / total


def compute_garbled_redaction_ratio(text: str) -> float:
    """Compute ratio of garbled redaction segments vs total redaction-like segments."""
    if not text:
        return 0.0
    
    # Find all redaction-like segments (sequences containing redaction chars)
    redaction_segments = re.findall(r"[*x_#=-]{2,}", text)
    
    if not redaction_segments:
        return 0.0
    
    garbled = sum(1 for seg in redaction_segments if not is_valid_redaction(seg))
    return garbled / len(redaction_segments)


def check_weak_signals(row: dict[str, Any], config: VerificationConfig) -> list[str]:
    """Check a document row for weak signals of quality issues."""
    signals = []
    
    # Low confidence (only if column exists)
    confidence = row.get("confidence")
    if confidence is not None and confidence < config.min_confidence:
        signals.append("low_confidence")
    
    # Page count anomaly (only if column exists)
    page_count = row.get("page_count")
    if page_count is not None:
        if page_count == 0 or page_count > config.max_page_count:
            signals.append("page_count_anomaly")
    
    # Text-based checks
    text = row.get("text", "")
    if text:
        # Garbled text
        garbled_ratio = compute_garbled_ratio(text)
        if garbled_ratio > config.max_garbled_ratio:
            signals.append("garbled_text")
        
        # Garbled redactions
        redaction_ratio = compute_garbled_redaction_ratio(text)
        if redaction_ratio > config.max_garbled_redaction_ratio:
            signals.append("redaction_garbled")
    
    return signals


def run_structural_verification(
    df: pd.DataFrame,
    config: VerificationConfig,
) -> tuple[bool, list[str]]:
    """Run structural verification checks on the dataframe."""
    errors = []
    
    # Check required columns
    missing = [c for c in config.required_columns if c not in df.columns]
    if missing:
        errors.append(f"Missing required columns: {missing}")
    
    # Check document_id uniqueness
    if "document_id" in df.columns:
        duplicates = df["document_id"].duplicated().sum()
        if duplicates > 0:
            errors.append(f"Found {duplicates} duplicate document_id values")
    
    # Check confidence range if present
    if "confidence" in df.columns:
        invalid_conf = ((df["confidence"] < 0) | (df["confidence"] > 1)).sum()
        if invalid_conf > 0:
            errors.append(f"Found {invalid_conf} rows with confidence outside [0, 1]")
    
    # Check page_count if present
    if "page_count" in df.columns:
        negative_pages = (df["page_count"] < 0).sum()
        if negative_pages > 0:
            errors.append(f"Found {negative_pages} rows with negative page_count")
    
    return len(errors) == 0, errors


def run_weak_signal_scan(
    df: pd.DataFrame,
    config: VerificationConfig,
) -> list[FlaggedDocument]:
    """Run weak-signal scan on all documents."""
    flagged = []
    
    for _, row in df.iterrows():
        signals = check_weak_signals(
            {str(k): v for k, v in row.to_dict().items()}, config
        )
        if signals:
            doc_id = row.get("document_id", "unknown")
            text = row.get("text", "")
            snippet = text[:200] if text else ""
            flagged.append(FlaggedDocument(
                document_id=doc_id,
                signals=signals,
                snippet=snippet,
            ))
    
    return flagged


def run_verifications(
    parquet_path: str | Path,
    config: VerificationConfig | None = None,
) -> VerificationResult:
    """Run all verification passes on a parquet file."""
    if config is None:
        config = VerificationConfig()
    
    result = VerificationResult()
    
    # Load parquet
    df = pd.read_parquet(parquet_path)
    result.total_docs = len(df)
    
    # Structural verification
    passed, errors = run_structural_verification(df, config)
    result.structural_passed = passed
    result.structural_errors = errors
    
    if not passed:
        result.overall_status = "failed"
        return result
    
    # Weak-signal scan
    flagged = run_weak_signal_scan(df, config)
    result.flagged_docs = flagged
    result.flagged_count = len(flagged)
    
    # Determine overall status
    if result.flagged_count == 0:
        result.overall_status = "passed"
    else:
        flagged_ratio = result.flagged_count / result.total_docs if result.total_docs > 0 else 0
        if flagged_ratio > config.fail_on_flagged_ratio:
            result.overall_status = "failed"
        else:
            result.overall_status = "warning"
    
    return result
