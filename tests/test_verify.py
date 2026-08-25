"""Tests for verification module."""

import pandas as pd

from womblex.verify.engine import (
    VerificationConfig,
    compute_garbled_ratio,
    compute_garbled_redaction_ratio,
    is_valid_redaction,
    run_structural_verification,
    run_weak_signal_scan,
)


class TestRedactionPatterns:
    """Test redaction pattern detection."""
    
    def test_valid_redaction_asterisks(self):
        assert is_valid_redaction("***") is True
        assert is_valid_redaction("*****") is True
    
    def test_valid_redaction_underscores(self):
        assert is_valid_redaction("___") is True
        assert is_valid_redaction("_____") is True
    
    def test_invalid_redaction_too_short(self):
        assert is_valid_redaction("**") is False
        assert is_valid_redaction("*") is False
    
    def test_invalid_redaction_mixed(self):
        assert is_valid_redaction("*_*") is False
        assert is_valid_redaction("**__") is False


class TestGarbledRatio:
    """Test garbled text ratio computation."""
    
    def test_clean_text(self):
        text = "This is clean text with normal punctuation."
        ratio = compute_garbled_ratio(text)
        assert ratio < 0.1
    
    def test_garbled_text(self):
        text = "Th!$ !$ g@rbl3d t3xt w!th w3!rd ch@r$"
        ratio = compute_garbled_ratio(text)
        # This text has some special chars but not extreme
        assert ratio > 0.1
    
    def test_text_with_valid_redactions(self):
        text = "Name: ***** Address: _____ Phone: #####"
        ratio = compute_garbled_ratio(text)
        # Valid redactions should be excluded
        assert ratio < 0.1
    
    def test_empty_text(self):
        assert compute_garbled_ratio("") == 0.0


class TestGarbledRedactionRatio:
    """Test garbled redaction ratio computation."""
    
    def test_all_valid_redactions(self):
        text = "Name: ***** and Address: _____"
        ratio = compute_garbled_redaction_ratio(text)
        assert ratio == 0.0
    
    def test_mixed_redactions(self):
        text = "Name: ***** and broken: *_ and valid: ___"
        ratio = compute_garbled_redaction_ratio(text)
        # 1 garbled out of 3 = 0.33
        assert 0.3 < ratio < 0.4
    
    def test_no_redactions(self):
        text = "No redactions here at all"
        ratio = compute_garbled_redaction_ratio(text)
        assert ratio == 0.0


class TestStructuralVerification:
    """Test structural verification."""
    
    def test_valid_dataframe(self):
        df = pd.DataFrame({
            "document_id": ["doc1", "doc2"],
            "source_path": ["/path/1", "/path/2"],
            "text": ["Some text here", "More text here"],
        })
        config = VerificationConfig()
        passed, errors = run_structural_verification(df, config)
        assert passed is True
        assert len(errors) == 0
    
    def test_missing_columns(self):
        df = pd.DataFrame({
            "document_id": ["doc1"],
        })
        config = VerificationConfig()
        passed, errors = run_structural_verification(df, config)
        assert passed is False
        assert any("Missing required columns" in e for e in errors)
    
    def test_duplicate_ids(self):
        df = pd.DataFrame({
            "document_id": ["doc1", "doc1"],
            "source_path": ["/path/1", "/path/2"],
            "text": ["Some text", "More text"],
        })
        config = VerificationConfig()
        passed, errors = run_structural_verification(df, config)
        assert passed is False
        assert any("duplicate" in e for e in errors)


class TestWeakSignalScan:
    """Test weak signal scanning."""
    
    def test_low_confidence_flagged(self):
        df = pd.DataFrame({
            "document_id": ["doc1"],
            "source_path": ["/path/1"],
            "doc_type": ["native"],
            "confidence": [0.5],
            "text": ["Normal text here"],
        })
        config = VerificationConfig(min_confidence=0.7)
        flagged = run_weak_signal_scan(df, config)
        assert len(flagged) == 1
        assert "low_confidence" in flagged[0].signals
    
    def test_page_count_anomaly(self):
        df = pd.DataFrame({
            "document_id": ["doc1"],
            "source_path": ["/path/1"],
            "doc_type": ["native"],
            "page_count": [0],
            "text": ["Normal text"],
        })
        config = VerificationConfig()
        flagged = run_weak_signal_scan(df, config)
        assert len(flagged) == 1
        assert "page_count_anomaly" in flagged[0].signals
    
    def test_clean_document_not_flagged(self):
        df = pd.DataFrame({
            "document_id": ["doc1"],
            "source_path": ["/path/1"],
            "doc_type": ["native"],
            "confidence": [0.9],
            "page_count": [5],
            "text": ["This is clean normal text with proper formatting."],
        })
        config = VerificationConfig()
        flagged = run_weak_signal_scan(df, config)
        assert len(flagged) == 0
