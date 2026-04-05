"""
gap_analyzer.py — Syllabus coverage gap detection for CBSE Math.

How it works
------------
1. Takes a list of ingested text chunks (from NCERT + past papers).
2. For each chapter in the CBSE syllabus, uses keyword matching + subtopic
   matching to estimate how well that chapter is covered in the source material.
3. Produces a ``CoverageReport`` that lists:
   - Well-covered chapters (enough examples)
   - Partially-covered chapters (some subtopics missing)
   - Uncovered chapters (no source examples found)
4. The ``MathGenerator`` uses the CoverageReport to prioritise generation of
   ``fill_gap`` items for underrepresented subtopics.

Coverage score
--------------
   score = (keyword_hits / total_keywords) * 0.5
         + (subtopic_hits / total_subtopics) * 0.5

   score ≥ 0.60  → COVERED
   score ∈ [0.25, 0.60) → PARTIAL
   score < 0.25  → GAP
"""
from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from typing import Literal

from cbse_math.cbse_syllabus import Chapter, ClassLevel, chapters_for_class

logger = logging.getLogger(__name__)

CoverageStatus = Literal["covered", "partial", "gap"]


@dataclass
class SubtopicCoverage:
    subtopic: str
    covered: bool
    matched_keywords: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "subtopic": self.subtopic,
            "covered": self.covered,
            "matched_keywords": self.matched_keywords,
        }


@dataclass
class ChapterCoverage:
    chapter: Chapter
    score: float                                   # 0.0 – 1.0
    status: CoverageStatus
    subtopic_details: list[SubtopicCoverage] = field(default_factory=list)
    uncovered_subtopics: list[str] = field(default_factory=list)
    matched_keywords: list[str] = field(default_factory=list)
    example_count: int = 0                         # number of example problems found

    def to_dict(self) -> dict:
        return {
            "chapter": {
                "chapter_id": self.chapter.chapter_id,
                "title": self.chapter.title,
                "unit": self.chapter.unit,
                "class_level": self.chapter.class_level,
                "marks": self.chapter.marks,
                "subtopics": self.chapter.subtopics,
                "problem_types": self.chapter.problem_types,
                "keywords": self.chapter.keywords,
            },
            "score": self.score,
            "status": self.status,
            "subtopic_details": [s.to_dict() for s in self.subtopic_details],
            "uncovered_subtopics": self.uncovered_subtopics,
            "matched_keywords": self.matched_keywords,
            "example_count": self.example_count,
        }


@dataclass
class CoverageReport:
    class_level: ClassLevel
    total_chapters: int
    covered_count: int
    partial_count: int
    gap_count: int
    chapter_coverages: list[ChapterCoverage] = field(default_factory=list)
    overall_coverage_pct: float = 0.0

    # Convenience accessors
    @property
    def gap_chapters(self) -> list[ChapterCoverage]:
        return [c for c in self.chapter_coverages if c.status == "gap"]

    @property
    def partial_chapters(self) -> list[ChapterCoverage]:
        return [c for c in self.chapter_coverages if c.status == "partial"]

    def prioritised_gaps(self) -> list[tuple[ChapterCoverage, str]]:
        """
        Return (ChapterCoverage, subtopic) pairs ordered by priority for gap-fill
        generation.  GAP chapters come first, then PARTIAL chapters.
        """
        result: list[tuple[ChapterCoverage, str]] = []
        for cc in self.gap_chapters:
            for sub in cc.uncovered_subtopics or cc.chapter.subtopics:
                result.append((cc, sub))
        for cc in self.partial_chapters:
            for sub in cc.uncovered_subtopics:
                result.append((cc, sub))
        return result

    def summary(self) -> str:
        lines = [
            f"Coverage Report — Class {self.class_level}",
            f"  Total chapters : {self.total_chapters}",
            f"  Covered        : {self.covered_count}",
            f"  Partial        : {self.partial_count}",
            f"  Gap            : {self.gap_count}",
            f"  Overall        : {self.overall_coverage_pct:.1f}%",
            "",
            "GAP chapters (need fill_gap problems):",
        ]
        for cc in self.gap_chapters:
            lines.append(f"  [{cc.chapter.chapter_id}] {cc.chapter.title} (score={cc.score:.2f})")
        lines.append("")
        lines.append("PARTIAL chapters (missing subtopics):")
        for cc in self.partial_chapters:
            for sub in cc.uncovered_subtopics:
                lines.append(f"  [{cc.chapter.chapter_id}] {cc.chapter.title} — missing: {sub}")
        return "\n".join(lines)

    def to_dict(self) -> dict:
        return {
            "class_level": self.class_level,
            "total_chapters": self.total_chapters,
            "covered_count": self.covered_count,
            "partial_count": self.partial_count,
            "gap_count": self.gap_count,
            "overall_coverage_pct": self.overall_coverage_pct,
            "chapter_coverages": [c.to_dict() for c in self.chapter_coverages],
            "prioritised_gaps": [
                {"chapter_id": cc.chapter.chapter_id, "chapter_title": cc.chapter.title, "subtopic": sub}
                for cc, sub in self.prioritised_gaps()
            ],
        }


# ─────────────────────────────────────────────────────────────────────────────
# Analyser
# ─────────────────────────────────────────────────────────────────────────────

# Thresholds
_COVERED_THRESHOLD = 0.60
_PARTIAL_THRESHOLD = 0.25

# Minimum word-length for keyword matching (avoid false positives on tiny words)
_MIN_KW_LEN = 3

# Patterns that indicate a solved worked example or problem
_EXAMPLE_PATTERNS = [
    re.compile(r"\bexample\b", re.IGNORECASE),
    re.compile(r"\bsolution\b", re.IGNORECASE),
    re.compile(r"\bsolved\b", re.IGNORECASE),
    re.compile(r"\bfind\s+the\b", re.IGNORECASE),
    re.compile(r"\bprove\s+that\b", re.IGNORECASE),
    re.compile(r"\bevaluate\b", re.IGNORECASE),
    re.compile(r"\bcalculate\b", re.IGNORECASE),
    re.compile(r"\bif\b.*\bfind\b", re.IGNORECASE),
]


def _count_examples(text: str) -> int:
    return sum(bool(p.search(text)) for p in _EXAMPLE_PATTERNS)


def _keyword_hits(text_lower: str, keywords: list[str]) -> list[str]:
    hits = []
    for kw in keywords:
        if len(kw) < _MIN_KW_LEN:
            continue
        if kw.lower() in text_lower:
            hits.append(kw)
    return hits


def _subtopic_covered(text_lower: str, subtopic: str) -> tuple[bool, list[str]]:
    """
    Determine if a subtopic is covered in the source text.
    Uses individual significant words from the subtopic as signal.
    """
    # Extract content words (4+ chars) from the subtopic description
    words = [w.lower() for w in re.findall(r"[A-Za-z]{4,}", subtopic)]
    if not words:
        return False, []
    matched = [w for w in words if w in text_lower]
    covered = len(matched) / len(words) >= 0.4
    return covered, matched


class GapAnalyzer:
    """
    Analyses coverage of the CBSE syllabus in a set of source text chunks.

    Args:
        class_level: CBSE class to analyse against (10 or 12).
    """

    def __init__(self, class_level: ClassLevel = 12):
        self.class_level = class_level
        self.chapters = chapters_for_class(class_level)

    def analyse(self, chunks: list[str]) -> CoverageReport:
        """
        Analyse a list of text chunks and return a CoverageReport.

        Args:
            chunks: Plain text chunks from ingested NCERT / past paper files.
        """
        combined = "\n\n".join(chunks).lower()

        chapter_coverages: list[ChapterCoverage] = []

        for ch in self.chapters:
            # ── keyword coverage ─────────────────────────────────────────────
            kw_hits = _keyword_hits(combined, ch.keywords)
            kw_score = len(kw_hits) / max(len(ch.keywords), 1)

            # ── subtopic coverage ────────────────────────────────────────────
            sub_details: list[SubtopicCoverage] = []
            uncovered: list[str] = []

            for sub in ch.subtopics:
                covered, sub_kws = _subtopic_covered(combined, sub)
                sub_details.append(SubtopicCoverage(
                    subtopic=sub,
                    covered=covered,
                    matched_keywords=sub_kws,
                ))
                if not covered:
                    uncovered.append(sub)

            sub_score = (len(ch.subtopics) - len(uncovered)) / max(len(ch.subtopics), 1)

            # ── composite score ──────────────────────────────────────────────
            score = kw_score * 0.5 + sub_score * 0.5

            if score >= _COVERED_THRESHOLD:
                status: CoverageStatus = "covered"
            elif score >= _PARTIAL_THRESHOLD:
                status = "partial"
            else:
                status = "gap"

            example_count = sum(_count_examples(c) for c in chunks if any(
                kw.lower() in c.lower() for kw in ch.keywords
            ))

            chapter_coverages.append(ChapterCoverage(
                chapter=ch,
                score=score,
                status=status,
                subtopic_details=sub_details,
                uncovered_subtopics=uncovered,
                matched_keywords=kw_hits,
                example_count=example_count,
            ))

        covered_count = sum(1 for c in chapter_coverages if c.status == "covered")
        partial_count = sum(1 for c in chapter_coverages if c.status == "partial")
        gap_count = sum(1 for c in chapter_coverages if c.status == "gap")
        n = len(chapter_coverages)
        overall = (covered_count * 1.0 + partial_count * 0.5) / max(n, 1) * 100

        report = CoverageReport(
            class_level=self.class_level,
            total_chapters=n,
            covered_count=covered_count,
            partial_count=partial_count,
            gap_count=gap_count,
            chapter_coverages=chapter_coverages,
            overall_coverage_pct=round(overall, 1),
        )

        logger.info(
            "Gap analysis complete — Class %d: %d covered, %d partial, %d gaps (%.1f%% overall)",
            self.class_level, covered_count, partial_count, gap_count, overall,
        )
        return report
