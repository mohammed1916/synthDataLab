"""
math/ — CBSE Mathematics synthetic content generation package.

Sub-modules
-----------
- ``cbse_syllabus``   : CBSE chapter/topic/marks registry for Classes 9–12.
- ``math_schema``     : Pydantic-style schema for math problems & explanations.
- ``math_prompts``    : LaTeX-output prompt templates for math generation.
- ``gap_analyzer``    : Coverage analysis against the CBSE syllabus.
- ``math_generator``  : Orchestrates problem generation from NCERT / past-papers.
- ``pdf_ingestor``    : PDF-to-text with math expression preservation.

Package name is ``cbse_math`` (not ``math``) to avoid shadowing Python's stdlib.
"""
