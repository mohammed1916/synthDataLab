"""
cbse_syllabus.py — CBSE Mathematics syllabus registry for Classes 10 and 12.

Structure
---------
Each entry maps to one CBSE unit/chapter with:
  - chapter id       : unique slug
  - title            : official NCERT chapter name
  - unit             : parent unit name (for marks calculation)
  - class_level      : 10 | 12
  - marks            : typical weightage (may vary by year; 2024–25 pattern)
  - subtopics        : list of specific subtopics / learning outcomes
  - problem_types    : typical CBSE question formats for this chapter
  - keywords         : alternative names & symbols for topic detection

Source references
-----------------
  CBSE Class 12 Maths Syllabus 2024–25 (Board Notice 23 Aug 2024)
  CBSE Class 10 Maths Basic/Standard Syllabus 2024–25
  NCERT Textbook chapter headings (Classes 10 and 12)
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

ClassLevel = Literal[10, 12]


@dataclass
class Chapter:
    chapter_id: str
    title: str
    unit: str
    class_level: ClassLevel
    marks: int                         # typical board exam marks weightage
    subtopics: list[str] = field(default_factory=list)
    problem_types: list[str] = field(default_factory=list)
    keywords: list[str] = field(default_factory=list)


# ─────────────────────────────────────────────────────────────────────────────
# CLASS 12 — Mathematics (Theory: 80 marks + Internal: 20 marks)
# ─────────────────────────────────────────────────────────────────────────────

CLASS_12_CHAPTERS: list[Chapter] = [
    Chapter(
        chapter_id="c12_relations_functions",
        title="Relations and Functions",
        unit="Relations and Functions",
        class_level=12,
        marks=8,
        subtopics=[
            "Types of relations: reflexive, symmetric, transitive, equivalence",
            "Types of functions: one-one, onto, bijective",
            "Composite functions and inverse of a function",
            "Binary operations",
        ],
        problem_types=["short_answer_1m", "short_answer_2m", "case_study"],
        keywords=["relation", "function", "domain", "range", "bijective", "surjective", "injective", "composition"],
    ),
    Chapter(
        chapter_id="c12_inverse_trig",
        title="Inverse Trigonometric Functions",
        unit="Relations and Functions",
        class_level=12,
        marks=8,
        subtopics=[
            "Definition and range of inverse trig functions",
            "Graphs of inverse trig functions",
            "Principal value branch",
            "Elementary properties and identities",
        ],
        problem_types=["short_answer_1m", "short_answer_2m", "long_answer_3m"],
        keywords=["arcsin", "arccos", "arctan", "sin⁻¹", "cos⁻¹", "tan⁻¹", "principal value"],
    ),
    Chapter(
        chapter_id="c12_matrices",
        title="Matrices",
        unit="Algebra",
        class_level=12,
        marks=10,
        subtopics=[
            "Concept, notation, order, equality, types of matrices",
            "Operations: addition, scalar multiplication, matrix multiplication",
            "Transpose of a matrix; symmetric and skew-symmetric",
            "Elementary row and column operations",
            "Invertible matrices and proof of uniqueness",
        ],
        problem_types=["short_answer_2m", "long_answer_3m", "long_answer_5m"],
        keywords=["matrix", "determinant", "transpose", "symmetric", "skew", "identity", "inverse matrix"],
    ),
    Chapter(
        chapter_id="c12_determinants",
        title="Determinants",
        unit="Algebra",
        class_level=12,
        marks=10,
        subtopics=[
            "Determinant of a square matrix (up to 3×3)",
            "Minors, cofactors",
            "Applications: area of triangle, adjoint and inverse",
            "Solving system of linear equations using Cramer's rule and matrix method",
        ],
        problem_types=["short_answer_2m", "long_answer_3m", "long_answer_5m"],
        keywords=["determinant", "cofactor", "adjoint", "Cramer", "system of equations", "singular"],
    ),
    Chapter(
        chapter_id="c12_continuity_differentiability",
        title="Continuity and Differentiability",
        unit="Calculus",
        class_level=12,
        marks=18,
        subtopics=[
            "Continuity at a point and on an interval",
            "Differentiability; relationship between continuity and differentiability",
            "Derivatives of composite, implicit, inverse trig, exponential, logarithmic functions",
            "Logarithmic differentiation",
            "Derivatives of parametric functions",
            "Second-order derivatives",
            "Rolle's theorem, Mean Value Theorem",
        ],
        problem_types=["short_answer_2m", "long_answer_3m", "long_answer_5m", "case_study"],
        keywords=["continuous", "differentiable", "chain rule", "implicit", "parametric", "MVT", "Rolle"],
    ),
    Chapter(
        chapter_id="c12_applications_derivatives",
        title="Applications of Derivatives",
        unit="Calculus",
        class_level=12,
        marks=18,
        subtopics=[
            "Rate of change of quantities",
            "Increasing and decreasing functions",
            "Tangents and normals",
            "Approximations using differentials",
            "Maxima and minima (first/second derivative test)",
            "Applications in real-world optimization problems",
        ],
        problem_types=["long_answer_3m", "long_answer_5m", "case_study"],
        keywords=["rate of change", "tangent", "normal", "maxima", "minima", "increasing", "decreasing", "optimization"],
    ),
    Chapter(
        chapter_id="c12_integrals",
        title="Integrals",
        unit="Calculus",
        class_level=12,
        marks=18,
        subtopics=[
            "Integration as inverse of differentiation",
            "Integration by substitution, by parts, by partial fractions",
            "Definite integrals as limit of sum",
            "Fundamental Theorem of Calculus",
            "Properties of definite integrals",
            "Evaluation of definite integrals",
        ],
        problem_types=["short_answer_2m", "long_answer_3m", "long_answer_5m", "case_study"],
        keywords=["integral", "integration", "substitution", "by parts", "definite", "antiderivative", "∫"],
    ),
    Chapter(
        chapter_id="c12_applications_integrals",
        title="Applications of Integrals",
        unit="Calculus",
        class_level=12,
        marks=18,
        subtopics=[
            "Area under simple curves (lines, parabolas, circles)",
            "Area between two curves",
        ],
        problem_types=["long_answer_3m", "long_answer_5m"],
        keywords=["area under curve", "area between curves", "bounded region"],
    ),
    Chapter(
        chapter_id="c12_differential_equations",
        title="Differential Equations",
        unit="Calculus",
        class_level=12,
        marks=18,
        subtopics=[
            "Definition, order, degree",
            "General and particular solutions",
            "Formation of differential equations",
            "Methods of solving: variable separable, homogeneous, linear first-order",
        ],
        problem_types=["short_answer_2m", "long_answer_3m", "long_answer_5m"],
        keywords=["differential equation", "ODE", "separable", "homogeneous", "linear DE", "order", "degree"],
    ),
    Chapter(
        chapter_id="c12_vectors",
        title="Vector Algebra",
        unit="Vectors and 3D Geometry",
        class_level=12,
        marks=14,
        subtopics=[
            "Vectors and scalars; magnitude and direction",
            "Position vector, negative of a vector",
            "Addition of vectors; multiplication by scalar",
            "Dot product (scalar product) and its properties",
            "Cross product (vector product) and its properties",
            "Scalar triple product",
        ],
        problem_types=["short_answer_2m", "long_answer_3m", "long_answer_5m"],
        keywords=["vector", "scalar", "dot product", "cross product", "unit vector", "position vector", "collinear"],
    ),
    Chapter(
        chapter_id="c12_3d_geometry",
        title="Three Dimensional Geometry",
        unit="Vectors and 3D Geometry",
        class_level=12,
        marks=14,
        subtopics=[
            "Direction cosines and direction ratios of a line",
            "Equation of a line in space (vector and Cartesian)",
            "Angle between two lines; skew lines; shortest distance",
            "Equation of a plane in various forms",
            "Angle between planes; distance of a point from a plane",
            "Coplanarity of two lines",
        ],
        problem_types=["short_answer_2m", "long_answer_3m", "long_answer_5m"],
        keywords=["direction cosines", "direction ratios", "line in 3D", "plane", "skew lines", "distance from plane"],
    ),
    Chapter(
        chapter_id="c12_lpp",
        title="Linear Programming",
        unit="Linear Programming",
        class_level=12,
        marks=5,
        subtopics=[
            "Introduction, terminology: constraints, objective function, feasible region",
            "Different types of LPP (manufacturing, diet, transportation)",
            "Graphical method for solving LPP (two variables)",
            "Feasible and infeasible solutions; optimal feasible solution",
        ],
        problem_types=["long_answer_5m", "case_study"],
        keywords=["linear programming", "LPP", "objective function", "constraint", "feasible region", "corner point"],
    ),
    Chapter(
        chapter_id="c12_probability",
        title="Probability",
        unit="Probability",
        class_level=12,
        marks=8,
        subtopics=[
            "Conditional probability and its properties",
            "Multiplication theorem; independent events",
            "Total probability theorem; Bayes' theorem",
            "Random variable and its probability distribution",
            "Mean, variance of a random variable",
            "Bernoulli trials and Binomial distribution",
        ],
        problem_types=["short_answer_2m", "long_answer_3m", "long_answer_5m", "case_study"],
        keywords=["probability", "conditional probability", "Bayes", "random variable", "binomial distribution", "mean variance"],
    ),
]

# ─────────────────────────────────────────────────────────────────────────────
# CLASS 10 — Mathematics (Standard) — 80 board marks
# ─────────────────────────────────────────────────────────────────────────────

CLASS_10_CHAPTERS: list[Chapter] = [
    Chapter(
        chapter_id="c10_real_numbers",
        title="Real Numbers",
        unit="Number Systems",
        class_level=10,
        marks=6,
        subtopics=[
            "Euclid's Division Lemma and Algorithm",
            "Fundamental Theorem of Arithmetic",
            "Irrational numbers and their decimal expansions",
            "Rational numbers and their decimal expansions",
        ],
        problem_types=["mcq_1m", "short_answer_2m", "long_answer_3m"],
        keywords=["HCF", "LCM", "irrational", "prime factorisation", "Euclid"],
    ),
    Chapter(
        chapter_id="c10_polynomials",
        title="Polynomials",
        unit="Algebra",
        class_level=10,
        marks=8,
        subtopics=[
            "Zeros of a polynomial; geometrical meaning",
            "Relationship between zeros and coefficients (quadratic, cubic)",
            "Division algorithm for polynomials",
        ],
        problem_types=["mcq_1m", "short_answer_2m", "long_answer_3m"],
        keywords=["polynomial", "zeros", "roots", "quadratic", "cubic", "remainder theorem"],
    ),
    Chapter(
        chapter_id="c10_pair_linear_equations",
        title="Pair of Linear Equations in Two Variables",
        unit="Algebra",
        class_level=10,
        marks=8,
        subtopics=[
            "Graphical method; consistency",
            "Algebraic methods: substitution, elimination, cross-multiplication",
            "Equations reducible to pair of linear equations",
        ],
        problem_types=["mcq_1m", "short_answer_2m", "long_answer_3m", "case_study"],
        keywords=["linear equations", "substitution", "elimination", "consistent", "inconsistent"],
    ),
    Chapter(
        chapter_id="c10_quadratic_equations",
        title="Quadratic Equations",
        unit="Algebra",
        class_level=10,
        marks=8,
        subtopics=[
            "Standard form; solution by factorisation and completing the square",
            "Quadratic formula",
            "Nature of roots: discriminant",
            "Real-world problems leading to quadratic equations",
        ],
        problem_types=["mcq_1m", "short_answer_2m", "long_answer_3m", "case_study"],
        keywords=["quadratic", "discriminant", "factorisation", "completing the square", "roots", "quadratic formula"],
    ),
    Chapter(
        chapter_id="c10_ap",
        title="Arithmetic Progressions",
        unit="Algebra",
        class_level=10,
        marks=8,
        subtopics=[
            "Introduction and motivation",
            "nth term of an AP",
            "Sum of first n terms of an AP",
            "Applications in real-life problems",
        ],
        problem_types=["mcq_1m", "short_answer_2m", "long_answer_3m"],
        keywords=["AP", "arithmetic progression", "common difference", "nth term", "sum of AP"],
    ),
    Chapter(
        chapter_id="c10_triangles",
        title="Triangles",
        unit="Geometry",
        class_level=10,
        marks=10,
        subtopics=[
            "Similar figures; similar triangles",
            "Basic Proportionality Theorem (Thales)",
            "AAA, SSS, SAS similarity criteria",
            "Areas of similar triangles",
            "Pythagoras theorem and converse",
        ],
        problem_types=["mcq_1m", "short_answer_2m", "long_answer_4m", "case_study"],
        keywords=["similar triangles", "Thales", "BPT", "Pythagoras", "similarity criteria", "area ratio"],
    ),
    Chapter(
        chapter_id="c10_coordinate_geometry",
        title="Coordinate Geometry",
        unit="Coordinate Geometry",
        class_level=10,
        marks=8,
        subtopics=[
            "Distance formula",
            "Section formula (internal, external)",
            "Area of triangle using coordinates",
        ],
        problem_types=["mcq_1m", "short_answer_2m", "long_answer_3m", "case_study"],
        keywords=["distance formula", "section formula", "mid-point", "collinear", "area of triangle"],
    ),
    Chapter(
        chapter_id="c10_trig_intro",
        title="Introduction to Trigonometry",
        unit="Trigonometry",
        class_level=10,
        marks=12,
        subtopics=[
            "Trigonometric ratios of an acute angle",
            "Trigonometric ratios of specific angles (0°, 30°, 45°, 60°, 90°)",
            "Trigonometric ratios of complementary angles",
            "Trigonometric identities",
        ],
        problem_types=["mcq_1m", "short_answer_2m", "long_answer_3m"],
        keywords=["sin", "cos", "tan", "cosec", "sec", "cot", "trigonometric identity", "complementary angles"],
    ),
    Chapter(
        chapter_id="c10_height_distance",
        title="Some Applications of Trigonometry",
        unit="Trigonometry",
        class_level=10,
        marks=12,
        subtopics=[
            "Heights and distances: angle of elevation and depression",
            "Problems involving two triangles",
        ],
        problem_types=["long_answer_3m", "case_study"],
        keywords=["angle of elevation", "angle of depression", "height", "distance", "ladder", "tower"],
    ),
    Chapter(
        chapter_id="c10_circles",
        title="Circles",
        unit="Geometry",
        class_level=10,
        marks=10,
        subtopics=[
            "Tangent to a circle; tangent from an external point",
            "Number of tangents from a point",
        ],
        problem_types=["mcq_1m", "short_answer_2m", "long_answer_4m"],
        keywords=["tangent", "radius", "chord", "external point", "point of tangency"],
    ),
    Chapter(
        chapter_id="c10_areas_circles",
        title="Areas Related to Circles",
        unit="Mensuration",
        class_level=10,
        marks=10,
        subtopics=[
            "Area of sector and segment",
            "Perimeter and area of combinations of plane figures",
        ],
        problem_types=["short_answer_2m", "long_answer_3m", "case_study"],
        keywords=["sector", "segment", "arc length", "area of circle", "perimeter"],
    ),
    Chapter(
        chapter_id="c10_surface_volumes",
        title="Surface Areas and Volumes",
        unit="Mensuration",
        class_level=10,
        marks=10,
        subtopics=[
            "Surface areas and volumes of combinations of solids",
            "Conversion of solids (cone, sphere, cylinder)",
            "Frustum of a cone",
        ],
        problem_types=["long_answer_3m", "long_answer_4m", "case_study"],
        keywords=["surface area", "volume", "cylinder", "cone", "sphere", "frustum", "hemisphere"],
    ),
    Chapter(
        chapter_id="c10_statistics",
        title="Statistics",
        unit="Statistics and Probability",
        class_level=10,
        marks=8,
        subtopics=[
            "Mean, median, mode of grouped data",
            "Cumulative frequency — ogive",
        ],
        problem_types=["short_answer_2m", "long_answer_3m"],
        keywords=["mean", "median", "mode", "grouped data", "frequency distribution", "ogive"],
    ),
    Chapter(
        chapter_id="c10_probability",
        title="Probability",
        unit="Statistics and Probability",
        class_level=10,
        marks=8,
        subtopics=[
            "Classical probability",
            "Simple problems on single events",
        ],
        problem_types=["mcq_1m", "short_answer_2m", "long_answer_3m"],
        keywords=["probability", "event", "sample space", "equally likely", "favourable outcomes"],
    ),
]

# ─────────────────────────────────────────────────────────────────────────────
# Convenience registry
# ─────────────────────────────────────────────────────────────────────────────

ALL_CHAPTERS: dict[str, Chapter] = {
    ch.chapter_id: ch
    for ch in CLASS_12_CHAPTERS + CLASS_10_CHAPTERS
}


def chapters_for_class(class_level: ClassLevel) -> list[Chapter]:
    """Return all chapters for a given class level."""
    return [ch for ch in ALL_CHAPTERS.values() if ch.class_level == class_level]


def find_chapters_by_keyword(text: str) -> list[Chapter]:
    """
    Given a block of text (e.g. a problem statement), return chapters whose
    keywords appear in the text.  Useful for tagging generated problems.
    """
    text_lower = text.lower()
    matched = []
    for ch in ALL_CHAPTERS.values():
        for kw in ch.keywords:
            if kw.lower() in text_lower:
                matched.append(ch)
                break
    return matched


# ─────────────────────────────────────────────────────────────────────────────
# Problem type descriptions (for prompt construction)
# ─────────────────────────────────────────────────────────────────────────────

PROBLEM_TYPE_DESCRIPTIONS: dict[str, str] = {
    "mcq_1m":         "1-mark Multiple Choice Question (4 options, one correct)",
    "short_answer_2m": "2-mark short-answer question requiring a brief solution",
    "long_answer_3m":  "3-mark question requiring a structured multi-step solution",
    "long_answer_4m":  "4-mark question requiring a detailed solution",
    "long_answer_5m":  "5-mark question requiring a complete & detailed solution",
    "case_study":      "4–5 mark case-study / application-based question with 4–5 sub-parts",
}

DIFFICULTY_LEVELS = ["easy", "medium", "hard"]

# CBSE marks distribution hint (approximate, Class 12, 2024–25)
MARKS_DISTRIBUTION_CLASS_12 = {
    "mcq_1m":          18,   # Section A: 18 × 1 = 18
    "assertion_reason": 2,   # Section A: 2 assertion-reason = 2
    "short_answer_2m": 10,   # Section B: 5 × 2 = 10
    "long_answer_3m":  18,   # Section C: 6 × 3 = 18
    "long_answer_5m":  20,   # Section D: 4 × 5 = 20
    "case_study":      12,   # Section E: 3 × 4 = 12
}
