from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class AssessmentQuestion:
    prompt: str
    answer: float
    level: int


class AssessmentEngine:
    """
    Gestiona preguntas de pre-test y post-test para estimar progreso real.
    """

    def __init__(self) -> None:
        self.questions = [
            AssessmentQuestion(prompt="x + 3 = 9", answer=6, level=1),
            AssessmentQuestion(prompt="2x + 5 = 17", answer=6, level=2),
            AssessmentQuestion(prompt="3x - 4 = 11", answer=5, level=2),
            AssessmentQuestion(prompt="4x + 2 = 3x + 10", answer=8, level=3),
            AssessmentQuestion(prompt="2(x + 3) = 18", answer=6, level=3),
            AssessmentQuestion(prompt="(x - 2)/3 = 4", answer=14, level=4),
            AssessmentQuestion(prompt="5 - 2x = -9", answer=7, level=4),
            AssessmentQuestion(prompt="3(x - 1) + 2 = 2x + 9", answer=10, level=5),
        ]

    def get_pretest(self) -> list[AssessmentQuestion]:
        return self.questions[:5]

    def get_posttest(self) -> list[AssessmentQuestion]:
        return self.questions[3:]

    @staticmethod
    def is_correct(user_answer: str, expected: float, tolerance: float = 1e-6) -> bool:
        try:
            value = float(str(user_answer).replace(",", "."))
        except ValueError:
            return False
        return abs(value - expected) <= tolerance

    @staticmethod
    def score_to_level(score: int, total: int) -> int:
        if total <= 0:
            return 3
        ratio = score / total
        if ratio < 0.35:
            return 1
        if ratio < 0.55:
            return 2
        if ratio < 0.75:
            return 3
        if ratio < 0.9:
            return 4
        return 5

