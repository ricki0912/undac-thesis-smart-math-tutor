from __future__ import annotations

import random
from dataclasses import dataclass


@dataclass(frozen=True)
class Question:
    text: str
    answer: int
    hint: str
    level: int


class StudentSimulator:
    """
    Banco de preguntas (>=100) con respuestas enteras, orientado a primaria.
    """

    def __init__(self, seed: int = 42) -> None:
        self.rng = random.Random(seed)
        self.exercise_bank = self._build_question_bank()
        self._validate_integer_bank()

    def _build_question_bank(self) -> dict[int, list[Question]]:
        bank: dict[int, list[Question]] = {1: [], 2: [], 3: [], 4: [], 5: []}

        # Nivel 1: x + b = c
        for ans in range(1, 21):
            b = (ans % 5) + 1
            c = ans + b
            bank[1].append(
                Question(
                    text=f"x + {b} = {c}",
                    answer=ans,
                    hint=f"Resta {b} en ambos lados.",
                    level=1,
                )
            )

        # Nivel 2: ax + b = c
        for ans in range(1, 21):
            a = (ans % 3) + 2  # 2..4
            b = (ans % 4) + 1  # 1..5
            c = a * ans + b
            bank[2].append(
                Question(
                    text=f"{a}x + {b} = {c}",
                    answer=ans,
                    hint=f"Resta {b} y luego divide entre {a}.",
                    level=2,
                )
            )

        # Nivel 3: ax - b = c
        for ans in range(1, 21):
            a = (ans % 4) + 2  # 2..5
            b = (ans % 5) + 2  # 2..6
            c = a * ans - b
            bank[3].append(
                Question(
                    text=f"{a}x - {b} = {c}",
                    answer=ans,
                    hint=f"Suma {b} y divide entre {a}.",
                    level=3,
                )
            )

        # Nivel 4: a(x + b) = c
        for ans in range(1, 21):
            a = (ans % 3) + 2  # 2..4
            b = (ans % 4) + 1  # 1..5
            c = a * (ans + b)
            bank[4].append(
                Question(
                    text=f"{a}(x + {b}) = {c}",
                    answer=ans,
                    hint=f"Divide entre {a} y luego resta {b}.",
                    level=4,
                )
            )

        # Nivel 5: ax + b = cx + d
        for ans in range(1, 21):
            c = (ans % 3) + 1  # 1..3
            a = c + ((ans % 2) + 2)  # a>c para evitar cero
            b = (ans % 5) + 1
            d = (a - c) * ans + b
            bank[5].append(
                Question(
                    text=f"{a}x + {b} = {c}x + {d}",
                    answer=ans,
                    hint="Pasa las x a un lado y los numeros al otro lado.",
                    level=5,
                )
            )

        return bank

    def _validate_integer_bank(self) -> None:
        """
        Garantiza que el banco use solo respuestas enteras y ecuaciones sin decimales.
        """
        for level, questions in self.exercise_bank.items():
            for question in questions:
                if not isinstance(question.answer, int):
                    raise ValueError(
                        f"Pregunta invalida en nivel {level}: respuesta no entera -> {question.text}"
                    )
                if "." in question.text or "/" in question.text:
                    raise ValueError(
                        f"Pregunta invalida en nivel {level}: contiene decimal/fraccion -> {question.text}"
                    )

    def total_questions(self) -> int:
        return sum(len(v) for v in self.exercise_bank.values())

    def get_question_for_level(self, level: int) -> Question:
        safe_level = max(1, min(5, level))
        return self.rng.choice(self.exercise_bank[safe_level])

    def simulate_step(self, student_id: str = "sim_student", level: int = 3) -> dict[str, float | int | str]:
        q = self.get_question_for_level(level)
        is_correct = self.rng.random() > 0.4
        attempt_number = 1 if is_correct else self.rng.randint(1, 3)
        hints = self.rng.randint(0, 2)
        duration = round(self.rng.uniform(25, 140), 2)
        if is_correct:
            incorrects = attempt_number - 1
            cfa = 1 if attempt_number == 1 else 0
        else:
            incorrects = attempt_number
            cfa = 0
        return {
            "student_id": student_id,
            "step_name": q.text,
            "incorrects": incorrects,
            "hints": hints,
            "correct_first_attempt": cfa,
            "step_duration_sec": duration,
            "level": level,
        }

    @staticmethod
    def is_correct_answer(user_answer: str, expected: int) -> bool:
        try:
            value = int(str(user_answer).strip())
        except ValueError:
            return False
        return value == expected
