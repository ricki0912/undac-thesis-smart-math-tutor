from __future__ import annotations

import csv
import random
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Question:
    text: str
    answer: int
    hint: str
    level: int


class StudentSimulator:
    """
    Banco de preguntas cargado desde CSV para facilitar edicion externa.
    """

    def __init__(
        self,
        seed: int = 42,
        question_bank_path: str | Path = "data/raw/question_bank.csv",
    ) -> None:
        self.rng = random.Random(seed)
        self.question_bank_path = Path(question_bank_path)
        self.exercise_bank = self._load_question_bank_from_csv(self.question_bank_path)
        self._validate_integer_bank()

    def _load_question_bank_from_csv(self, path: Path) -> dict[int, list[Question]]:
        if not path.exists():
            raise FileNotFoundError(
                f"No se encontro el banco de preguntas en {path}. "
                "Crea el archivo CSV con columnas: level,text,answer,hint"
            )

        bank: dict[int, list[Question]] = {1: [], 2: [], 3: [], 4: [], 5: []}
        with path.open("r", encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f)
            required = {"level", "text", "answer", "hint"}
            if not reader.fieldnames or not required.issubset(set(reader.fieldnames)):
                raise ValueError(
                    "El CSV de preguntas debe incluir columnas: level,text,answer,hint"
                )

            for idx, row in enumerate(reader, start=2):
                try:
                    level = int(str(row["level"]).strip())
                    text = str(row["text"]).strip()
                    answer = int(str(row["answer"]).strip())
                    hint = str(row["hint"]).strip()
                except Exception as exc:
                    raise ValueError(f"Fila invalida en {path}:{idx}") from exc

                if level not in bank:
                    raise ValueError(f"Nivel fuera de rango en {path}:{idx} -> {level}")
                if not text:
                    raise ValueError(f"Texto vacio en {path}:{idx}")

                bank[level].append(
                    Question(
                        text=text,
                        answer=answer,
                        hint=hint if hint else "Piensa paso a paso.",
                        level=level,
                    )
                )

        total = sum(len(v) for v in bank.values())
        if total == 0:
            raise ValueError("El banco de preguntas esta vacio.")
        return bank

    def _validate_integer_bank(self) -> None:
        """
        Garantiza respuestas enteras y evita preguntas con decimal/fraccion.
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
        if not self.exercise_bank[safe_level]:
            raise ValueError(f"No hay preguntas en el nivel {safe_level}")
        return self.rng.choice(self.exercise_bank[safe_level])

    def simulate_step(
        self,
        student_id: str = "sim_student",
        level: int = 3,
    ) -> dict[str, float | int | str]:
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

