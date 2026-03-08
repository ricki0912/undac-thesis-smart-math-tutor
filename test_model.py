from __future__ import annotations

from src.app.student_simulator import StudentSimulator
from src.models.difficulty_model import DifficultyModel
from src.utils.config import ProjectConfig


def main() -> None:
    config = ProjectConfig()
    simulator = StudentSimulator(seed=7)
    model = DifficultyModel(config)
    model.load()

    sample = simulator.simulate_step(student_id="test_student", level=3)
    prediction = model.predict(sample)

    print("Dato simulado:")
    print(sample)
    print("\nPrediccion del modelo:")
    print(prediction)


if __name__ == "__main__":
    main()
