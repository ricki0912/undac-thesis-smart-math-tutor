from __future__ import annotations

from typing import Any

class AdaptiveEngine:
    """
    Clase que adapta el nivel del siguiente ejercicio segun desempeno reciente.
    """

    level_map = {1: "muy_facil", 2: "facil", 3: "medio", 4: "dificil", 5: "muy_dificil"}

    def __init__(self, initial_level: int = 3) -> None:
        self.current_level = max(1, min(5, initial_level))

    @staticmethod
    def performance_score(
        is_correct: int,
        incorrects: int,
        hints: int,
        time_spent: float,
        time_reference: float,
    ) -> float:
        """
        Score 0..1 (mayor es mejor) basado en desempeno observable post-intento.

        Nota: si quieres un comportamiento mas estricto o mas tolerante, ajusta pesos y escalas.
        """
        if int(is_correct) != 1:
            return 0.0

        ref = max(1.0, float(time_reference))
        time_penalty = min(max(float(time_spent), 0.0) / ref, 1.0)
        error_penalty = min(max(int(incorrects), 0) / 3.0, 1.0)
        hint_penalty = min(max(int(hints), 0) / 2.0, 1.0)

        score = 1.0 - ((0.45 * time_penalty) + (0.35 * error_penalty) + (0.20 * hint_penalty))
        return float(max(0.0, min(1.0, score)))

    def _policy_label(
        self,
        is_correct: int,
        incorrects: int,
        hints: int,
        time_spent: float,
        average_time: float,
    ) -> tuple[str, list[str], float]:
        score = self.performance_score(
            is_correct=is_correct,
            incorrects=incorrects,
            hints=hints,
            time_spent=time_spent,
            time_reference=average_time,
        )
        reasons: list[str] = []

        if int(is_correct) != 1:
            reasons.append("respuesta incorrecta")
            return "bajar", reasons, score

        if score >= 0.75 and int(hints) == 0 and int(incorrects) == 0:
            reasons.append("correcto sin errores/pistas")
            return "subir", reasons, score

        if score < 0.45 or int(hints) >= 2:
            if score < 0.45:
                reasons.append("desempeno bajo (score<0.45)")
            if int(hints) >= 2:
                reasons.append("muchas pistas")
            return "bajar", reasons, score

        if int(hints) > 0 or int(incorrects) > 0:
            reasons.append("correcto con apoyo (refuerzo)")
            return "reforzar", reasons, score

        reasons.append("correcto estable")
        return "mantener", reasons, score

    def recommend_next_level(
        self,
        incorrects: int,
        hints: int,
        time_spent: float,
        average_time: float,
        is_correct: int,
        recent_attempts: list[dict[str, Any]] | None = None,
    ) -> dict[str, object]:
        """
        Politica post-intento (reglas):
        - calcula un score de desempeno (0..1)
        - usa voto en ventana corta (anti-oscilacion) con los ultimos intentos
        """
        current_label, reasons, score = self._policy_label(
            is_correct=is_correct,
            incorrects=incorrects,
            hints=hints,
            time_spent=time_spent,
            average_time=average_time,
        )

        window = list(recent_attempts or [])[-2:]
        window.append(
            {
                "is_correct": int(is_correct),
                "incorrects": int(incorrects),
                "hints": int(hints),
                "time_spent": float(time_spent),
            }
        )

        votes = {"subir": 0, "bajar": 0}
        for item in window:
            label, _, _ = self._policy_label(
                is_correct=int(item.get("is_correct", 0)),
                incorrects=int(item.get("incorrects", 0)),
                hints=int(item.get("hints", 0)),
                time_spent=float(item.get("time_spent", 0.0)),
                average_time=float(average_time),
            )
            if label in votes:
                votes[label] += 1

        level_delta = 0
        required_votes = 2
        if len(window) <= 1:
            required_votes = 1
        elif len(window) == 2:
            required_votes = 2

        if votes["bajar"] >= required_votes:
            level_delta = -1
            reasons.append(f"ventana: {votes['bajar']}/{len(window)} sugiere bajar")
        elif votes["subir"] >= required_votes and votes["bajar"] == 0:
            level_delta = 1
            reasons.append(f"ventana: {votes['subir']}/{len(window)} sugiere subir")
        elif current_label == "reforzar":
            reasons.append("ventana: refuerzo sin cambiar nivel")

        new_level = max(1, min(5, self.current_level + level_delta))
        action = "mantener"
        if new_level > self.current_level:
            action = "subir"
        elif new_level < self.current_level:
            action = "bajar"
        self.current_level = new_level

        return {
            "action": action,
            "next_level": self.current_level,
            "level_name": self.level_map[self.current_level],
            "reason": ", ".join(reasons) if reasons else "sin cambios por reglas/IA",
            "policy_label": current_label,
            "performance_score": float(score),
        }
