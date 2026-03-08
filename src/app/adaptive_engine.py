from __future__ import annotations


class AdaptiveEngine:
    """
    Clase que adapta el nivel del siguiente ejercicio segun desempeno reciente.
    """

    level_map = {1: "muy_facil", 2: "facil", 3: "medio", 4: "dificil", 5: "muy_dificil"}

    def __init__(self, initial_level: int = 3) -> None:
        self.current_level = max(1, min(5, initial_level))

    def recommend_next_level(
        self,
        errors: int,
        time_spent: float,
        average_time: float,
        correct_first_attempt: int,
        predicted_difficulty: str,
        predicted_probability: float,
    ) -> dict[str, str | int]:
        """
        Ajuste hibrido:
        - reglas de desempeno del jugador
        - senal IA (dificultad predicha + confianza)
        """
        level_delta = 0
        reasons: list[str] = []

        # Reglas basadas en desempeno observable.
        if errors > 2:
            level_delta -= 1
            reasons.append("muchos errores")
        elif correct_first_attempt == 1 and errors == 0 and time_spent < average_time:
            level_delta += 1
            reasons.append("resolucion rapida y correcta")

        # Reglas IA-driven (solo con confianza suficiente).
        label = predicted_difficulty.strip().lower()
        conf = float(predicted_probability)
        if conf >= 0.75:
            if label == "alta":
                level_delta -= 1
                reasons.append("IA detecta alta dificultad")
            elif label == "baja":
                level_delta += 1
                reasons.append("IA detecta baja dificultad")
        elif conf >= 0.60 and label == "alta" and correct_first_attempt == 0:
            level_delta -= 1
            reasons.append("IA sugiere reforzar base")

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
        }
