from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import pandas as pd
from pandas.errors import ParserError

from src.utils.config import ProjectConfig


class AuthStore:
    """
    Maneja registro/login simple y persistencia de progreso por usuario.
    """

    def __init__(self, config: ProjectConfig) -> None:
        self.config = config
        self.config.logs_dir.mkdir(parents=True, exist_ok=True)

    def register_user(self, username: str, password: str, role: str = "player") -> tuple[bool, str]:
        clean_user = username.strip().lower()
        if not clean_user or len(password) < 4:
            return False, "Usuario invalido o contrasena muy corta."
        if role not in {"player", "admin"}:
            return False, "Rol invalido."

        users = self._load_users()
        if not users.empty and clean_user in users["username"].astype(str).tolist():
            return False, "El usuario ya existe."

        users = pd.concat(
            [
                users,
                pd.DataFrame(
                    [
                        {
                            "username": clean_user,
                            "password_hash": self._hash_password(password),
                            "role": role,
                        }
                    ]
                ),
            ],
            ignore_index=True,
        )
        self._save_users(users)
        return True, "Usuario registrado correctamente."

    def login(self, username: str, password: str) -> tuple[bool, str, str]:
        clean_user = username.strip().lower()
        users = self._load_users()
        if users.empty:
            return False, "No hay usuarios registrados. Crea una cuenta primero.", "player"

        target = users[users["username"].astype(str) == clean_user]
        if target.empty:
            return False, "Usuario no encontrado.", "player"

        expected_hash = str(target.iloc[0]["password_hash"])
        if expected_hash != self._hash_password(password):
            return False, "Contrasena incorrecta.", "player"
        role = str(target.iloc[0].get("role", "player"))
        if role not in {"player", "admin"}:
            role = "player"
        return True, "Login exitoso.", role

    def load_progress(self, username: str) -> dict[str, Any] | None:
        path = self.config.user_progress_path
        if not path.exists():
            return None
        with path.open("r", encoding="utf-8") as f:
            payload = json.load(f)
        return payload.get(username.strip().lower())

    def save_progress(self, username: str, progress: dict[str, Any]) -> None:
        path = self.config.user_progress_path
        data: dict[str, Any] = {}
        if path.exists():
            with path.open("r", encoding="utf-8") as f:
                data = json.load(f)
        data[username.strip().lower()] = progress
        with path.open("w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

    def _load_users(self) -> pd.DataFrame:
        if not self.config.users_path.exists():
            return pd.DataFrame(columns=["username", "password_hash", "role"])
        try:
            users = pd.read_csv(self.config.users_path)
            if "role" not in users.columns:
                users["role"] = "player"
            users = users[["username", "password_hash", "role"]].copy()
        except ParserError:
            users = self._recover_users_file()

        users["username"] = users["username"].astype(str).str.strip().str.lower()
        users["password_hash"] = users["password_hash"].astype(str).str.strip()
        users["role"] = users["role"].astype(str).str.strip().str.lower()
        users.loc[~users["role"].isin(["player", "admin"]), "role"] = "player"
        users = users.dropna(subset=["username", "password_hash"])
        users = users[users["username"] != ""]
        users = users.drop_duplicates(subset=["username"], keep="last")
        self._save_users(users)
        return users

    def _recover_users_file(self) -> pd.DataFrame:
        rows: list[dict[str, str]] = []
        with self.config.users_path.open("r", encoding="utf-8") as f:
            lines = [line.strip() for line in f if line.strip()]

        if not lines:
            return pd.DataFrame(columns=["username", "password_hash", "role"])

        for raw_line in lines[1:]:
            parts = [part.strip() for part in raw_line.split(",", 2)]
            if len(parts) == 2:
                username, password_hash = parts
                role = "player"
            else:
                username, password_hash, role = parts[0], parts[1], parts[2]
            rows.append(
                {
                    "username": username,
                    "password_hash": password_hash,
                    "role": role or "player",
                }
            )
        return pd.DataFrame(rows, columns=["username", "password_hash", "role"])

    def _save_users(self, users: pd.DataFrame) -> None:
        self.config.logs_dir.mkdir(parents=True, exist_ok=True)
        stable = users[["username", "password_hash", "role"]].copy()
        stable.to_csv(self.config.users_path, index=False)

    @staticmethod
    def _hash_password(password: str) -> str:
        return hashlib.sha256(password.encode("utf-8")).hexdigest()
