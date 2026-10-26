
"""Persistent JSON storage for research conversations."""

import json
from datetime import datetime, timezone
from pathlib import Path
from threading import RLock
from uuid import uuid4

from app.core.config import PROJECT_ROOT


CONVERSATIONS_DIR = PROJECT_ROOT / "data" / "conversations"
CONVERSATIONS_FILE = CONVERSATIONS_DIR / "conversations.json"


class ConversationStore:
    """Create conversations and persist their message histories."""

    def __init__(self, file_path: Path | None = None):
        self.file_path = Path(file_path or CONVERSATIONS_FILE)
        self.lock = RLock()
        self.file_path.parent.mkdir(parents=True, exist_ok=True)

        if not self.file_path.exists():
            self._write({"conversations": {}})

    @staticmethod
    def _now() -> str:
        return datetime.now(timezone.utc).isoformat()

    def _read(self) -> dict:
        if not self.file_path.exists():
            return {"conversations": {}}

        with self.file_path.open("r", encoding="utf-8") as file:
            data = json.load(file)

        if not isinstance(data, dict):
            raise ValueError("Invalid conversation storage format.")

        conversations = data.get("conversations")

        if not isinstance(conversations, dict):
            raise ValueError("Conversation registry must be an object.")

        return data

    def _write(self, data: dict) -> None:
        self.file_path.parent.mkdir(parents=True, exist_ok=True)

        temporary_file = self.file_path.with_suffix(".tmp")

        with temporary_file.open("w", encoding="utf-8") as file:
            json.dump(data, file, ensure_ascii=False, indent=2)

        temporary_file.replace(self.file_path)

    def create_conversation(self, title: str = "New conversation") -> dict:
        title = title.strip() or "New conversation"
        now = self._now()
        conversation_id = str(uuid4())

        with self.lock:
            data = self._read()

            conversation = {
                "conversation_id": conversation_id,
                "title": title[:200],
                "created_at": now,
                "updated_at": now,
                "messages": [],
            }

            data["conversations"][conversation_id] = conversation
            self._write(data)

        return conversation.copy()

    def list_conversations(self) -> list[dict]:
        with self.lock:
            data = self._read()

            conversations = [
                {
                    "conversation_id": item["conversation_id"],
                    "title": item["title"],
                    "created_at": item["created_at"],
                    "updated_at": item["updated_at"],
                    "message_count": len(item["messages"]),
                }
                for item in data["conversations"].values()
            ]

        return sorted(
            conversations,
            key=lambda item: item["updated_at"],
            reverse=True,
        )

    def get_conversation(self, conversation_id: str) -> dict | None:
        with self.lock:
            data = self._read()
            conversation = data["conversations"].get(conversation_id)

            if conversation is None:
                return None

            # Return a detached copy so callers cannot mutate stored data.
            return json.loads(json.dumps(conversation))

    def add_message(
        self,
        conversation_id: str,
        role: str,
        content: str,
        sources: list[dict] | None = None,
    ) -> dict:
        if role not in {"user", "assistant"}:
            raise ValueError("Role must be 'user' or 'assistant'.")

        content = content.strip()

        if not content:
            raise ValueError("Message content cannot be empty.")

        message = {
            "message_id": str(uuid4()),
            "role": role,
            "content": content,
            "created_at": self._now(),
            "sources": sources or [],
        }

        with self.lock:
            data = self._read()
            conversation = data["conversations"].get(conversation_id)

            if conversation is None:
                raise KeyError("Conversation not found.")

            conversation["messages"].append(message)
            conversation["updated_at"] = message["created_at"]

            # Use the first user message as the default title.
            if role == "user" and not conversation["messages"][:-1]:
                conversation["title"] = content[:80]

            self._write(data)

        return message.copy()
