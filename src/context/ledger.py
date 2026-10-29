from __future__ import annotations

from dataclasses import asdict, dataclass, field


@dataclass
class ContextItem:
    item_id: str
    episode_id: str
    step: int
    content: str
    source: str
    tags: list[str]
    immutable: bool = True
    active: bool = True
    archived: bool = False
    parent_item_ids: list[str] = field(default_factory=list)


@dataclass
class ManagementDecision:
    decision_id: str
    episode_id: str
    step: int
    operation: str
    input_item_ids: list[str]
    output_item_id: str | None
    output_content: str | None
    rationale: str
    active_tokens_before: int
    active_tokens_after: int


class ContextLedger:
    def __init__(self, episode_id: str) -> None:
        self.episode_id = episode_id
        self.items: dict[str, ContextItem] = {}
        self.decisions: list[ManagementDecision] = []
        self._item_counter = 0
        self._view_counter = 0
        self._decision_counter = 0

    def add_raw(self, step: int, content: str, tags: list[str], source: str = "task_environment") -> ContextItem:
        self._item_counter += 1
        item = ContextItem(
            item_id=f"ctx_{self._item_counter:04d}",
            episode_id=self.episode_id,
            step=step,
            content=content,
            source=source,
            tags=tags,
        )
        self.items[item.item_id] = item
        return item

    def active_items(self) -> list[ContextItem]:
        return sorted([item for item in self.items.values() if item.active], key=lambda item: item.step)

    def active_context(self) -> str:
        lines = []
        for item in self.active_items():
            lines.append(f"[{item.item_id}] {item.content}")
        return "\n".join(lines)

    def token_estimate(self) -> int:
        # Simple reproducible approximation for pilot reporting.
        return max(1, len(self.active_context().split()))

    def record_decision(
        self,
        step: int,
        operation: str,
        input_item_ids: list[str],
        output_item_id: str | None,
        output_content: str | None,
        rationale: str,
        tokens_before: int,
    ) -> ManagementDecision:
        self._decision_counter += 1
        decision = ManagementDecision(
            decision_id=f"dec_{self._decision_counter:04d}",
            episode_id=self.episode_id,
            step=step,
            operation=operation,
            input_item_ids=input_item_ids,
            output_item_id=output_item_id,
            output_content=output_content,
            rationale=rationale,
            active_tokens_before=tokens_before,
            active_tokens_after=self.token_estimate(),
        )
        self.decisions.append(decision)
        return decision

    def compress_item(self, item_id: str, output_content: str, rationale: str, step: int) -> ManagementDecision:
        tokens_before = self.token_estimate()
        original = self.items[item_id]
        original.active = False
        original.archived = True
        self._view_counter += 1
        view_id = f"ctx_view_{self._view_counter:04d}"
        view = ContextItem(
            item_id=view_id,
            episode_id=self.episode_id,
            step=original.step,
            content=output_content,
            source="context_manager",
            tags=[tag for tag in original.tags if tag != "critical"],
            immutable=False,
            active=True,
            archived=False,
            parent_item_ids=[item_id],
        )
        self.items[view_id] = view
        return self.record_decision(step, "COMPRESS", [item_id], view_id, output_content, rationale, tokens_before)

    def archive_item(self, item_id: str, rationale: str, step: int) -> ManagementDecision:
        tokens_before = self.token_estimate()
        self.items[item_id].active = False
        self.items[item_id].archived = True
        return self.record_decision(step, "ARCHIVE", [item_id], None, None, rationale, tokens_before)

    def keep_item(self, item_id: str, rationale: str, step: int) -> ManagementDecision:
        tokens_before = self.token_estimate()
        self.items[item_id].active = True
        return self.record_decision(step, "KEEP", [item_id], item_id, self.items[item_id].content, rationale, tokens_before)

    def restore_original(self, compressed_item_id: str, rationale: str, step: int) -> ManagementDecision | None:
        compressed = self.items[compressed_item_id]
        if not compressed.parent_item_ids:
            return None
        original_id = compressed.parent_item_ids[0]
        tokens_before = self.token_estimate()
        compressed.active = False
        compressed.archived = True
        original = self.items[original_id]
        original.active = True
        original.archived = False
        return self.record_decision(step, "RESTORE", [compressed_item_id], original_id, original.content, rationale, tokens_before)

    def to_dict(self) -> dict:
        return {
            "episode_id": self.episode_id,
            "items": [asdict(item) for item in self.items.values()],
            "decisions": [asdict(decision) for decision in self.decisions],
            "active_context": self.active_context(),
        }

