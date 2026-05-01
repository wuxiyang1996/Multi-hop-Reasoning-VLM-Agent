"""Deterministic ``<state>...</state>`` producer for live gymnasium envs
(GamingAgent ``make_gaming_env(...)`` family).

The cold-start labeler writes ``schema_canonical`` blocks by prompting a
VLM on rendered frames; for **runtime** evaluation we have direct access
to the env's underlying state via ``env.step(...)``'s ``info`` dict (and
the textual obs). This module turns that structured info dict into the
same on-disk ``<state>`` block shape so:

  * ``parse_schema_canonical`` (already extended to parse
    ``<attributes>`` / ``<state_flags>`` in Day-3) round-trips cleanly,
  * ``harness.gymv_success.evaluate_predicate`` can decide
    ``entity_value_increased``, ``entity_count_changed``,
    ``entity_appeared``, ``entity_disappeared``, and
    ``cumulative_reward_increased`` against numeric facts rather than
    bottom out at "entity_attrs missing on both sides" (the Day-3
    failure mode for ``attribute_changed`` on text-obs envs).

Producers are pure: ``producer(info, obs, *, step, task, goal) -> str``.
No env reference is held; no side effects. This keeps the unit tests
hermetic and lets ``make_gymv_executor`` plug in any third-party
producer (e.g. a real VLM head later) without code changes.

Shape contract: every producer's output is parseable by
``labeling_supplement._harness_io_helpers.parse_schema_canonical`` and
populates ``StateSchema.facts`` with at least:

  * ``score`` (float / int) — the per-step cumulative env reward
  * ``phase`` (str) — ``play``, ``gameover``, …
  * ``entity_attrs`` (dict, label → field → value) and
    ``entity_label_count`` (dict, label → count)

Hot-path scalars promoted into ``state.facts`` for cheap evaluator
access: ``score``, ``highest_tile``, ``lines_cleared``, ``tetris_score``,
``moves_remaining``.

Cross-refs:
  * harness/README.md §22 (Day-4 status block)
  * labeling_supplement/harness_io_out/_phase3_report.md
  * harness/gymv_executor.py::_state_from_env_obs (the consumer)
"""
from __future__ import annotations

from typing import Any, Callable, List, Mapping, Optional, Sequence, Tuple

# Public type alias. Importable from ``harness`` directly.
SchemaProducer = Callable[..., str]


__all__ = [
    "SchemaProducer",
    "make_gaming_env_producer",
    "render_state_block",
    "twenty_forty_eight_producer",
    "tetris_producer",
]


# ---------------------------------------------------------------------------
# Shared rendering helpers
# ---------------------------------------------------------------------------


def _coerce_number(v: Any, default: float = 0.0) -> float:
    """Best-effort cast to ``float``. Survives ``np.int64`` / ``np.float_``
    / strings; returns ``default`` on failure (so the producer never
    raises just because the env emitted a weird scalar)."""
    try:
        if v is None:
            return float(default)
        return float(v)
    except (TypeError, ValueError):
        return float(default)


def _flatten_board_2d(board: Any) -> Optional[List[List[int]]]:
    """Coerce ``board`` to a 2-D ``int`` grid; return ``None`` if the
    shape isn't recognisable. Accepts numpy arrays and nested lists."""
    if board is None:
        return None
    try:
        # np.ndarray → tolist()
        if hasattr(board, "tolist"):
            board = board.tolist()
        if not isinstance(board, list):
            return None
        rows: List[List[int]] = []
        for row in board:
            if not isinstance(row, list):
                return None
            rows.append([int(_coerce_number(c, 0)) for c in row])
        return rows
    except Exception:                                                  # noqa: BLE001
        return None


def render_state_block(
    *,
    domain: str,
    task: str,
    goal: str,
    step: int,
    entities: Sequence[Mapping[str, Any]],
    attributes: Mapping[str, Mapping[str, Any]],
    state_flags: Mapping[str, Any],
    affordances: Optional[Mapping[str, Sequence[str]]] = None,
    relations: Optional[Sequence[str]] = None,
    actions: Optional[Sequence[str]] = None,
) -> str:
    """Assemble a ``<state>...</state>`` block from per-section tables.

    The output schema matches the cold-start labeler's prompt convention
    (``labeling/skill_actions_out/run_<ts>/.../episode_<n>.json``'s
    ``schema_canonical`` field). Only the sections that
    ``parse_schema_canonical`` reads are required (``<entities>``,
    ``<attributes>``, ``<state_flags>``); ``<affordances>`` /
    ``<relations>`` / ``<actions>`` are emitted when supplied so the
    block is human-readable and forward-compatible with a richer parser.
    """
    lines: List[str] = ["<state>"]
    lines.append(f"domain={domain}")
    lines.append(f"task={task}")
    lines.append(f"goal={goal}")
    lines.append(f"step={step}")
    lines.append("")

    # <entities>
    lines.append("<entities>")
    for ent in entities:
        eid = ent.get("id")
        etype = ent.get("type", "object")
        label = ent.get("label", "")
        bid = ent.get("bid", "null")
        pos = ent.get("pos", "null")
        ontology = ent.get("ontology", "")
        lines.append(
            f"{eid}[type={etype}, label={label}, bid={bid}, pos={pos}, "
            f"ontology={ontology}]"
        )
    lines.append("")

    # <attributes>
    lines.append("<attributes>")
    for eid, attrs in attributes.items():
        for field, val in attrs.items():
            lines.append(f"{eid}.{field}={val}")
    lines.append("")

    # <state_flags>
    lines.append("<state_flags>")
    for k, v in state_flags.items():
        lines.append(f"{k}={v}")
    lines.append("")

    # <affordances>
    if affordances:
        lines.append("<affordances>")
        for eid, affs in affordances.items():
            lines.append(f"{eid}.affords=[{', '.join(affs)}]")
        lines.append("")

    # <relations>
    if relations:
        lines.append("<relations>")
        lines.extend(relations)
        lines.append("")

    # <actions>
    if actions:
        lines.append("<actions>")
        for i, a in enumerate(actions, 1):
            lines.append(f"a{i}={a}")
        lines.append("")

    lines.append("</state>")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Per-game producers
# ---------------------------------------------------------------------------


_2048_GOAL = (
    "Play 2048 on a 4x4 grid. Slide tiles up/down/left/right to merge "
    "matching numbers; goal is to create the highest tile possible "
    "(2048 wins). Larger merged tiles score more."
)


def twenty_forty_eight_producer(
    info: Mapping[str, Any],
    obs: Any,
    *,
    step: int = 0,
    task: str = "make_gaming_env/twenty_forty_eight",
    goal: str = _2048_GOAL,
    domain: str = "gymv",
) -> str:
    """Render a ``<state>`` block for ``custom_01_2048`` from
    ``env.step(...)``'s ``info`` dict.

    Reads:
        * ``info["board"]``       — 4×4 list / np.ndarray of tile values
        * ``info["total_score"]`` — cumulative env reward
        * ``info["max_tile_power"]`` — log2 of the largest tile (or
          ``info["highest_tile"]`` if directly surfaced)
        * ``info["is_legal_move"]`` (optional) — last move legality
        * ``info["illegal_move_count"]`` (optional)

    Emits one ``tile_<value>`` entity per non-zero cell at row-major
    pixel coords, plus ``board`` (container), ``empty_cells`` (region),
    ``highest_tile`` and ``score`` text entities. ``phase`` is
    ``gameover`` if no legal action remains; ``play`` otherwise.
    """
    board = _flatten_board_2d(info.get("board")) or []
    rows = len(board)
    cols = max((len(r) for r in board), default=0)

    total_score = _coerce_number(info.get("total_score", 0))
    step_score = _coerce_number(info.get("step_score", 0))
    max_power = info.get("max_tile_power", 0)
    if "highest_tile" in info:
        highest_tile = int(_coerce_number(info["highest_tile"], 0))
    else:
        # ``max_tile_power`` is log2 of the largest tile (e.g. ``5`` →
        # ``32``). When the board is empty, ``2**0 == 1`` is still the
        # reasonable null. Cap blindly at 65536 to dodge accidental
        # overflows from a corrupt env state.
        try:
            highest_tile = int(2 ** int(_coerce_number(max_power, 0)))
            highest_tile = min(highest_tile, 65_536)
            if highest_tile <= 1:
                highest_tile = 0
        except (OverflowError, ValueError):
            highest_tile = 0

    entities: List[Mapping[str, Any]] = []
    attributes: dict = {}
    affordances: dict = {}

    # e1: the board container itself.
    entities.append({
        "id": "e1", "type": "region", "label": "board", "bid": "null",
        "pos": f"0,0,{cols},{rows}", "ontology": "container_entity",
    })
    attributes["e1"] = {"state": "visible"}
    affordances["e1"] = ["inspect"]

    # e2..eN: one entity per non-zero tile, deterministic row-major.
    next_eid = 2
    n_empty = 0
    for r, row in enumerate(board):
        for c, v in enumerate(row):
            if v == 0:
                n_empty += 1
                continue
            eid = f"e{next_eid}"
            entities.append({
                "id": eid, "type": "object",
                "label": f"tile_{int(v)}", "bid": "null",
                "pos": f"{c},{r},1,1", "ontology": "selectable_entity",
            })
            attributes[eid] = {
                "state": "visible",
                "value": int(v),
            }
            affordances[eid] = ["select", "track", "compare"]
            next_eid += 1

    # Empty-cells aggregate region (matches cold-start convention).
    eid_empty = f"e{next_eid}"; next_eid += 1
    entities.append({
        "id": eid_empty, "type": "region", "label": "empty_cells",
        "bid": "null", "pos": "null", "ontology": "navigable_region",
    })
    attributes[eid_empty] = {"value": n_empty}
    affordances[eid_empty] = ["inspect"]

    # Goal-indicator text entities (highest_tile, score).
    eid_high = f"e{next_eid}"; next_eid += 1
    entities.append({
        "id": eid_high, "type": "text", "label": "highest_tile",
        "bid": "null", "pos": "null", "ontology": "goal_indicator",
    })
    attributes[eid_high] = {"value": highest_tile}
    affordances[eid_high] = ["read"]

    eid_score = f"e{next_eid}"; next_eid += 1
    entities.append({
        "id": eid_score, "type": "text", "label": "score",
        "bid": "null", "pos": "null", "ontology": "goal_indicator",
    })
    attributes[eid_score] = {"value": int(total_score)}
    affordances[eid_score] = ["read"]

    # Phase: gameover when no legal moves remain. The 2048 env exposes
    # ``info["is_legal_move"]`` for the *last* attempted move, which
    # isn't quite "no moves remain" — but combined with ``board ==
    # full`` it's a fair proxy. We bias toward "play" when uncertain.
    full = (n_empty == 0)
    last_move_legal = bool(info.get("is_legal_move", True))
    no_legal_moves_left = full and not last_move_legal
    phase = "gameover" if no_legal_moves_left else "play"
    progress = (
        # Crude progress signal: ratio of filled cells to total.
        round(1.0 - (n_empty / (rows * cols)), 3) if rows and cols else 0.0
    )

    state_flags = {
        "phase": phase,
        "progress": progress,
        "scene_type": "game_play",
        "error": "null",
        "dialog_open": "false",
        "input_pending": "true",
        # Cold-start convention also surfaces step-score so transient
        # rewards (e.g. a single merge) are visible per-frame.
        "step_score": int(step_score),
        "cumulative_reward": float(total_score),
    }

    actions = info.get("action_names") or ["up", "right", "down", "left"]
    return render_state_block(
        domain=domain, task=task, goal=goal, step=step,
        entities=entities, attributes=attributes,
        state_flags=state_flags, affordances=affordances,
        relations=None, actions=list(actions),
    )


_TETRIS_GOAL = (
    "Play Tetris. Drop tetrominoes into the playfield, fill rows to "
    "clear them, and avoid topping out. Higher line clears award more "
    "points; the level rises as lines accumulate."
)


_TETRIS_PIECE_LABELS = {
    "I": "active_piece_I", "O": "active_piece_O", "T": "active_piece_T",
    "S": "active_piece_S", "Z": "active_piece_Z",
    "J": "active_piece_J", "L": "active_piece_L",
}


def _parse_tetris_text_obs(text: str) -> Tuple[List[List[str]], Optional[str]]:
    """Best-effort parse of GamingAgent's tetris text obs.

    Returns ``(rows, active_piece_letter)``. Falls back to ``([], None)``
    on shape mismatch. The text obs is e.g.::

        Board:
        .....T....
        ....TTT...
        ...
        ( '.' = empty, IOTSZJL = tetrominoes. Active piece is rendered on board.)

    The active piece letter is whichever non-``.`` letter appears the
    fewest times (active piece is always rendered fresh each step).
    """
    rows: List[List[str]] = []
    if not text:
        return rows, None
    in_board = False
    for line in text.splitlines():
        line = line.strip()
        if line.startswith("Board:"):
            in_board = True
            continue
        if not in_board:
            continue
        if not line or line.startswith("("):
            break
        # Board rows are sequences of '.' and tetromino letters only.
        if all(ch == "." or ch.isalpha() for ch in line):
            rows.append(list(line))
    if not rows:
        return rows, None

    # Heuristic for the active piece: among tetromino letters present,
    # the one with exactly 4 cells (a single tetromino's footprint).
    counts: dict = {}
    for row in rows:
        for ch in row:
            if ch != "." and ch.isalpha():
                counts[ch] = counts.get(ch, 0) + 1
    active = None
    for letter, n in counts.items():
        if n == 4:
            active = letter
            break
    return rows, active


def tetris_producer(
    info: Mapping[str, Any],
    obs: Any,
    *,
    step: int = 0,
    task: str = "make_gaming_env/tetris",
    goal: str = _TETRIS_GOAL,
    domain: str = "gymv",
) -> str:
    """Render a ``<state>`` block for ``custom_04_tetris`` from
    ``env.step(...)``'s ``info`` dict + the textual obs.

    Reads:
        * ``info["score"]``, ``info["lines"]``, ``info["level"]``
        * ``info["next_piece_ids"]`` — list of upcoming tetrominoes
        * ``obs["text"]`` (or ``obs["textual_representation"]``) —
          the rendered board for active-piece detection.
    """
    text = ""
    if isinstance(obs, Mapping):
        text = str(obs.get("text") or obs.get("textual_representation") or "")

    score = _coerce_number(info.get("score", 0))
    lines = int(_coerce_number(info.get("lines", 0)))
    level = int(_coerce_number(info.get("level", 1)))
    next_ids = info.get("next_piece_ids") or []

    rows, active = _parse_tetris_text_obs(text)
    n_rows = len(rows)
    n_cols = len(rows[0]) if rows else 0
    n_filled = sum(1 for row in rows for ch in row if ch != ".")
    n_holes = _count_tetris_holes(rows)

    entities: List[Mapping[str, Any]] = []
    attributes: dict = {}
    affordances: dict = {}

    # e1: board container.
    entities.append({
        "id": "e1", "type": "region", "label": "board", "bid": "null",
        "pos": f"0,0,{n_cols},{n_rows}", "ontology": "container_entity",
    })
    attributes["e1"] = {"state": "visible"}
    affordances["e1"] = ["inspect"]

    next_eid = 2
    if active is not None:
        label = _TETRIS_PIECE_LABELS.get(active, f"active_piece_{active}")
        eid = f"e{next_eid}"; next_eid += 1
        entities.append({
            "id": eid, "type": "object", "label": label,
            "bid": "null", "pos": "null", "ontology": "selectable_entity",
        })
        attributes[eid] = {"state": "visible", "value": active}
        affordances[eid] = ["select", "rotate", "drop"]

    # Goal-indicator text entities — score, lines, level, holes.
    for label, value, ontology in (
        ("score", int(score), "goal_indicator"),
        ("lines_cleared", lines, "goal_indicator"),
        ("level", level, "goal_indicator"),
        ("holes", n_holes, "goal_indicator"),
        ("filled_cells", n_filled, "goal_indicator"),
    ):
        eid = f"e{next_eid}"; next_eid += 1
        entities.append({
            "id": eid, "type": "text", "label": label,
            "bid": "null", "pos": "null", "ontology": ontology,
        })
        attributes[eid] = {"value": value}
        affordances[eid] = ["read"]

    state_flags = {
        "phase": "play",  # tetris env doesn't surface gameover in info
        "progress": 0.0,
        "scene_type": "game_play",
        "error": "null",
        "dialog_open": "false",
        "input_pending": "true",
        "cumulative_reward": float(score),
        "next_piece_ids": ",".join(str(x) for x in next_ids),
    }

    actions = info.get("action_names") or [
        "no_op", "left", "right",
        "rotate_left", "rotate_right",
        "soft_drop", "hard_drop",
    ]
    return render_state_block(
        domain=domain, task=task, goal=goal, step=step,
        entities=entities, attributes=attributes,
        state_flags=state_flags, affordances=affordances,
        relations=None, actions=list(actions),
    )


def _count_tetris_holes(rows: Sequence[Sequence[str]]) -> int:
    """Count tetris "holes" — empty cells that have at least one
    non-empty cell anywhere above them in the same column.

    This matches the standard tetris-AI convention used in
    ``Commit/Position``'s prose ('Hole count increases from 3 to 4').
    """
    if not rows:
        return 0
    n_cols = max(len(r) for r in rows)
    n = 0
    for c in range(n_cols):
        seen_filled = False
        for r in range(len(rows)):
            if c >= len(rows[r]):
                continue
            ch = rows[r][c]
            if ch != ".":
                seen_filled = True
            elif seen_filled:
                n += 1
    return n


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------


_PRODUCERS: dict = {
    "twenty_forty_eight": twenty_forty_eight_producer,
    "tetris": tetris_producer,
}


def make_gaming_env_producer(game: str) -> Optional[SchemaProducer]:
    """Look up a deterministic schema producer for one
    ``env_wrappers.gym_like.make_gaming_env`` game name.

    Returns ``None`` for envs we haven't built a producer for yet
    (candy_crush, super_mario, …) so callers can fall back gracefully
    to the executor's plain-text path. Day-4 ships 2048 + tetris;
    Day-5+ extends to the rest as the lift coverage expands.
    """
    return _PRODUCERS.get(game)
