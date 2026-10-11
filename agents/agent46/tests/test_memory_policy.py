"""Memory label invariant: every label is user-scoped, collisions are impossible, and no code
path writes a semantic memory with an unscoped label."""

import ast
import os
import uuid
from pathlib import Path

import pytest

from agent_agent46.memory_policy import SEPARATOR, is_scoped_to, normalize_key, user_scoped_label

SRC = Path(__file__).resolve().parent.parent / "src" / "agent_agent46"


def test_label_includes_user_and_key():
    label = user_scoped_label("user-1", "Home Airport")
    assert label == f"user-1{SEPARATOR}home_airport"
    assert is_scoped_to(label, "user-1") and not is_scoped_to(label, "user-2")


def test_same_key_different_users_get_different_labels():
    assert user_scoped_label("user-1", "email") != user_scoped_label("user-2", "email")


def test_underscore_ambiguity_cannot_collide():
    # With a plain "_" join, ("a", "b_c") and ("a_b", "c") would both be "a_b_c".
    assert user_scoped_label("a", "b_c") != user_scoped_label("a_b", "c")


@pytest.mark.parametrize("user_id", ["", "   ", f"bad{SEPARATOR}user"])
def test_bad_user_id_rejected(user_id):
    with pytest.raises(ValueError):
        user_scoped_label(user_id, "email")


@pytest.mark.parametrize("key", ["", "a" * 65, f"x{SEPARATOR}y", "has/slash", "emoji\U0001F600"])
def test_bad_key_rejected(key):
    with pytest.raises(ValueError):
        user_scoped_label("user-1", key)


def test_normalize_key():
    assert normalize_key("  Vehicle Preference ") == "vehicle_preference"


def _is_helper_call(node: ast.AST) -> bool:
    return (
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "user_scoped_label"
    )


def _label_comes_from_helper(tree: ast.AST, call: ast.Call) -> bool:
    """The save_semantic label= argument must be user_scoped_label(...) or a name assigned from it."""
    label_arg = next((kw.value for kw in call.keywords if kw.arg == "label"), None)
    if label_arg is None:
        return False
    if _is_helper_call(label_arg):
        return True
    if not isinstance(label_arg, ast.Name):
        return False
    for fn in ast.walk(tree):
        if isinstance(fn, (ast.FunctionDef, ast.AsyncFunctionDef)) and any(
            node is call for node in ast.walk(fn)
        ):
            return any(
                isinstance(n, ast.Assign)
                and _is_helper_call(n.value)
                and any(isinstance(t, ast.Name) and t.id == label_arg.id for t in n.targets)
                for n in ast.walk(fn)
            )
    return False


def test_every_save_semantic_in_source_uses_scoped_labels():
    """The invariant itself: a new tool that writes a fact without user_scoped_label fails here."""
    found = 0
    for path in SRC.glob("*.py"):
        tree = ast.parse(path.read_text())
        for node in ast.walk(tree):
            if (
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Attribute)
                and node.func.attr == "save_semantic"
            ):
                found += 1
                assert _label_comes_from_helper(tree, node), (
                    f"{path.name}:{node.lineno} saves a semantic memory with a label that "
                    "does not come from user_scoped_label()"
                )
    assert found >= 1, "expected at least one save_semantic call to guard"


# ---- live check against the local Docker memory server (skipped when it is not running) ----

MEMORY_URL = os.environ.get("MEMORY_BASE_URL", "http://localhost:53085")
MEMORY_DB = os.environ.get("MEMORY_DB", "mdb_memory_99de6ec9720fcea9fc760ccc")
MONGO_URI = os.environ.get("TEST_MONGODB_URI", "mongodb://localhost:53087/?directConnection=true")


@pytest.fixture
def live():
    from pymongo import MongoClient
    from agent_engine_sdk_memory import Memory

    try:
        client = MongoClient(MONGO_URI, serverSelectionTimeoutMS=1500)
        client.admin.command("ping")
        db = client[MEMORY_DB]
        if "memory_semantic" not in db.list_collection_names():
            pytest.skip("memory database not provisioned")
    except Exception:
        pytest.skip("local Docker Mongo not reachable")
    tag = f"memtest-{uuid.uuid4().hex[:8]}"
    users = [f"{tag}-a", f"{tag}-b"]
    yield Memory(base_url=MEMORY_URL), db, users
    db["memory_semantic"].delete_many({"user_id": {"$in": users}})


def test_live_two_users_same_key_do_not_overwrite_each_other(live):
    from agent_engine_sdk_memory import MemoryRequestContext

    memory, db, (user_a, user_b) = live
    for user, text in ((user_a, "Customer email: a@example.com"), (user_b, "Customer email: b@example.com")):
        chat = memory.bind(MemoryRequestContext(user_id=user, session_id="s"))
        try:
            chat.save_semantic(label=user_scoped_label(user, "email"), text=text, source="test")
        except Exception as e:  # server unreachable -> skip rather than fail
            if "Connection" in type(e).__name__:
                pytest.skip("memory server not reachable")
            raise
    docs = {d["user_id"]: d["content"] for d in db["memory_semantic"].find({"user_id": {"$in": [user_a, user_b]}})}
    assert docs == {user_a: "Customer email: a@example.com", user_b: "Customer email: b@example.com"}


def test_live_unscoped_label_reproduces_the_overwrite_hazard(live):
    """Documents WHY the policy exists: a shared label moves one user's fact to the other user."""
    from agent_engine_sdk_memory import MemoryRequestContext

    memory, db, (user_a, user_b) = live
    shared = f"shared-{uuid.uuid4().hex[:6]}"
    try:
        memory.bind(MemoryRequestContext(user_id=user_a, session_id="s")).save_semantic(label=shared, text="A's fact", source="test")
        memory.bind(MemoryRequestContext(user_id=user_b, session_id="s")).save_semantic(label=shared, text="B's fact", source="test")
    except Exception as e:
        if "Connection" in type(e).__name__:
            pytest.skip("memory server not reachable")
        raise
    owners = [d["user_id"] for d in db["memory_semantic"].find({"label": shared})]
    assert owners == [user_b]  # user A's fact is gone
    db["memory_semantic"].delete_many({"label": shared})
