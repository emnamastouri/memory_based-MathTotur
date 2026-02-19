from memory.schema import MemoryItem
from memory.store import append_memory, load_memory
from memory.embedder import Embedder
from memory.index import build_or_load_index, search

MEM_PATH = "data/memory.jsonl"
IDX_PATH = "data/faiss.index"


def seed_if_empty():
    items = load_memory(MEM_PATH)
    if items:
        return

    append_memory(MEM_PATH, MemoryItem(
        memory_id="m1",
        student_id="s1",
        topic="limits",
        problem="Compute lim_{x->0} sin(x)/x",
        student_attempt="I replaced x=0 so sin(0)/0 = 0",
        error_type="indeterminate_form",
        teacher_move="error_signaling",
        assistant_response="Direct substitution gives 0/0, which is indeterminate. Use the standard limit or small-angle approximation.",
        verified=True,
        tags=["bac", "trigonometry", "limits"],
        created_at=MemoryItem.now_iso(),
    ))

    append_memory(MEM_PATH, MemoryItem(
        memory_id="m2",
        student_id="s1",
        topic="derivatives",
        problem="Differentiate f(x)=x^2",
        student_attempt="f'(x)=2x^2",
        error_type="power_rule",
        teacher_move="hint",
        assistant_response="Recall: d/dx x^n = n x^(n-1). Apply with n=2.",
        verified=True,
        tags=["bac", "derivatives"],
        created_at=MemoryItem.now_iso(),
    ))


def main():
    seed_if_empty()
    items = load_memory(MEM_PATH)
    embedder = Embedder()
    index = build_or_load_index(items, embedder, IDX_PATH)

    results = search(
        index=index,
        memory_items=items,
        embedder=embedder,
        topic="limits",
        problem="Compute lim_{x->0} sin(x)/x",
        student_attempt="I think it is 0 because sin(0)=0",
        error_type="indeterminate_form",
        k=3,
    )

    print("Top retrieved memories:")
    for m, score in results:
        print("-", m.memory_id, "score=", round(score, 3), "|", m.error_type, "|", m.teacher_move)


if __name__ == "__main__":
    main()

