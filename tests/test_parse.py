import sys; sys.path.insert(0, "src")
from parse import parse_pair, group_id, split_of

C = "\n\nHuman: q"
def pair(a, b, c=C): return parse_pair(c + "\n\nAssistant: " + a, c + "\n\nAssistant: " + b)

def test_basic():
    c = "\n\nHuman: hi\n\nAssistant: hello\n\nHuman: more?"
    r, _ = pair("A", "B", c)
    assert r == (c.strip(), "A", "B")

def test_drops():
    assert pair("A", "A")[1] == "identical_responses"
    assert parse_pair(C + "\n\nAssistant: A", "\n\nHuman: z\n\nAssistant: B")[1] == "context_mismatch"
    assert pair("", "B")[1] == "empty_response"
    assert parse_pair("no turn", "no turn")[1] == "no_assistant_turn"

def test_echo_dropped():
    assert pair("q", "B")[1] == "echoes_user"
    assert pair("Human: q", "B")[1] == "echoes_user"
    assert pair("quick answer", "B")[1] is None  # only exact echoes are dropped

def test_group_stable_across_continuations():
    a = "\n\nHuman: q"; b = a + "\n\nAssistant: x\n\nHuman: y"
    assert group_id(a) == group_id(b) and split_of(group_id(a)) == split_of(group_id(b))
