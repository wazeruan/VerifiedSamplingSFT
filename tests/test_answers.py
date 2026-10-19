import pytest

from vsft.answers import answers_match, extract_answer, extract_boxed, normalize_answer


@pytest.mark.parametrize(
    "text,expected",
    [
        ("so \\boxed{42}.", "42"),
        ("\\boxed{\\frac{3}{4}}", "\\frac{3}{4}"),
        ("\\boxed{\\frac{\\sqrt{2}}{2}} done", "\\frac{\\sqrt{2}}{2}"),
        ("first \\boxed{1} then \\boxed{2}", "2"),
        ("\\fbox{7}", "7"),
        ("\\boxed{1} and \\fbox{9}", "9"),
        ("\\boxed 5", "5"),
        ("\\boxed{unclosed", None),
        ("no box here", None),
        ("\\boxed{\\{a, b\\}}", "\\{a, b\\}"),  # balanced escaped braces
    ],
)
def test_extract_boxed(text, expected):
    assert extract_boxed(text) == expected


@pytest.mark.parametrize(
    "text,expected",
    [
        ("\\boxed 5. Then the set {1, 2} is irrelevant", "5"),
        ("\\boxed 12, done {x}", "12"),
        ("\\boxed   {7}", "7"),
        ("\\fbox {8}", "8"),
    ],
)
def test_extract_boxed_space_forms(text, expected):
    assert extract_boxed(text) == expected


def test_extract_answer_fallbacks():
    assert extract_answer("blah\nFinal answer: 12") == "12"
    assert extract_answer("The answer is: 7 apples") == "7 apples"
    assert extract_answer("nothing") is None
    # boxed wins over a textual answer
    assert extract_answer("answer: 3\n\\boxed{4}") == "4"


@pytest.mark.parametrize(
    "text,expected",
    [
        ("The answer is (C).", "C"),
        ("answer: b", "B"),
        ("The answer is (a).", "A"),
        ("Answer is c", "C"),
        ("the answer is a tricky one", None),
        ("I think a good idea is this", None),
        ("(D)", "D"),
        ("Answer is A. Wait, the answer is E", "E"),
        ("\\boxed{B}", "B"),
        ("no letter", None),
    ],
)
def test_extract_answer_mc(text, expected):
    assert extract_answer(text, "mc") == expected


def test_extract_answer_mc_trailing_sentence():
    assert extract_answer("The answer is (C).\nI am confident in this.", "mc") == "C"
    assert extract_answer("Some reasoning.\n(B)\nA final remark.", "mc") == "B"


@pytest.mark.parametrize("boxed", ["\\text{C}", "\\textbf{C}", "\\mathrm{C}", "(C)", "C."])
def test_mc_boxed_wrappers(boxed):
    assert normalize_answer(boxed, "mc") == "C"
    assert answers_match(extract_answer(f"\\boxed{{{boxed}}}", "mc"), "C", "mc")


def test_normalize_non_str_reference():
    assert answers_match("0", 0, "math")
    assert answers_match("42", 42, "math")


@pytest.mark.parametrize(
    "pred,ref",
    [
        ("42", "42"),
        ("$42$", "42"),
        ("42.", "42"),
        ("\\frac{3}{4}", "3/4"),
        ("\\dfrac{3}{4}", "\\frac{3}{4}"),
        ("\\frac34", "\\frac{3}{4}"),
        ("0.75", "\\frac{3}{4}"),
        ("-\\frac{1}{2}", "-0.5"),
        ("1,000", "1000"),
        ("1,234,567.5", "1234567.5"),
        ("\\$1,000", "1000"),
        ("50\\%", "50"),
        ("90^\\circ", "90"),
        ("\\text{5}", "5"),
        ("\\left(1,2\\right)", "(1,2)"),
        ("1000", "1e3"),
    ],
)
def test_math_match(pred, ref):
    assert answers_match(pred, ref, "math")


@pytest.mark.parametrize(
    "pred,ref",
    [("41", "42"), ("\\frac{1}{0}", "1"), (None, "1"), ("1", None), ("1,2", "12"), ("x", "y")],
)
def test_math_mismatch(pred, ref):
    assert not answers_match(pred, ref, "math")


def test_text_and_mc_match():
    assert answers_match("New  York!", "new york", "text")
    assert answers_match("(c)", "C", "mc")
    assert not answers_match("B", "C", "mc")
    assert normalize_answer(None) is None
