"""Scenario assistant: knowledge base, grounding checks, keyword mode, and the Gemma path against a fake API."""
import json
import pickle
from pathlib import Path

import pytest

from fiscalsim import assistant as A
from fiscalsim import assistant_eval as AE
from fiscalsim import config as C
from fiscalsim import knowledge as K

BUNDLE = Path(__file__).resolve().parent.parent / "outputs" / "dashboard_bundle.pkl"


@pytest.fixture(scope="module")
def env():
    with open(BUNDLE, "rb") as f:
        return A.env_from_bundle(pickle.load(f))


class FakeResp:
    def __init__(self, code, body):
        self.status_code, self._body, self.text = code, body, json.dumps(body)

    def json(self):
        return self._body


class FakeAPI:
    """Answers like the Gemini API: a function call for routing requests, text for writing requests."""

    def __init__(self, route, write, fail_first=0, fail_code=429):
        self.calls, self.route, self.write, self.fail, self.code = [], route, write, fail_first, fail_code

    def post(self, url, json=None, timeout=None, headers=None):
        self.calls.append((url, json, headers))
        if self.fail:
            self.fail -= 1
            return FakeResp(self.code, {"error": {"message": "Resource exhausted. Please retry in 0.01s."}})
        text = json["contents"][0]["parts"][0]["text"]
        if "tools" in json:
            name, args = self.route(text)
            parts = [{"text": "thinking", "thought": True}, {"functionCall": {"name": name, "args": args}}]
        else:
            parts = [{"text": "draft 9.99%", "thought": True}, {"text": self.write(text)}]
        return FakeResp(200, {"candidates": [{"content": {"role": "model", "parts": parts}}]})


def client(api):
    return A.GemmaClient("test-key", "gemma-test", session=api)


def test_knowledge_matches_config_and_appendices():
    ids = set(K.BY_ID)
    assert {f"A{i}" for i in range(1, 17)} <= ids and {f"P{i}" for i in range(1, 15)} <= ids and set(C.SCENARIOS) <= ids
    assert f"{C.S0.employee_contribution:.0%}" in K.BY_ID["A4"]["text"]
    assert f"{C.S0.employer_contribution:.0%}" in K.BY_ID["A4"]["text"]
    assert f"{C.BASE.headcount:,}" in K.BY_ID["P1"]["text"]
    assert K.RETRIEVER.search("current pension formula")[0] == "A1"
    assert K.RETRIEVER.search("weather in Phnom Penh") == []


def test_numbers_and_citations_are_checked(env):
    r = A.run_tool("run_reform", A.clean_args("run_reform", {"retirement_age": 62}), env)
    cost = A.fmt(r.facts["cost_2046"])
    assert A.check(f"Total cost is {cost} of GDP in 2046 [A9].", r, "q")[0]
    assert A.check(f"ថ្លៃដើម {cost.translate(str.maketrans('0123456789', '០១២៣៤៥៦៧៨៩'))} [A9]។", r, "q")[0]
    assert not A.check("Total cost is 9.87% of GDP [A9].", r, "q")[0]          # invented number
    assert not A.check(f"Total cost is {cost} of GDP [A3].", r, "q")[0]       # source not behind this result
    assert not A.check(f"Total cost is {cost} of GDP.", r, "q")[0]            # no citation
    assert A.check("Retirement age 62 [A6].", r, "What if the retirement age is 62?")[0]  # numbers from the question


def test_keyword_mode_runs_the_simulator(env):
    ans = A.answer("What if the retirement age is raised to 62?", env)
    assert ans.mode == "rules" and ans.tool == "run_reform" and ans.settings == {"retirement_age": 62}
    s, _ = A._run(env, A.scenario_from({"retirement_age": 62}))
    assert A.fmt(ans.result.facts["cost_2046"]) == f"{s['total_cost_gdp_2046_p50']:.2f}%"
    assert A.check(ans.text, ans.result, ans.question)[0] and "A6" in ans.cited
    km = A.answer("បើដំឡើងអាយុចូលនិវត្តន៍ដល់ ៦២ ឆ្នាំ តើមានអ្វីកើតឡើង?", env)
    assert km.lang == "km" and km.settings == {"retirement_age": 62}
    assert A.answer("What's the weather in Phnom Penh today?", env).tool == "none"


def test_gemma_answer_uses_only_model_numbers(env):
    api = FakeAPI(lambda q: ("run_reform", {"retirement_age": 62.0}),
                  lambda _: "Raising the retirement age to 62 [A6] puts total cost in 2046 at {cost_2046} of GDP, "
                            "against { cost_2046_s0 } now [A9]. The fund runs out in {runout_year} [D2].")
    ans = A.answer("Suppose staff work until 62?", env, client(api))
    assert ans.mode == "gemma" and ans.grounded_raw and not ans.fallback
    assert A.fmt(ans.result.facts["cost_2046"]) in ans.text and "{" not in ans.text
    assert ans.cited == ["A6", "A9", "D2"]
    url, body, headers = api.calls[0]
    assert "gemma-test:generateContent" in url and headers["x-goog-api-key"] == "test-key"
    assert {f["name"] for f in body["tools"][0]["functionDeclarations"]} == set(A.TOOL_NAMES) - {"none"}
    assert "systemInstruction" in body and "9.99" not in ans.text  # thought parts are ignored


@pytest.mark.parametrize("text, why", [("Cost falls to 2.91% of GDP [A9].", "numbers"),
                                       ("Cost is {cost_2046} [Z9].", "citations"),
                                       ("Cost is {cost_2099} [A9].", "placeholders")])
def test_gemma_answer_failing_checks_falls_back_to_template(env, text, why):
    api = FakeAPI(lambda q: ("run_reform", {"retirement_age": 60}), lambda _: text)
    ans = A.answer("Retire at 60?", env, client(api))
    assert ans.grounded_raw is False and why in ans.fallback
    assert ans.text == A.template(ans.result, "en")


def test_gemma_retries_rate_limits_and_falls_back_on_errors(env, monkeypatch):
    monkeypatch.setattr(A.time, "sleep", lambda s: None)
    api = FakeAPI(lambda q: ("compare_scenarios", {"codes": ["S0", "S5"]}), lambda _: "S5 costs {S5_cost_2046} [S5].",
                  fail_first=1)
    ans = A.answer("Compare S0 and S5", env, client(api))
    assert ans.mode == "gemma" and ans.settings == {"codes": ["S0", "S5"]} and len(api.calls) == 3
    bad = FakeAPI(lambda q: ("run_reform", {}), lambda _: "", fail_first=1, fail_code=400)
    ans = A.answer("What if the retirement age is 62?", env, client(bad))
    assert ans.mode == "rules" and "HTTP 400" in ans.error and ans.settings == {"retirement_age": 62}


def test_eval_scores_keyword_baseline(env):
    df, summ = AE.evaluate(env)
    assert len(df) == len(AE.QUESTIONS) + len(AE.HELD_OUT)
    assert all(v["model_numbers"] == 1.0 for v in summ.values())
    assert summ[("held_out", "all")]["right_settings"] < summ[("development", "all")]["right_settings"]


def test_no_key_means_no_client(monkeypatch):
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)

    def missing(name):
        raise FileNotFoundError("no secrets file")
    assert A.client_from_settings(missing) is None
    c = A.client_from_settings(lambda n: {"GEMINI_API_KEY": "k", "GEMMA_MODEL": "gemma-4-26b-a4b-it"}.get(n))
    assert c.key == "k" and c.model == "gemma-4-26b-a4b-it"
