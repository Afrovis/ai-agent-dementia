import json

from agent.llm import Composition, FakeLLM, Intent, Interpretation

from dialogue_bench.report import print_json_report, result_to_dict
from dialogue_bench.scenarios import DialogueScenario
from dialogue_bench.scoring import run_model


def _result():
    client = FakeLLM(
        interpretations=[Interpretation(intent=Intent.FINE, distress=0)],
        compositions=[Composition(text="I hear you, and we can rest together now.")],
    )
    scenario = DialogueScenario(id="one", utterance="I'm okay", expected_intent=Intent.FINE)
    return run_model("fake", client, [scenario])


def test_json_report_omits_generated_text_by_default(capsys):
    print_json_report([_result()])
    report = json.loads(capsys.readouterr().out)
    assert report["results"][0]["intent_accuracy"] == 1.0
    assert "composition_text" not in report["results"][0]["scenarios"][0]


def test_generated_text_can_be_explicitly_included():
    report = result_to_dict(_result(), include_text=True)
    assert report["scenarios"][0]["composition_text"].startswith("I hear you")
