"""Headless Streamlit page tests using streamlit.testing.v1.AppTest.

These exercise each page's golden path (and a couple of edge cases) without
a real browser, catching import errors, widget-wiring bugs, and exceptions
raised by gauge_core calls triggered from the UI.
"""
import os
from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

from gauge_core._env import load_dotenv

load_dotenv()
HAS_DEEPSEEK_KEY = bool(os.environ.get("DEEPSEEK_API_KEY"))

APP_DIR = Path(__file__).resolve().parent.parent / "app"


def _no_exceptions(at: AppTest) -> None:
    exceptions = list(at.exception)
    assert not exceptions, f"Unhandled exception(s) in app: {exceptions}"


def test_home_page():
    at = AppTest.from_file(str(APP_DIR / "Home.py"))
    at.run(timeout=60)
    _no_exceptions(at)
    assert len(at.metric) >= 2


def test_single_prediction_demo_golden_path():
    at = AppTest.from_file(str(APP_DIR / "pages" / "2_🔬_Single_Prediction.py"))
    at.run(timeout=60)
    _no_exceptions(at)
    # Click the one-click demo button explicitly by label match.
    demo_buttons = [b for b in at.button if "demo" in (b.label or "").lower()]
    assert demo_buttons, "Expected a one-click demo button on the Single Prediction page"
    demo_buttons[0].click().run(timeout=60)
    _no_exceptions(at)
    predict_buttons = [b for b in at.button if "Predict" in (b.label or "")]
    assert predict_buttons
    predict_buttons[0].click().run(timeout=60)
    _no_exceptions(at)
    assert len(at.markdown) > 0


@pytest.mark.parametrize(
    "mode_key,expected_label",
    [("gdsc_cell_split", "Known GDSC cell line"), ("prism_cell_split", "Known DepMap cell line (PRISM)")],
)
def test_known_sample_label_matches_mode(mode_key, expected_label):
    at = AppTest.from_file(str(APP_DIR / "pages" / "2_🔬_Single_Prediction.py"))
    at.run(timeout=60)
    at.sidebar.radio[0].set_value(mode_key).run(timeout=60)
    _no_exceptions(at)
    assert expected_label in at.radio(key="sp_sample_source").options


def test_expression_upload_download_templates_present():
    at = AppTest.from_file(str(APP_DIR / "pages" / "2_🔬_Single_Prediction.py"))
    at.run(timeout=60)
    at.radio(key="sp_sample_source").set_value("Upload my own expression file").run(timeout=60)
    _no_exceptions(at)
    labels = [d.label for d in at.get("download_button")]
    assert any("samples as rows" in (label or "") for label in labels)
    assert any("genes as rows" in (label or "") for label in labels)


def test_single_prediction_custom_smiles():
    at = AppTest.from_file(str(APP_DIR / "pages" / "2_🔬_Single_Prediction.py"))
    at.run(timeout=60)
    at.radio(key="sp_drug_source").set_value("Custom SMILES").run(timeout=60)
    _no_exceptions(at)
    at.text_input(key="sp_smiles").set_value("CC(=O)OC1=CC=CC=C1C(=O)O").run(timeout=60)
    predict_buttons = [b for b in at.button if "Predict" in (b.label or "")]
    predict_buttons[0].click().run(timeout=60)
    _no_exceptions(at)


def test_drug_ranking_full_library():
    at = AppTest.from_file(str(APP_DIR / "pages" / "4_🏆_Drug_Ranking.py"))
    at.run(timeout=60)
    _no_exceptions(at)
    rank_buttons = [b for b in at.button if "Rank" in (b.label or "")]
    assert rank_buttons
    rank_buttons[0].click().run(timeout=120)
    _no_exceptions(at)
    assert len(at.dataframe) > 0


def test_single_prediction_invalid_smiles_shows_error_not_crash():
    at = AppTest.from_file(str(APP_DIR / "pages" / "2_🔬_Single_Prediction.py"))
    at.run(timeout=60)
    at.radio(key="sp_drug_source").set_value("Custom SMILES").run(timeout=60)
    at.text_input(key="sp_smiles").set_value("this_is_not_a_smiles_string_!!").run(timeout=60)
    predict_buttons = [b for b in at.button if "Predict" in (b.label or "")]
    predict_buttons[0].click().run(timeout=60)
    _no_exceptions(at)  # the app must show a friendly st.error, not raise
    assert len(at.error) > 0


def test_batch_prediction_demo():
    at = AppTest.from_file(str(APP_DIR / "pages" / "3_📊_Batch_Prediction.py"))
    at.run(timeout=60)
    at.radio(key="bp_input_mode").set_value("Bundled demo file").run(timeout=60)
    _no_exceptions(at)
    run_buttons = [b for b in at.button if "Run batch" in (b.label or "")]
    assert run_buttons
    run_buttons[0].click().run(timeout=120)
    _no_exceptions(at)
    assert len(at.dataframe) > 0


def test_batch_prediction_known_cell_lines():
    # Default input mode is "known cell lines (no upload needed)".
    at = AppTest.from_file(str(APP_DIR / "pages" / "3_📊_Batch_Prediction.py"))
    at.run(timeout=60)
    cells = at.multiselect(key="bp_known_cells")
    assert len(cells.options) > 0  # the regression the user reported: no selectable cell lines
    cells.set_value(cells.options[:2]).run(timeout=60)
    _no_exceptions(at)
    run_buttons = [b for b in at.button if "Run batch" in (b.label or "")]
    run_buttons[0].click().run(timeout=120)
    _no_exceptions(at)
    assert len(at.dataframe) > 0


def test_molecular_design_generate_and_score():
    at = AppTest.from_file(str(APP_DIR / "pages" / "12_🧬_Molecular_Design.py"))
    at.run(timeout=60)
    # Paste two valid SMILES and score them (deterministic, no generation needed).
    at.radio(key="md_candidate_source").set_value("✍️ Paste SMILES").run(timeout=60)
    at.text_area(key="md_smiles_text").set_value(
        "CC(=O)OC1=CC=CC=C1C(=O)O\nCOCCOc1cc2ncnc(Nc3cccc(c3)C#C)c2cc1OCCOC"
    ).run(timeout=60)
    score_buttons = [b for b in at.button if "Score candidates" in (b.label or "")]
    score_buttons[0].click().run(timeout=120)
    _no_exceptions(at)
    assert len(at.dataframe) > 0


def test_combination_scoring():
    at = AppTest.from_file(str(APP_DIR / "pages" / "5_🧩_Combination_Scoring.py"))
    at.run(timeout=60)
    lib_names_widget = at.multiselect(key="cs_drug_select")
    options = lib_names_widget.options
    lib_names_widget.set_value(options[:3]).run(timeout=60)
    score_buttons = [b for b in at.button if "Score" in (b.label or "")]
    score_buttons[0].click().run(timeout=120)
    _no_exceptions(at)
    assert len(at.dataframe) > 0


def test_kg_explainability():
    at = AppTest.from_file(str(APP_DIR / "pages" / "6_🧠_Knowledge_Graph_Explainability.py"))
    at.run(timeout=60)
    explain_buttons = [b for b in at.button if "Explain" in (b.label or "")]
    explain_buttons[0].click().run(timeout=60)
    _no_exceptions(at)


def test_tcga_stratification():
    at = AppTest.from_file(str(APP_DIR / "pages" / "7_🩺_Patient_Stratification.py"))
    at.run(timeout=60)
    compare_buttons = [b for b in at.button if "Compare" in (b.label or "")]
    compare_buttons[0].click().run(timeout=60)
    _no_exceptions(at)


def test_expression_data_analysis_demo():
    at = AppTest.from_file(str(APP_DIR / "pages" / "8_📈_Expression_Data_Analysis.py"))
    at.run(timeout=60)
    at.checkbox(key="ea_demo").set_value(True).run(timeout=60)
    _no_exceptions(at)
    assert len(at.tabs) > 0


def test_about_page():
    at = AppTest.from_file(str(APP_DIR / "pages" / "13_ℹ️_About_Model_Card.py"))
    at.run(timeout=60)
    _no_exceptions(at)
    assert len(at.markdown) > 0


@pytest.mark.parametrize("mode_key", ["gdsc_cell_split", "gdsc_drug_split", "prism_cell_split", "prism_drug_split"])
def test_home_page_both_modes(mode_key):
    at = AppTest.from_file(str(APP_DIR / "Home.py"))
    at.run(timeout=60)
    at.sidebar.radio[0].set_value(mode_key).run(timeout=60)
    _no_exceptions(at)


def test_tcga_stratification_real_patient_demo():
    at = AppTest.from_file(str(APP_DIR / "pages" / "7_🩺_Patient_Stratification.py"))
    at.run(timeout=60)
    at.radio(key="tcga_sample_source").set_value("Real de-identified TCGA patient (demo)").run(timeout=60)
    _no_exceptions(at)
    compare_buttons = [b for b in at.button if "Compare" in (b.label or "")]
    compare_buttons[0].click().run(timeout=60)
    _no_exceptions(at)
    assert len(at.info) > 0


def test_combination_scoring_drugcomb_validation():
    at = AppTest.from_file(str(APP_DIR / "pages" / "5_🧩_Combination_Scoring.py"))
    at.run(timeout=60)
    compute_buttons = [b for b in at.button if "Compute GAUGE" in (b.label or "")]
    compute_buttons[0].click().run(timeout=120)
    _no_exceptions(at)
    assert len(at.dataframe) > 0


def test_pharmacogenomic_explorer():
    at = AppTest.from_file(str(APP_DIR / "pages" / "10_🧪_Pharmacogenomic_Explorer.py"))
    at.run(timeout=60)
    _no_exceptions(at)
    demo_buttons = [b for b in at.button if "EGFR" in (b.label or "")]
    demo_buttons[0].click().run(timeout=60)
    _no_exceptions(at)
    assert len(at.tabs) == 3


def test_kg_network_viewer():
    at = AppTest.from_file(str(APP_DIR / "pages" / "11_🕸️_KG_Network_Viewer.py"))
    at.run(timeout=60)
    demo_buttons = [b for b in at.button if "demo" in (b.label or "").lower()]
    demo_buttons[0].click().run(timeout=60)
    render_buttons = [b for b in at.button if "Render" in (b.label or "")]
    render_buttons[0].click().run(timeout=60)
    _no_exceptions(at)


def test_molecular_design_scoring():
    at = AppTest.from_file(str(APP_DIR / "pages" / "12_🧬_Molecular_Design.py"))
    at.run(timeout=60)
    # The published-demo button lives under the "Published REINVENT4 demo" source.
    at.radio(key="md_candidate_source").set_value("📄 Published REINVENT4 demo").run(timeout=60)
    demo_buttons = [b for b in at.button if "demo" in (b.label or "").lower()]
    demo_buttons[0].click().run(timeout=60)
    _no_exceptions(at)
    score_buttons = [b for b in at.button if "Score candidates" in (b.label or "")]
    score_buttons[0].click().run(timeout=120)
    _no_exceptions(at)
    assert len(at.dataframe) > 0


def test_drug_ranking_demo_button():
    at = AppTest.from_file(str(APP_DIR / "pages" / "4_🏆_Drug_Ranking.py"))
    at.run(timeout=60)
    demo_buttons = [b for b in at.button if "demo" in (b.label or "").lower()]
    demo_buttons[0].click().run(timeout=120)
    _no_exceptions(at)
    assert len(at.dataframe) > 0


def test_kg_explainability_demo_button():
    at = AppTest.from_file(str(APP_DIR / "pages" / "6_🧠_Knowledge_Graph_Explainability.py"))
    at.run(timeout=60)
    demo_buttons = [b for b in at.button if "demo" in (b.label or "").lower()]
    demo_buttons[0].click().run(timeout=60)
    _no_exceptions(at)


def test_expression_data_analysis_gtex_tab():
    at = AppTest.from_file(str(APP_DIR / "pages" / "8_📈_Expression_Data_Analysis.py"))
    at.run(timeout=60)
    at.checkbox(key="ea_demo").set_value(True).run(timeout=60)
    _no_exceptions(at)
    assert len(at.tabs) == 5


def test_gauge_assistant_page_loads():
    at = AppTest.from_file(str(APP_DIR / "pages" / "1_🤖_GAUGE_Assistant.py"))
    at.run(timeout=60)
    _no_exceptions(at)
    # No key entered → info message shown, chat input blocked
    assert len(at.button) >= 4  # demo-question quick-start cards + New chat
    assert len(at.info) >= 1    # "enter API key" info box visible


def test_gauge_assistant_new_chat_button_present():
    at = AppTest.from_file(str(APP_DIR / "pages" / "1_🤖_GAUGE_Assistant.py"))
    at.run(timeout=60)
    _no_exceptions(at)
    assert any("New chat" in (b.label or "") for b in at.button)
    # The live-process toggle exists and defaults on.
    assert any(t.label and "process" in t.label.lower() for t in at.toggle)


def test_drug_sensitivity_kb_page_loads():
    # GAUGE-free page: loading must not require the model bundle or network.
    at = AppTest.from_file(str(APP_DIR / "pages" / "9_💊_Drug_Sensitivity_KB.py"))
    at.run(timeout=60)
    _no_exceptions(at)
    assert len(at.tabs) == 4


@pytest.mark.skipif(not HAS_DEEPSEEK_KEY, reason="DEEPSEEK_API_KEY not configured")
def test_gauge_assistant_demo_question_round_trip():
    import os
    at = AppTest.from_file(str(APP_DIR / "pages" / "1_🤖_GAUGE_Assistant.py"))
    at.run(timeout=60)
    # Inject user's own key via the sidebar text input
    at.text_input(key="assistant_api_key_deepseek-chat").set_value(os.environ["DEEPSEEK_API_KEY"]).run(timeout=30)
    demo_buttons = [b for b in at.button if "Erlotinib" in (b.label or "")]
    assert demo_buttons, "No Erlotinib demo button found"
    demo_buttons[0].click().run(timeout=90)
    _no_exceptions(at)
    assistant_messages = [cm for cm in at.chat_message if cm.name == "assistant"]
    assert assistant_messages
    assert any("predict_drug_response" in (exp.label or "") for exp in at.expander)
