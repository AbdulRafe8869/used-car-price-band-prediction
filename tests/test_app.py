"""The Streamlit app starts, renders every tab without an exception, and reacts to the inputs."""
import re
from pathlib import Path

from streamlit.testing.v1 import AppTest

APP = str(Path(__file__).resolve().parents[1] / "app" / "streamlit_app.py")


def test_app_runs_without_exception():
    at = AppTest.from_file(APP, default_timeout=120).run()
    assert not at.exception, [e.value for e in at.exception]


def _shown_price(at) -> int:
    """The rupee amount in the big 'Estimated selling price' card."""
    card = next(m.value for m in at.markdown if "Estimated selling price" in m.value)
    return int(re.search(r"Rs ([\d,]+)", card).group(1).replace(",", ""))


def test_a_newer_car_gets_a_higher_price_estimate():
    at = AppTest.from_file(APP, default_timeout=120).run()
    at.slider[0].set_value(2010).run()
    old = _shown_price(at)
    at.slider[0].set_value(2019).run()          # newer car
    assert not at.exception, [e.value for e in at.exception]
    assert _shown_price(at) > old
