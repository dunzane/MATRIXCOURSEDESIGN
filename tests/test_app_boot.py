import ast
import sys
from pathlib import Path

from PIL import Image
from streamlit.testing.v1 import AppTest


APP_PATH = Path(__file__).resolve().parents[1] / "app.py"
sys.path.insert(0, str(APP_PATH.parent))


def read_app_password():
    tree = ast.parse(APP_PATH.read_text(encoding="utf-8"))
    for node in tree.body:
        if not isinstance(node, ast.Assign):
            continue
        if any(isinstance(target, ast.Name) and target.id == "APP_PASSWORD" for target in node.targets):
            return ast.literal_eval(node.value)
    raise AssertionError("app.py 中未找到 APP_PASSWORD")


def assert_clean_run(app_test):
    assert not app_test.exception, [exception.message for exception in app_test.exception]


def find_by_label(elements, label):
    for element in elements:
        if element.label == label:
            return element
    raise AssertionError(f"未找到控件：{label}")


def test_password_navigation_state_and_reflection():
    at = AppTest.from_file(str(APP_PATH), default_timeout=60).run()
    assert_clean_run(at)

    assert any(title.value == "西电高等代数实验室" for title in at.title)
    at.text_input(key="password_input").input(read_app_password())
    at.button(key="password_submit").click().run()
    assert_clean_run(at)
    assert any(title.value == "西电高等代数实验室" for title in at.title)

    marker_image = Image.new("RGB", (4, 4), "#FAF9F5")
    marker_run = {
        "input": marker_image,
        "edited": marker_image,
        "output": marker_image,
        "params": {"风格模型": "状态保留测试"},
        "debug_history": {},
    }
    at.session_state["last_run"] = marker_run

    at.button(key="navigate_principle").click().run()
    assert_clean_run(at)
    assert any(title.value == "📚 原理介绍" for title in at.title)
    assert at.session_state["last_run"]["params"]["风格模型"] == "状态保留测试"

    at.button(key="navigate_manual").click().run()
    assert_clean_run(at)
    assert any(title.value == "📖 使用说明" for title in at.title)
    assert at.session_state["last_run"]["params"]["风格模型"] == "状态保留测试"

    at.button(key="navigate_workbench").click().run()
    assert_clean_run(at)
    assert any(title.value == "西电高等代数实验室" for title in at.title)
    assert at.session_state["last_run"]["params"]["风格模型"] == "状态保留测试"

    find_by_label(at.checkbox, "启用：几何变换 (仿射/透视)").set_value(True).run()
    assert_clean_run(at)
    find_by_label(at.checkbox, "沿 y = x 反射").set_value(True).run()
    assert_clean_run(at)
