from pathlib import Path
import sys
import importlib.util


CURRENT_DIR = Path(__file__).resolve().parent

TARGET = CURRENT_DIR / "30_0_external_matched_common_v32.py"


MODULE_NAME = "external_matched_common_v32_impl"


if str(CURRENT_DIR) not in sys.path:
    sys.path.insert(0, str(CURRENT_DIR))


spec = importlib.util.spec_from_file_location(
    MODULE_NAME,
    TARGET
)

module = importlib.util.module_from_spec(spec)


# dataclass需要这里
sys.modules[MODULE_NAME] = module


spec.loader.exec_module(module)


FrozenProjectScorer = module.FrozenProjectScorer
RequestBudgetOracleBroker = module.RequestBudgetOracleBroker