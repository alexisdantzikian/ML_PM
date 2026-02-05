import warnings
import sys
import logging
from typing import Dict, Optional

from skfin.dataloaders import DatasetLoader

warnings.filterwarnings("ignore", category=UserWarning, module="openpyxl")

logging.basicConfig(stream=sys.stdout, level=logging.INFO)
logger = logging.getLogger(__name__)


def load_kf_returns(
    filename: str = "12_Industry_Portfolios",
    force_reload: bool = False,
    cache_dir: Optional[str] = "data",
) -> Dict:
    """Load Ken French return data."""
    loader = DatasetLoader(cache_dir=cache_dir)
    return loader.load_kf_returns(filename, force_reload)



