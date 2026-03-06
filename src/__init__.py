"""
Expose the QuickModel class and the run_quick_model super-function at the
package level, allowing imports like:

    from src import QuickModel
    from src import run_quick_model
"""
from .quick_model import QuickModel, run_quick_model