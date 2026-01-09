from pycallgraph2 import PyCallGraph
from pycallgraph2 import Config
from pycallgraph2.output import GraphvizOutput
from pycallgraph2.globbing_filter import GlobbingFilter

from Scripts.adversarial_tts_classes import main  # Import your actual main function


def run_viz():
    config = Config()

    # === THE MAGIC PART: HIDING NOISE ===
    config.trace_filter = GlobbingFilter(exclude=[
        'pycallgraph2.*',
        'tqdm*',  # Hides all tqdm calls
        'torch.*',  # Hides internal torch mechanics
        'numpy.*',
        'os.*',  # Hides path joins, etc.
        'builtins.print',  # Hides print statements
        '*.format',  # Hides string formatting
        'logging.*',  # Hides loggers
    ])

    graphviz = GraphvizOutput()
    graphviz.output_file = 'clean_architecture_flow.png'

    with PyCallGraph(output=graphviz, config=config):
        main()

if __name__ == "__main__":
    run_viz()