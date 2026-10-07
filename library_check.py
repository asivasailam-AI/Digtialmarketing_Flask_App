import sys
import flask
import numpy as np
import pandas as pd
import sklearn
import pickle


def print_library_versions():
    print("=" * 50)
    print("Python Environment Information")
    print("=" * 50)

    print(f"Python         : {sys.version}")
    print(f"Flask          : {flask.__version__}")
    print(f"NumPy          : {np.__version__}")
    print(f"Pandas         : {pd.__version__}")
    print(f"Scikit-learn   : {sklearn.__version__}")
    #print(f"gunicorn       : {gunicorn.__version__}")

    print("=" * 50)
print_library_versions()