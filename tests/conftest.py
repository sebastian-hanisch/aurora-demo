import os
import sys
from pathlib import Path

# Die Matrizen der Emitter sind klein (bis 280 × 280): mehrere BLAS-Threads bringen nichts und bremsen in Containern mit wenigen zugewiesenen Kernen die Eigenzerlegung stark (Docker: 705 s statt etwa 60 s).
# Die Variablen müssen vor dem ersten Import von NumPy stehen.
for _name in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_name, "1")

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
