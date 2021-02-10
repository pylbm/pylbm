import os
import tempfile

pylbm_tmp_dir = tempfile.TemporaryDirectory(prefix="pylbm_", dir=os.getcwd())