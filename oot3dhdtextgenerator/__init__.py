import pandas as pd

from oot3dhdtextgenerator.common import package_root

hanzi_frequency = pd.read_csv(
    f"{package_root}/data/characters.txt",
    sep="\t",
    names=["character", "frequency", "cumulative frequency"],
)
