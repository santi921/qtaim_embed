from setuptools import setup, find_packages
from pathlib import Path


setup(
    name="qtaim_embed",
    version="0.0.0",
    packages=find_packages(),
    author="Santiago Vargas",
    author_email="santiagovargas921@gmail.com",
    zip_safe=False,
    scripts=[
        "./qtaim_embed/scripts/train/bayes_opt_graph.py",
        "./qtaim_embed/scripts/train/train_qtaim_graph.py",
        "./qtaim_embed/scripts/train/bayes_opt_node.py",
        "./qtaim_embed/scripts/train/train_qtaim_node.py",
        "./qtaim_embed/scripts/train/bayes_opt_graph_classifier.py",
        "./qtaim_embed/scripts/train/train_qtaim_graph_classifier.py",
        "./qtaim_embed/scripts/vis/data_summary.py",
        "./qtaim_embed/scripts/helpers/mol2lmdb.py"
    ],
)
