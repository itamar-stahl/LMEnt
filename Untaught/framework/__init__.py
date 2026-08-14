"""Untaught framework, split by the machine each half runs on.

``client_node`` needs Elasticsearch and runs on the login node; ``gpu_node``
runs inside the SLURM job and never touches the index. The blacklist artifact
is the only thing that crosses between them, and imports only ever point
client_node -> gpu_node.
"""
