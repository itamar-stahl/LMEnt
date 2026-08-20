"""Untaught, login-node half: everything that needs Elasticsearch.

Resolving which chunks mention an entity happens here, before a job is
submitted, because the GPU nodes cannot reach the index. The result travels to
the compute node as the artifact file that ``node`` reads.
"""
