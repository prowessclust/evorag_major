"""
persona_evolution — EvoRAG Evolutionary Persona Replacement System

Modules
-------
tracker   : Reads score_store.json and computes per-persona rolling averages.
detector  : Detects consistently underperforming personas.
mutator   : Generates candidate replacement personas.
replacer  : Executes the replacement and persists version history.
"""
