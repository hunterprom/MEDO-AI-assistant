"""MEDO Maker Studio — generate real engineering artifacts by writing library
code and executing it in a sandbox (electrical schematics, printable 3D, code).

S1 here is the shared, domain-agnostic engine: intent → code → sandboxed execute
→ self-correct → check → version. The three domain skills (schematic / 3D / code)
plug into it via ``studio.domain.Domain``.
"""
