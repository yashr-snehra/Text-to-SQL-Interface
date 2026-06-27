"""Adaptive few-shot prompting.

We detect the question's categories and show worked examples of those categories
(all over one shared toy schema, distinct from any real target schema so the
model learns the *pattern*, not the answer). A question that is both a JOIN and
an aggregation gets examples of each -- that combo is where LLMs fail most.

JOIN detection is *schema-aware*: if the question mentions columns or sampled
values from two different tables (e.g. "tracks by Radiohead" -> track + artist),
it is flagged as a join even with no join keyword present.
"""
import re

from .schema import _loose_match

INSTRUCTION = (
    "You are an expert data analyst who writes SQLite queries. Given a database "
    "schema and a question, output a single valid SQLite query that answers it. "
    "Use the foreign keys shown in the schema to join tables. Qualify columns "
    "with their table when a name is ambiguous. Prefer values shown as 'e.g.' "
    "hints when filtering. "
    "When the question asks for a total, sum, average, count, maximum, or "
    "minimum, apply the matching aggregate function (SUM, AVG, COUNT, MAX, MIN) "
    "rather than selecting the raw column. "
    "Use SELECT DISTINCT when the question asks which or what values appear, "
    "exist, or are present, so duplicate rows are removed. "
    "Output ONLY the SQL query -- no explanation, no markdown fences, no comments."
)

EXAMPLE_SCHEMA = (
    'Table "student" has columns: id (INT), name (TEXT), age (INT), dept_id (INT). '
    "Primary key: id. Column dept_id references department.id.\n"
    'Table "department" has columns: id (INT), name (TEXT), budget (INT). '
    "Primary key: id.\n"
    'Table "enrollment" has columns: student_id (INT), course (TEXT), grade (INT). '
    "Column student_id references student.id."
)

EXAMPLES = {
    "simple": [
        ("What are the names of all students?", "SELECT name FROM student"),
        ("List the names of students older than 20.", "SELECT name FROM student WHERE age > 20"),
        ("Show all department names.", "SELECT name FROM department"),
        ("What is the age of the student named Alice?", "SELECT age FROM student WHERE name = 'Alice'"),
        ("List all distinct courses.", "SELECT DISTINCT course FROM enrollment"),
    ],
    "aggregation": [
        ("How many students are there?", "SELECT count(*) FROM student"),
        ("What is the average age of students?", "SELECT avg(age) FROM student"),
        ("What is the total budget of all departments?", "SELECT sum(budget) FROM department"),
        ("How many students are in each department?",
         "SELECT dept_id, count(*) FROM student GROUP BY dept_id"),
        ("What is the maximum age among students?", "SELECT max(age) FROM student"),
    ],
    "join": [
        ("List the names of students and their department names.",
         "SELECT T1.name, T2.name FROM student AS T1 JOIN department AS T2 ON T1.dept_id = T2.id"),
        ("What are the names of students enrolled in 'Math'?",
         "SELECT T1.name FROM student AS T1 JOIN enrollment AS T2 ON T1.id = T2.student_id "
         "WHERE T2.course = 'Math'"),
        # "what ... appear/are taken" -> DISTINCT across the join.
        ("What courses are taken by students in department 1?",
         "SELECT DISTINCT T2.course FROM student AS T1 JOIN enrollment AS T2 "
         "ON T1.id = T2.student_id WHERE T1.dept_id = 1"),
        ("List department names along with the number of students in them.",
         "SELECT T2.name, count(*) FROM student AS T1 JOIN department AS T2 ON T1.dept_id = T2.id "
         "GROUP BY T2.id"),
        # Multi-hop bridge: chain two joins through an intermediate table.
        ("What courses are taken by students in the 'Physics' department?",
         "SELECT T3.course FROM department AS T1 JOIN student AS T2 ON T2.dept_id = T1.id "
         "JOIN enrollment AS T3 ON T3.student_id = T2.id WHERE T1.name = 'Physics'"),
    ],
    "subquery": [
        ("List students older than the average age.",
         "SELECT name FROM student WHERE age > (SELECT avg(age) FROM student)"),
        ("What is the name of the department with the highest budget?",
         "SELECT name FROM department WHERE budget = (SELECT max(budget) FROM department)"),
        ("Find the names of students not enrolled in any course.",
         "SELECT name FROM student WHERE id NOT IN (SELECT student_id FROM enrollment)"),
        ("List students whose age is greater than the oldest student in department 1.",
         "SELECT name FROM student WHERE age > (SELECT max(age) FROM student WHERE dept_id = 1)"),
        ("List the names of departments that have no students.",
         "SELECT name FROM department WHERE id NOT IN (SELECT dept_id FROM student)"),
    ],
}

# Keyword cues, checked in priority order.
_CUES = [
    ("subquery", ["than the average", "than any", "than every", "than all",
                  "not in", "without", "that have no", "do not", "does not",
                  "never", "no students", "none of", "more than the"]),
    ("aggregation", ["how many", "number of", "count", "average", "avg", "total",
                     "sum", "maximum", "minimum", "highest", "lowest", "most", "least"]),
    ("join", ["each", "their", "along with", "for every", "and the name", "both",
              "as well as", "together with", "enrolled in", "belongs to"]),
]


def detect_categories(question, schema=None):
    """Return all matching categories. Adds 'join' when the question spans two
    tables in the schema, even without a join keyword."""
    q = question.lower()
    cats = [category for category, cues in _CUES if any(cue in q for cue in cues)]

    tables = getattr(schema, "tables", None)
    if tables:
        qtokens = set(re.findall(r"[a-z0-9]+", q))
        hit = set()
        for table in tables:
            if _loose_match(table.name, qtokens):
                hit.add(table.name)
            for col in table.columns:
                if _loose_match(col.name, qtokens):
                    hit.add(table.name)
                for v in col.samples:
                    if isinstance(v, str) and len(v) > 2 and v.lower() in q:
                        hit.add(table.name)
        if len(hit) >= 2 and "join" not in cats:
            cats.append("join")

    return cats or ["simple"]


def select_examples(categories, k=5):
    """Interleave examples across the detected categories, up to k total."""
    selected = []
    idx = 0
    while len(selected) < k:
        progressed = False
        for category in categories:
            pool = EXAMPLES.get(category, [])
            if idx < len(pool) and pool[idx] not in selected:
                selected.append(pool[idx])
                progressed = True
                if len(selected) >= k:
                    break
        idx += 1
        if not progressed:
            break
    return selected[:k]


def _resolve_desc(schema, question, max_tables):
    """Accept either a Schema (preferred) or a pre-rendered description string."""
    if hasattr(schema, "describe"):
        return schema.describe(question=question, max_tables=max_tables)
    return schema


def build_prompt(question, schema, few_shot=True, max_tables=10,
                 examples=None, value_hints=None, schema_desc=None):
    """Build the generation prompt.

    examples     : optional retrieved (question, sql) pairs -> dynamic few-shot.
                   When None, schema-aware category examples are used instead.
    value_hints  : optional ["table.col = 'value'", ...] grounding WHERE literals.
    schema_desc  : optional pre-rendered description; pass it to avoid re-pruning
                   (and re-embedding) the schema when the caller already has it.
    """
    if schema_desc is None:
        schema_desc = _resolve_desc(schema, question, max_tables)
    parts = [INSTRUCTION, ""]
    if few_shot:
        if examples is not None:
            parts += ["Examples from other databases:", ""]
            for q, sql in examples:
                parts += [f"Question: {q}", f"SQL: {' '.join(sql.split())}", ""]
        else:
            chosen = select_examples(detect_categories(question, schema))
            parts += ["Examples (using a different schema):", "Schema:", EXAMPLE_SCHEMA, ""]
            for q, sql in chosen:
                parts += [f"Question: {q}", f"SQL: {sql}", ""]
        parts.append("Now answer for the real schema below.")
    parts += ["Schema:", schema_desc]
    if value_hints:
        parts.append("Values present in the database that may match the question:")
        parts += [f"  {h}" for h in value_hints]
    parts += ["", f"Question: {question}", "SQL:"]
    return "\n".join(parts)


def build_retry_prompt(question, schema_desc, bad_sql, error):
    """Error-feedback prompt: hand the failed SQL and its problem back to the model."""
    return "\n".join([
        INSTRUCTION, "",
        "Schema:", schema_desc, "",
        f"Question: {question}", "",
        "Your previous SQL was not acceptable:",
        bad_sql,
        f"Problem: {error}",
        "Write a corrected single SQLite query. Output ONLY the SQL.",
        "SQL:",
    ])


def build_verify_prompt(question, schema_desc, candidate_sql):
    """Self-check prompt: ask the model to confirm or repair its own query."""
    return "\n".join([
        INSTRUCTION, "",
        "Schema:", schema_desc, "",
        f"Question: {question}", "",
        "A candidate SQLite query is:",
        candidate_sql,
        "Check it against the schema and question. If it correctly answers the "
        "question, repeat it exactly. If it is wrong (wrong tables, joins, "
        "columns, or filters), output a corrected single SQLite query. Output "
        "ONLY the SQL.",
        "SQL:",
    ])
