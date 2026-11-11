
import sqlite3
from pathlib import Path
from datetime import datetime
from contextlib import contextmanager


# --------------------------------------------------
# DATABASE CONFIGURATION
# --------------------------------------------------

DB_PATH = Path(__file__).resolve().parent / "cineforge.db"

DB_TIMEOUT = 30


# --------------------------------------------------
# DATABASE CONNECTION
# --------------------------------------------------

@contextmanager
def connect():
    """
    Opens a database connection and guarantees cleanup.
    Automatically commits successful operations and
    rolls back failed transactions.
    """

    conn = sqlite3.connect(
        DB_PATH,
        timeout=DB_TIMEOUT
    )

    conn.row_factory = sqlite3.Row

    try:
        conn.execute("PRAGMA busy_timeout = 30000")
        yield conn
        conn.commit()

    except Exception:
        conn.rollback()
        raise

    finally:
        conn.close()


# --------------------------------------------------
# DATABASE INITIALIZATION & MIGRATIONS
# --------------------------------------------------

def init_db():
    """
    Creates the projects table and safely adds missing
    columns to existing CineForge AI databases.
    """

    DB_PATH.parent.mkdir(parents=True, exist_ok=True)

    with connect() as conn:

        conn.execute("""
            CREATE TABLE IF NOT EXISTS projects (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                title TEXT NOT NULL,
                idea TEXT NOT NULL,
                genre TEXT NOT NULL,
                duration TEXT NOT NULL,
                status TEXT NOT NULL DEFAULT 'Draft',
                script TEXT DEFAULT '',
                scenes TEXT DEFAULT '',
                error TEXT DEFAULT '',
                voice_path TEXT DEFAULT '',
                subtitle_path TEXT DEFAULT '',
                visual_video_path TEXT DEFAULT '',
                final_video_path TEXT DEFAULT '',
                created_at TEXT NOT NULL
            )
        """)

        existing_columns = {
            row["name"]
            for row in conn.execute(
                "PRAGMA table_info(projects)"
            ).fetchall()
        }

        migrations = {
            "status": "TEXT NOT NULL DEFAULT 'Draft'",
            "script": "TEXT DEFAULT ''",
            "scenes": "TEXT DEFAULT ''",
            "error": "TEXT DEFAULT ''",
            "voice_path": "TEXT DEFAULT ''",
            "subtitle_path": "TEXT DEFAULT ''",
            "visual_video_path": "TEXT DEFAULT ''",
            "final_video_path": "TEXT DEFAULT ''",
            "created_at": "TEXT DEFAULT ''"
        }

        for column, definition in migrations.items():

            if column not in existing_columns:

                conn.execute(
                    f"ALTER TABLE projects ADD COLUMN {column} {definition}"
                )

        # Add timestamps to older records that do not have one.
        conn.execute("""
            UPDATE projects
            SET created_at = ?
            WHERE created_at IS NULL OR created_at = ''
        """, (
            datetime.now().strftime("%Y-%m-%d %H:%M"),
        ))

        # Improve project listing performance.
        conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_projects_created
            ON projects(id DESC)
        """)


# --------------------------------------------------
# CREATE PROJECT
# --------------------------------------------------

def create_project(title, idea, genre, duration):
    """
    Creates a new CineForge AI production project.
    Returns the newly generated project ID.
    """

    title = str(title).strip()
    idea = str(idea).strip()
    genre = str(genre).strip()
    duration = str(duration).strip()

    if not title:
        raise ValueError("Project title cannot be empty.")

    if not idea:
        raise ValueError("Creative idea cannot be empty.")

    if not genre:
        raise ValueError("Project genre cannot be empty.")

    if not duration:
        raise ValueError("Project duration cannot be empty.")

    created_at = datetime.now().strftime(
        "%Y-%m-%d %H:%M"
    )

    with connect() as conn:

        cursor = conn.execute("""
            INSERT INTO projects (
                title,
                idea,
                genre,
                duration,
                status,
                created_at
            )
            VALUES (?, ?, ?, ?, ?, ?)
        """, (
            title,
            idea,
            genre,
            duration,
            "Generating script",
            created_at
        ))

        return cursor.lastrowid


# --------------------------------------------------
# GET ALL PROJECTS
# --------------------------------------------------

def get_projects():
    """
    Returns all projects, newest first.
    """

    with connect() as conn:

        rows = conn.execute("""
            SELECT *
            FROM projects
            ORDER BY id DESC
        """).fetchall()

        return [dict(row) for row in rows]


# --------------------------------------------------
# GET SINGLE PROJECT
# --------------------------------------------------

def get_project(project_id):
    """
    Retrieves a project using its database ID.
    Returns None if the project does not exist.
    """

    with connect() as conn:

        row = conn.execute("""
            SELECT *
            FROM projects
            WHERE id = ?
        """, (project_id,)).fetchone()

        return dict(row) if row else None


# --------------------------------------------------
# UPDATE PROJECT
# --------------------------------------------------

def update_project(project_id, **fields):
    """
    Safely updates project information.

    Only approved database fields can be modified.
    Uses parameterized SQL values to prevent SQL injection.
    """

    if not fields:
        return

    allowed_fields = {
        "title",
        "idea",
        "genre",
        "duration",
        "status",
        "script",
        "scenes",
        "error",
        "voice_path",
        "subtitle_path",
        "visual_video_path",
        "final_video_path"
    }

    invalid_fields = set(fields) - allowed_fields

    if invalid_fields:
        raise ValueError(
            f"Invalid project fields: {', '.join(sorted(invalid_fields))}"
        )

    assignments = ", ".join(
        f"{field} = ?" for field in fields
    )

    values = list(fields.values())
    values.append(project_id)

    with connect() as conn:

        cursor = conn.execute(
            f"""
            UPDATE projects
            SET {assignments}
            WHERE id = ?
            """,
            values
        )

        if cursor.rowcount == 0:
            raise ValueError(
                f"Project with ID {project_id} does not exist."
            )


# --------------------------------------------------
# DELETE PROJECT
# --------------------------------------------------

def delete_project(project_id):
    """
    Deletes a project record from the database.

    Generated files are not automatically deleted.
    """

    with connect() as conn:

        cursor = conn.execute("""
            DELETE FROM projects
            WHERE id = ?
        """, (project_id,))

        return cursor.rowcount > 0


# --------------------------------------------------
# DATABASE HEALTH CHECK
# --------------------------------------------------

def check_database():
    """
    Checks whether the SQLite database is accessible.
    """

    try:

        with connect() as conn:

            result = conn.execute(
                "PRAGMA integrity_check"
            ).fetchone()

            return result[0] == "ok"

    except sqlite3.Error:
        return False
