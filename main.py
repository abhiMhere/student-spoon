"""
AI-Based Student Performance Prediction System
Console application using SQLite + scikit-learn Decision Tree.

IMPORTANT:
- synthetic_student_performance.csv is synthetic/demo data, not real student data.
- Model predictions are educational demonstrations, not reliable judgments of a real student's future.
"""

import sqlite3
import csv
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

BASE_DIR = Path(__file__).resolve().parent
DB_PATH = BASE_DIR / "students.db"
DATASET_PATH = BASE_DIR / "synthetic_student_performance.csv"
RANDOM_STATE = 42

FEATURES = ["previous_marks", "attendance_percent", "study_hours_per_day"]
TARGET = "performance_category"
CLASSES = ["Needs Support", "Average", "Strong"]


def create_synthetic_dataset(path=DATASET_PATH, rows=1200):
    """Create a reproducible, clearly synthetic training dataset if one is missing."""
    rng = np.random.default_rng(RANDOM_STATE)
    previous_marks = np.clip(rng.normal(62, 18, rows), 0, 100)
    attendance = np.clip(rng.normal(78, 15, rows), 0, 100)
    study_hours = np.clip(rng.gamma(shape=2.2, scale=1.5, size=rows), 0, 12)
    simulated_score = (
        0.52 * previous_marks
        + 0.27 * attendance
        + 2.0 * study_hours
        + rng.normal(0, 11, rows)
    )
    categories = np.where(
        simulated_score < 48, "Needs Support",
        np.where(simulated_score < 70, "Average", "Strong")
    )
    data = pd.DataFrame({
        "previous_marks": np.round(previous_marks, 1),
        "attendance_percent": np.round(attendance, 1),
        "study_hours_per_day": np.round(study_hours, 1),
        "performance_category": categories
    })
    data.to_csv(path, index=False)
    print(f"Created synthetic demonstration dataset: {path.name}")


def connect_db():
    conn = sqlite3.connect(DB_PATH)
    conn.execute("PRAGMA foreign_keys = ON")
    return conn


def setup_database(conn):
    """Create the table and safely add columns when an older students.db already exists."""
    conn.execute("""
        CREATE TABLE IF NOT EXISTS students (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL,
            course TEXT NOT NULL,
            semester INTEGER NOT NULL,
            marks REAL NOT NULL
        )
    """)
    existing = {
        row[1] for row in conn.execute("PRAGMA table_info(students)").fetchall()
    }
    additions = {
        "previous_marks": "REAL",
        "attendance_percent": "REAL",
        "study_hours_per_day": "REAL",
        "predicted_category": "TEXT",
        "prediction_date": "TEXT"
    }
    for column, sql_type in additions.items():
        if column not in existing:
            conn.execute(f"ALTER TABLE students ADD COLUMN {column} {sql_type}")
    conn.commit()


def load_model():
    """Train/test split and fit a Decision Tree using synthetic labelled examples."""
    if not DATASET_PATH.exists():
        create_synthetic_dataset()

    data = pd.read_csv(DATASET_PATH)
    required = set(FEATURES + [TARGET])
    missing = required - set(data.columns)
    if missing:
        raise ValueError(f"Dataset is missing required columns: {', '.join(sorted(missing))}")
    data = data.dropna(subset=FEATURES + [TARGET])
    if data.empty or data[TARGET].nunique() < 2:
        raise ValueError("Dataset must contain rows and at least two performance categories.")

    X = data[FEATURES].astype(float)
    y = data[TARGET].astype(str)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=RANDOM_STATE, stratify=y
    )
    model = DecisionTreeClassifier(
        max_depth=5, min_samples_leaf=8, random_state=RANDOM_STATE,
        class_weight="balanced"
    )
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "report": classification_report(
            y_test, y_pred, labels=CLASSES, zero_division=0
        ),
        "matrix": confusion_matrix(y_test, y_pred, labels=CLASSES),
        "labels": CLASSES,
        "train_rows": len(X_train),
        "test_rows": len(X_test),
        "dataset_rows": len(data)
    }
    return model, metrics


def read_text(prompt, allow_blank=False):
    while True:
        value = input(prompt).strip()
        if value or allow_blank:
            return value
        print("This field cannot be empty.")


def read_number(prompt, minimum, maximum, integer=False):
    while True:
        raw = input(prompt).strip()
        try:
            value = int(raw) if integer else float(raw)
            if not np.isfinite(value):
                raise ValueError
            if not minimum <= value <= maximum:
                print(f"Enter a value from {minimum} to {maximum}.")
                continue
            return value
        except ValueError:
            kind = "whole number" if integer else "number"
            print(f"Please enter a valid {kind} between {minimum} and {maximum}.")


def get_student_inputs():
    print("\nEnter academic details (these are model inputs):")
    previous = read_number("Previous marks (0-100): ", 0, 100)
    attendance = read_number("Attendance percentage (0-100): ", 0, 100)
    hours = read_number("Average study hours per day (0-24): ", 0, 24)
    return previous, attendance, hours


def add_student(conn):
    print("\n--- Add Student ---")
    name = read_text("Student name: ")
    course = read_text("Course: ")
    semester = read_number("Semester (1-20): ", 1, 20, integer=True)
    marks = read_number("Current marks (0-100): ", 0, 100)
    previous, attendance, hours = get_student_inputs()
    conn.execute("""
        INSERT INTO students
        (name, course, semester, marks, previous_marks, attendance_percent, study_hours_per_day)
        VALUES (?, ?, ?, ?, ?, ?, ?)
    """, (name, course, semester, marks, previous, attendance, hours))
    conn.commit()
    print("Student added successfully.")


def view_students(conn):
    rows = conn.execute("""
        SELECT id, name, course, semester, marks, previous_marks,
               attendance_percent, study_hours_per_day, predicted_category
        FROM students ORDER BY id
    """).fetchall()
    if not rows:
        print("\nNo student records found.")
        return
    print("\nID | Name | Course | Sem | Marks | Prev. | Attend% | Study h/day | Prediction")
    print("-" * 100)
    for r in rows:
        print(f"{r[0]} | {r[1]} | {r[2]} | {r[3]} | {r[4]} | "
              f"{display(r[5])} | {display(r[6])} | {display(r[7])} | {display(r[8])}")


def display(value):
    return "-" if value is None else str(value)


def search_student(conn):
    query = read_text("Enter a name or part of a name: ")
    rows = conn.execute("""
        SELECT id, name, course, semester, marks, previous_marks,
               attendance_percent, study_hours_per_day, predicted_category
        FROM students WHERE name LIKE ? ORDER BY id
    """, (f"%{query}%",)).fetchall()
    if not rows:
        print("No matching student found.")
        return
    for r in rows:
        print(f"\nID: {r[0]}\nName: {r[1]}\nCourse: {r[2]}\nSemester: {r[3]}"
              f"\nCurrent marks: {r[4]}\nPrevious marks: {display(r[5])}"
              f"\nAttendance: {display(r[6])}%\nStudy hours/day: {display(r[7])}"
              f"\nLast prediction: {display(r[8])}")


def update_student(conn):
    student_id = read_number("Student ID to update: ", 1, 2_000_000_000, integer=True)
    row = conn.execute("SELECT id FROM students WHERE id = ?", (student_id,)).fetchone()
    if not row:
        print("Student not found.")
        return
    print("Enter the updated details.")
    name = read_text("Name: ")
    course = read_text("Course: ")
    semester = read_number("Semester (1-20): ", 1, 20, integer=True)
    marks = read_number("Current marks (0-100): ", 0, 100)
    previous, attendance, hours = get_student_inputs()
    conn.execute("""
        UPDATE students SET name=?, course=?, semester=?, marks=?,
            previous_marks=?, attendance_percent=?, study_hours_per_day=?,
            predicted_category=NULL, prediction_date=NULL
        WHERE id=?
    """, (name, course, semester, marks, previous, attendance, hours, student_id))
    conn.commit()
    print("Student updated. Run prediction again to refresh the prediction.")


def delete_student(conn):
    student_id = read_number("Student ID to delete: ", 1, 2_000_000_000, integer=True)
    cursor = conn.execute("DELETE FROM students WHERE id = ?", (student_id,))
    conn.commit()
    print("Student deleted." if cursor.rowcount else "Student not found.")


def predict_values(model, previous, attendance, hours):
    row = pd.DataFrame([{
        "previous_marks": previous,
        "attendance_percent": attendance,
        "study_hours_per_day": hours
    }])
    category = str(model.predict(row)[0])
    probabilities = model.predict_proba(row)[0]
    confidence = float(max(probabilities))
    return category, confidence


def make_recommendations(previous, attendance, hours, category):
    suggestions = []
    if attendance < 75:
        suggestions.append("Work toward improving attendance and discuss missed topics with the instructor.")
    if previous < 50:
        suggestions.append("Review foundational topics and practise a few questions each day.")
    if hours < 1.5:
        suggestions.append("Try a consistent, manageable daily study session with breaks.")
    if not suggestions:
        suggestions.append("Maintain your routine, practise regularly, and review mistakes.")
    suggestions.append(
        "Treat this as a learning aid only; discuss academic concerns with a teacher."
    )
    return suggestions


def predict_existing_student(conn, model):
    student_id = read_number("Student ID to predict: ", 1, 2_000_000_000, integer=True)
    row = conn.execute("""
        SELECT id, name, previous_marks, attendance_percent, study_hours_per_day
        FROM students WHERE id=?
    """, (student_id,)).fetchone()
    if not row:
        print("Student not found.")
        return
    if any(v is None for v in row[2:]):
        print("This older record has no model inputs. Update the student record first.")
        return
    category, confidence = predict_values(model, row[2], row[3], row[4])
    timestamp = datetime.now().isoformat(timespec="seconds")
    conn.execute("""
        UPDATE students SET predicted_category=?, prediction_date=? WHERE id=?
    """, (category, timestamp, student_id))
    conn.commit()
    print(f"\nPrediction for {row[1]}: {category}")
    print(f"Model probability for predicted category: {confidence:.1%}")
    print("This probability is not a guarantee of real-world performance.")
    print("Suggestions:")
    for suggestion in make_recommendations(row[2], row[3], row[4], category):
        print(f"- {suggestion}")


def predict_new_student(model):
    previous, attendance, hours = get_student_inputs()
    category, confidence = predict_values(model, previous, attendance, hours)
    print(f"\nPredicted performance category: {category}")
    print(f"Model probability for predicted category: {confidence:.1%}")
    print("This probability is not a guarantee of real-world performance.")
    print("Suggestions:")
    for suggestion in make_recommendations(previous, attendance, hours, category):
        print(f"- {suggestion}")


def show_evaluation(metrics):
    print("\n--- Model Evaluation (held-out test set) ---")
    print("Dataset: synthetic demonstration data")
    print(f"Rows used: {metrics['dataset_rows']}")
    print(f"Training rows: {metrics['train_rows']}")
    print(f"Test rows: {metrics['test_rows']}")
    print(f"Accuracy: {metrics['accuracy']:.3f} ({metrics['accuracy']:.1%})")
    print("\nClassification report:")
    print(metrics["report"])
    print("Confusion matrix (actual rows, predicted columns)")
    print("Class order:", " | ".join(metrics["labels"]))
    print(metrics["matrix"])
    print("\nEvaluation measures performance on a held-out synthetic test split.")
    print("It does not establish accuracy on real students.")


def main():
    if not DATASET_PATH.exists():
        create_synthetic_dataset()
    conn = connect_db()
    setup_database(conn)
    try:
        model, metrics = load_model()
        print("=" * 58)
        print(" AI-BASED STUDENT PERFORMANCE PREDICTION SYSTEM")
        print("=" * 58)
        print("ML model: Decision Tree Classifier")
        print("Training source: SYNTHETIC demonstration dataset")
        print("This app is for educational demonstration, not high-stakes decisions.")
        while True:
            print("\n============== MAIN MENU ==============")
            print("1. Add student")
            print("2. View all students")
            print("3. Search student")
            print("4. Update student")
            print("5. Delete student")
            print("6. Predict performance for a saved student")
            print("7. Predict performance for new inputs (not saved)")
            print("8. Evaluate the ML model")
            print("9. Exit")
            choice = input("Choose an option (1-9): ").strip()
            try:
                if choice == "1":
                    add_student(conn)
                elif choice == "2":
                    view_students(conn)
                elif choice == "3":
                    search_student(conn)
                elif choice == "4":
                    update_student(conn)
                elif choice == "5":
                    delete_student(conn)
                elif choice == "6":
                    predict_existing_student(conn, model)
                elif choice == "7":
                    predict_new_student(model)
                elif choice == "8":
                    show_evaluation(metrics)
                elif choice == "9":
                    print("Thank you for using the system.")
                    break
                else:
                    print("Invalid choice. Enter a number from 1 to 9.")
            except (sqlite3.Error, ValueError) as exc:
                print(f"Operation could not be completed: {exc}")
    finally:
        conn.close()


if __name__ == "__main__":
    main()
