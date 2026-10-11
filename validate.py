from datetime import datetime
import json
from pathlib import Path


def validate(predictions_path, schema_path, artifacts_dir) -> bool:
    pred_file = Path(predictions_path)
    schema_file = Path(schema_path)
    art_dir = Path(artifacts_dir)

    if not pred_file.exists():
        print(f"error: predictions file not found at '{predictions_path}'.")
        return False
    if not schema_file.exists():
        print(f"error: schema file not found at '{schema_path}'.")
        return False

    try:
        with open(pred_file, "r", encoding="utf-8") as f:
            predictions = json.load(f)
    except json.JSONDecodeError as e:
        print(f"error: '{predictions_path}' contains invalid JSON: {e}")
        return False

    try:
        with open(schema_file, "r", encoding="utf-8") as f:
            schema = json.load(f)
    except json.JSONDecodeError as e:
        print(f"error: '{schema_path}' contains invalid JSON: {e}")
        return False

    valid_categories = set(schema.get("categories", []))
    valid_urgencies = set(schema.get("urgency_levels", []))

    auto_triage_count = 0
    human_review_count = 0
    correct_categories = 0
    correct_urgencies = 0
    total_tickets = len(predictions)

    if total_tickets == 0:
        print("warning: predictions list is empty.")
        return False

    for idx, ticket in enumerate(predictions):
        ticket_id = ticket.get("ticket_id", f"Index {idx}")

        # Check 3: Routing decision exists
        route = ticket.get("route")
        if not route:
            print(f"[Ticket {ticket_id}] Missing 'route' decision.")
            has_errors = True
        else:
            if route == "auto_triage":
                auto_triage_count += 1
            elif route == "human_review":
                human_review_count += 1
            else:
                print(f"[Ticket {ticket_id}] Invalid route type: '{route}'.")
                has_errors = True

        # Check 4: auto_triage tickets have a customer reply
        customer_reply = ticket.get("customer_reply")
        if route == "auto_triage" and not customer_reply:
            print(
                f"[Ticket {ticket_id}] Route is 'auto_triage' but 'customer_reply'"
                " is missing or null."
            )
            has_errors = True

        # Check 5: human_review tickets have an internal note
        internal_note = ticket.get("internal_note")
        if route == "human_review" and not internal_note:
            print(
                f"[Ticket {ticket_id}] Route is 'human_review' but 'internal_note'"
                " is missing or null."
            )
            has_errors = True

        # Check 6: Predicted labels belong to allowed schema
        pred_cat = ticket.get("predicted_category")
        pred_urg = ticket.get("predicted_urgency")

        if pred_cat not in valid_categories:
            print(
                f"[Ticket {ticket_id}] Predicted category '{pred_cat}' is not in"
                f" allowed schema."
            )
            has_errors = True
        else:
            if (
                "expected_category" in ticket
                and ticket["expected_category"] == pred_cat
            ):
                correct_categories += 1

        if pred_urg not in valid_urgencies:
            print(
                f"[Ticket {ticket_id}] Predicted urgency '{pred_urg}' is not in"
                f" allowed schema."
            )
            has_errors = True
        else:
            if "expected_urgency" in ticket and ticket["expected_urgency"] == pred_urg:
                correct_urgencies += 1

        # Check if artifact exists on disk (if referenced)
        artifact_path = art_dir / f"{ticket_id}_triage.json"
        if art_dir.exists() and not artifact_path.exists():
            pass

    if not has_errors:
        print("All routing rules, replies, notes, and label schemas are valid.")

    try:
        category_accuracy = (correct_categories / total_tickets) * 100
        urgency_accuracy = (correct_urgencies / total_tickets) * 100

        print("\n--- Computed Evaluation Metrics ---")
        print(f"Total Tickets Evaluated: {total_tickets}")
        print(f"Category Accuracy:       {category_accuracy:.1f}%")
        print(f"Urgency Accuracy:        {urgency_accuracy:.1f}%")
        print(f"Total Auto Triage:       {auto_triage_count}")
        print(f"Total Human Review:      {human_review_count}")
        print("Evaluation metrics computed successfully.")
    except Exception as e:
        print(f"Error computing metrics: {e}")
        has_errors = True

    if has_errors:
        print("\nValidation FAILED with errors.")
        return False
    else:
        print("\nValidation PASSED successfully!")
        return True


if __name__ == "__main__":
  validate(predictions_path="./triage_results.json", schema_path="label_schema.json", artifacts_dir="./artifacts")