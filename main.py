import json
from google import genai
import re
from datetime import datetime, timezone
import hashlib
from pathlib import Path
import argparse

def create_llm_log(stage, ticket_id, provider, model, prompt, output):
    try:
        with open("llm_calls.jsonl", "r") as file:
            data = json.load(file)
    except FileNotFoundError or json.JSONDecodeError:
        data = []

    timestamp = datetime.now(timezone.utc).isoformat()
    prompt_hash = hashlib.sha256(prompt.encode("utf-8")).hexdigest()

    Path("./artifacts").mkdir(parents=True, exist_ok=True)
    artifact_filename = f"{ticket_id}_{stage}.json"
    output_artifact_path = Path("./artifacts") / artifact_filename

    tracking_log = {
      "stage": stage,
      "ticket_id": ticket_id,
      "timestamp": timestamp,
      "provider": provider,
      "model": model,
      "prompt_hash": prompt_hash,
      "output_artifact": str(output_artifact_path),
    }
    data.append(tracking_log)

    with open(output_artifact_path, "w", encoding="utf-8") as f:
        if isinstance(output, (dict, list)):
            json.dump(output, f, indent=2)
        else:
            f.write(str(output))

    with open("llm_calls.jsonl", "w") as file:
        json.dump(data, file, indent=4)

    return tracking_log



def process_tickets(input_file):
    with open(input_file, "r", encoding="utf-8") as file:
        data = json.load(file)

    processed_tickets = []

    for ticket in data:
        id = ticket["ticket_id"]
        orig_text = ticket["customer_message"]
        cleaned_text = orig_text.strip()
        word_count = len(cleaned_text.split(" "))
        char_count = len(cleaned_text)

        processed_tickets.append({
            "id": id, 
            "original_text": orig_text,
            "cleaned_text": cleaned_text,
            "char_count": char_count,
            "word_count": word_count,
        })

    with open("preprocessed_tickets.json", "w") as file:
        file.write(str(json.dumps(processed_tickets, indent=4)))   
    return processed_tickets
    
def get_llm_classification(processed_tickets, label_schema):
    prompt="""
        You are an expert customer support operations and ticket-triage assistant. 

        ### TASK
        You will be provided with a JSON list of customer support tickets. Your task is to analyze each ticket, classify it into the correct category and urgency level based strictly on the provided label schema, calculate a confidence score, provide a concise reasoning summary, and determine if human review is required.

        ### ALLOWED LABEL SCHEMA
        {label_schema}

        ### OUTPUT FORMAT
        You must return a valid JSON array containing an object for each processed ticket, strictly matching this schema:

        [
        {
            "id": "string (ticket_id from input)",
            "message": "string (cleaned_text from input)",
            "expected_category": "(expected_category from input)",
            "expected_urgency": "(expected_urgency from input)",
            "category": "string (must be chosen EXACTLY from the allowed categories list)",
            "urgency": "string (must be chosen EXACTLY from the allowed urgency levels list)",
            "confidence": 0.0, 
            "reasoning_summary": "string (a concise 1-2 sentence explanation for the classification)",
            "needs_human_review": false 
        }
        ]

        ### RULES & CONSTRAINTS
        1. STRICT SCHEMA ADHERENCE: The `category` and `urgency` fields must strictly match values from the allowed label schema. Do not invent new categories.
        2. CONFIDENCE SCORE: Provide a floating-point confidence score between 0.0 and 1.0.
        3. HUMAN REVIEW FLAG: Set `needs_human_review` to `true` if confidence is below 0.8, if the message is highly ambiguous, or if it involves sensitive issues (like security or major billing disputes). Otherwise, set it to `false`.
        4. OUTPUT STRUCTURE: Return ONLY a valid JSON array. Do not include introductory text, explanations, or commentary outside of the JSON block.
    """
    client = genai.Client()

    response = client.models.generate_content(
        model="gemini-3.8-flash",
        config=genai.types.GenerateContentConfig(
            system_instruction=prompt
        ),
        contents=str(processed_tickets)
    )

    for ticket in processed_tickets:
        create_llm_log(
            stage="classification",
            ticket_id=ticket.get("id"),
            model="gemini-3.8-flash",
            output=response.text,
            prompt=prompt,
            provider="Google"
        )

    return response.text


def get_categorized(processed_tickets, label_schema):
    with open(label_schema, "r", encoding="utf-8") as file:
        schema = json.load(file)

    response = get_llm_classification(processed_tickets, schema)

    llm_tickets = json.loads(response)

    valid_categories = set(schema["categories"])
    valid_urgencies = set(schema["urgency_levels"])
    valid_tickets = []
    invalid_tickets = []
    for ticket in llm_tickets:
        if (ticket.get("category") in valid_categories and ticket.get("urgency") in valid_urgencies):
            valid_tickets.append(ticket)
        else:
            invalid_tickets.append(ticket)

    return valid_tickets, invalid_tickets

def route_tickets(valid, invalid):
    routing_summary = []
    for ticket in valid:
        confidence = ticket.get("confidence")
        if confidence < 0.65:
            route = "human_review"
            reasoning = "confidence too low"
        else:
            route = "auto_triage"
            reasoning = "high confidence"

        ticket["route"] = route
        ticket["routing_reason"] = reasoning
        routing_summary.append(ticket)
    for ticket in invalid:
        ticket["route"] = "human_review"
        ticket["routing_reason"] = "invalid ticket"
        routing_summary.append(ticket)

    with open("routing_decisions.json", "w") as file:
            file.write(str(json.dumps(routing_summary, indent=4)))
    return routing_summary

def generate_reply(tickets):
    prompt = """
    You are an automated support ticket triage and response system. You will receive input data containing metadata about a customer support ticket.
    Your task is to analyze the ticket metadata and produce a JSON object adhering strictly to the required output schema and business rules.

    ---

    ### OUTPUT JSON SCHEMA:
    Return a single JSON object with the following exact keys:
    {
    "ticket_id": "string",
    "expected_category": expected_category,
    "expected_urgency": expected_urgency
    "predicted_category": "string",
    "predicted_urgency": "string",
    "confidence": float,
    "route": "string",
    "customer_reply": "string | null",
    "internal_note": "string | null"
    }

    ---

    ### PROCESSING RULES:

    1. METADATA MAPPING:
    - "ticket_id": Map directly from the input "id" field.
    - "predicted_category": Map directly from the input "category" field.
    - "predicted_urgency": Map directly from the input "urgency" field.
    - "confidence": Map directly from the input "confidence" field.
    - "route": Map directly from the input "route" field.

    2. CONDITIONAL OUTPUT GENERATION (Choose SCENARIO A or B):

    SCENARIO A: If "route" is "auto_triage" (and "needs_human_review" is false):
    - "customer_reply": Generate a string matching ALL of these conditions:
        * Length: Exactly 2 to 4 sentences.
        * Content: Explicitly acknowledge the customer's issue matching "predicted_category" and the ticket details.
        * Tone: Match "predicted_urgency" (e.g., prioritized/empathetic for "high", helpful/professional for "medium" or "low").
        * Facts: DO NOT invent account-specific facts, status updates, or details not provided in the input.
        * Commitments: DO NOT promise specific actions, resolutions, or timelines not stated in the ticket.
    - "internal_note": Set explicitly to null.

    SCENARIO B: If "route" is "human_review" (or "needs_human_review" is true):
    - "customer_reply": Set explicitly to null.
    - "internal_note": Generate a string containing a brief internal note for support agents:
        * Explain why the ticket was escalated (e.g., low confidence, sensitive category, complex reasoning).
        * Summarize the core issue requiring manual intervention in 2-3 concise sentences.

    ---

    ### OUTPUT FORMAT:
    Return ONLY valid JSON. Do not include markdown code block formatting (like ```json), commentary, or preambles.
    """
    client = genai.Client()

    response = client.models.generate_content(
        model="gemini-3.8-flash",
        config=genai.types.GenerateContentConfig(
            system_instruction=prompt
        ),
        contents=str(tickets)
    )

    for ticket in tickets:
        create_llm_log(
            stage="classification",
            ticket_id=ticket.get("id"),
            model="gemini-3.8-flash",
            output=response.text,
            prompt=prompt,
            provider="Google"
        )

    with open("triage_results.json", "w") as file:
        file.write(str(response.text))
    return response.text
    

def create_evaluation_report(triage_results):
    total_tickets = len(triage_results)
    correct_categories = 0
    correct_urgencies = 0
    auto_triage_count = 0
    human_review_count = 0

    evaluation_comparison = []  
    records = json.loads(triage_results)
    for record in records:
        t_id = record["ticket_id"]

        # Check matches
        cat_match = record["expected_category"] == record["predicted_category"]
        urg_match = record["expected_urgency"] == record["predicted_urgency"]

        if cat_match:
            correct_categories += 1
        if urg_match:
            correct_urgencies += 1

        # Count routing types
        if record["route"] == "auto_triage":
            auto_triage_count += 1
        elif record["route"] == "human_review":
            human_review_count += 1

        evaluation_comparison.append({
            "ticket_id": t_id,
            "category_match": cat_match,
            "urgency_match": urg_match,
            "route": record['route'],
            "confidence": record['confidence']
        })

    category_accuracy = (correct_categories / total_tickets) * 100
    urgency_accuracy = (correct_urgencies / total_tickets) * 100

    overall_eval = {
        "total_tickets": total_tickets,
        "category_accuracy": f"{category_accuracy:.2f}%",
        "urgency_accuracy": f"{urgency_accuracy:.2f}%",
        "auto_triage_count": auto_triage_count,
        "human_review_count": human_review_count
    }

    with open("prediction_comparison.json", "w") as file:
        file.write(str(json.dumps(evaluation_comparison, indent=4)))

    with open("evaluation_report.json", "w") as file:
        file.write(str(json.dumps(overall_eval, indent=4)))




if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="An auto-triage LLM tool")
    parser.add_argument("-t", "--tickets", type=str, default="tickets.json", help="Location of the tickets file.")
    parser.add_argument("-s", "--schema", type=str, default="label_schema.json", help="Location of the schema file.")

    args = parser.parse_args()

    print("Processing raw tickets")
    tickets = process_tickets(args.tickets)
    print("Tickets processed")

    print("Categorizing tickets via LLM")
    valid, invalid = get_categorized(tickets, args.schema)
    print("Tickets categorized")

    print("Routing tickets")
    routing = route_tickets(valid, invalid)
    print("Tickets routed")

    print("Running auto-triage")
    reply = generate_reply(routing)
    print("Auto-triage complete")

    print("Generating evaluation report")
    create_evaluation_report(reply)
    print("Evaluation report generated")