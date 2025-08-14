# heuristic_rules.py

from core_act import generate_heuristic_rules

def clasificar_por_heuristica(app, title):
    subact, act = match_heuristic_rule(app, title)
    if subact:
        return subact, act
    else:
        return "Unclassified", None

category_to_activity = {
    "Communication": "Coordination",
    "Distraction": "Leisure",
    "Writing": "Content Creation",
    "Programming": "Development",
    "Personal Organization": "Planning",
    "General Browsing": "Exploration",
    "AI Consultation": "Technology",
    "Unclassified": None
}

# --- Auto-generated heuristic rules from core_act.py ---
HEURISTIC_RULES = generate_heuristic_rules()

# --- Additional manual heuristic rules for improved coverage ---
HEURISTIC_RULES += [
    {
        "keywords": ["tockler", "trackitems"],
        "subactivity": "Tracking work time",
        "activity": "Personal productivity"
    },
    {
        "keywords": ["tfm", "master", "tesis"],
        "subactivity": "Writing thesis",
        "activity": "Writing"
    },
    {
        "keywords": ["chatgpt", "openai"],
        "subactivity": "AI Consultation",
        "activity": "Technology"
    },
    {
        "keywords": ["guardar", "recientes", "documentos"],
        "subactivity": "File management",
        "activity": "Organizational matters"
    },
    {
        "keywords": ["dbeaver", "database", "db"],
        "subactivity": "Database work",
        "activity": "Programming"
    },
    {
        "keywords": ["calendar", "notion", "agenda"],
        "subactivity": "Using Notion or Calendar",
        "activity": "Planning"
    },
    {
        "keywords": ["whatsapp", "teams", "slack"],
        "subactivity": "Checking WhatsApp",
        "activity": "Communication"
    },
    {
        "keywords": ["firefox", "chrome", "browser", "mozilla"],
        "subactivity": "General Browsing",
        "activity": "Exploration"
    },
    {
        "keywords": ["word", "docs", "notas", "notepad"],
        "subactivity": "Writing in Word or Docs",
        "activity": "Writing"
    },
    {
        "keywords": ["online"],
        "subactivity": "Chatting online",
        "activity": "Communication"
    },
    {
        "keywords": ["spotify", "youtube", "netflix", "video"],
        "subactivity": "Unfocused browsing",
        "activity": "Distraction"
    },
    {
        "keywords": ["colonia", "vuelos", "viaje", "hotel", "booking"],
        "subactivity": "Planning travel",
        "activity": "Planning"
    }
]

def match_heuristic_rule(app, title):
    app = app.lower() if isinstance(app, str) else ""
    title = title.lower() if isinstance(title, str) else ""

    # print(f"[DEBUG] Checking heuristics for app='{app}' and title='{title}'")

    for rule in HEURISTIC_RULES:
        for keyword in rule["keywords"]:
            if keyword in app or keyword in title:
                # print(f"[DEBUG] Matched keyword '{keyword}' -> Subactivity: {rule['subactivity']}, Activity: {rule['activity']}")
                return rule["subactivity"], rule["activity"]
    return None, None