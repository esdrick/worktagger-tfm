def clasificar_eisenhower_por_heuristica(subactivity):
    if not subactivity or not isinstance(subactivity, str):
        return None

    subactivity = subactivity.lower()

    # Urgente e Importante
    if any(k in subactivity for k in [
        "urgent", "urgente", "asap", "client call", "incident", "emergency", "deadline", "production issue",
        "system down", "critical", "fix now", "security breach", "outage", "crash", "restore", "service unavailable",
        "soporte", "problema urgente", "bloqueado", "interrupción", "fallo", "issue", "high priority", "firefighting",
        "incident resolution", "incident report", "downtime", "server error", "cut off", "resolver ahora",
        "interrupción crítica", "urgente soporte", "problema grave", "error fatal", "colapso"
    ]):
        return "I: Urgent & Important"

    # No urgente pero Importante
    elif any(k in subactivity for k in [
        "planning", "planificación", "organizing", "organizando", "calendar", "agenda", "estrategia", "strategy",
        "notion", "research", "investigación", "design", "goal setting", "development", "refactoring", "debugging",
        "learning", "training", "formación", "estudio", "study", "read", "reading", "journaling", "writing",
        "writing thesis", "TFM", "master project", "documenting ideas", "setup", "project design", "code",
        "coding", "programming", "python", "streamlit", "sql", "database", "mongodb", "query", "performance",
        "productivity", "analytics", "time tracking", "tracking work", "tockler", "chatgpt", "openai", "copilot",
        "content creation", "blogging", "summary", "focus work", "proyecto final", "mejora continua",
        "documentación", "lectura técnica", "modelo productivo"
    ]):
        return "II: Not urgent but Important"

    # Urgente pero No importante
    elif any(k in subactivity for k in [
        "email", "correo", "mail", "meeting", "reunión", "zoom", "check", "follow-up", "reminder", "notification",
        "admin", "administración", "paperwork", "form", "ticket", "form submission", "report", "doc", "document",
        "revision", "status update", "teams", "slack", "whatsapp", "messages", "reply", "inbox", "responding",
        "call", "schedule call", "attend", "invitation", "calendar invite", "actualización", "asistencia",
        "responder mensaje", "revisar correo", "documento", "solicitud", "enviar archivo", "mensajes internos",
        "reportes", "seguimiento"
    ]):
        return "III: Urgent but Not important"

    # No urgente & No importante
    elif any(k in subactivity for k in [
        "scrolling", "youtube", "tiktok", "instagram", "facebook", "twitter", "video", "videos", "netflix",
        "spotify", "music", "reddit", "news", "browsing", "shopping", "memes", "shorts", "reels", "games",
        "gaming", "idle", "entertainment", "cat videos", "wasting time", "watching", "media", "leisure",
        "fun", "distracted", "break", "pausa", "ocio", "descanso", "chismes", "tendencias", "clips", "trending",
        "historias", "reel", "post", "likes", "comentarios", "cuentas", "videos graciosos"
    ]):
        return "IV: Not urgent & Not important"

    else:
        return None