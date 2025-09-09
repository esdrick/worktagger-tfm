def calculate_productive_time(df):
    """Función centralizada para calcular tiempo productivo de forma consistente"""
    if df is None or df.empty or 'Eisenhower' not in df.columns:
        return 0
    
    productive_categories = ['I: Urgent & Important', 'II: Not urgent but Important']
    productive_time = df[df['Eisenhower'].isin(productive_categories)]['Duration'].sum() / 60
    return productive_time