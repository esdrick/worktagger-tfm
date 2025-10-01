import json
import streamlit as st
import pandas as pd 
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime


class CustomHeuristicManager:
    """Gestor de reglas heurísticas personalizables por el usuario - Session State"""
    
    def __init__(self):
        # Inicializar en session_state si no existe
        if "user_heuristic_rules" not in st.session_state:
            st.session_state.user_heuristic_rules = []
    
    @property
    def user_rules(self) -> List[Dict[str, Any]]:
        """Acceso a las reglas desde session_state"""
        return st.session_state.user_heuristic_rules
    
    def add_rule(self, keywords: List[str], subactivity: str, activity: str, description: str = "") -> bool:
        """Añade una nueva regla heurística"""
        new_id = max([rule["id"] for rule in self.user_rules], default=0) + 1
        
        new_rule = {
            "id": new_id,
            "keywords": [kw.strip().lower() for kw in keywords if kw.strip()],
            "subactivity": subactivity.strip(),
            "activity": activity.strip(),
            "description": description.strip(),
            "active": True,
            "created_by": "user",
            "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }
        
        # Validación básica
        if not new_rule["keywords"] or not new_rule["subactivity"] or not new_rule["activity"]:
            return False
        
        st.session_state.user_heuristic_rules.append(new_rule)
        
        # Sincronizar categorías personalizadas
        self._sync_custom_categories()
        
        return True
    
    def edit_rule(self, rule_id: int, keywords: List[str], subactivity: str, activity: str, description: str = "", active: bool = True) -> bool:
        """Edita una regla existente"""
        for rule in self.user_rules:
            if rule["id"] == rule_id:
                rule["keywords"] = [kw.strip().lower() for kw in keywords if kw.strip()]
                rule["subactivity"] = subactivity.strip()
                rule["activity"] = activity.strip()
                rule["description"] = description.strip()
                rule["active"] = active
                return True
        return False
    
    def delete_rule(self, rule_id: int) -> bool:
        """Elimina una regla"""
        st.session_state.user_heuristic_rules = [
            rule for rule in self.user_rules if rule["id"] != rule_id
        ]
        return True
    
    def toggle_rule(self, rule_id: int) -> bool:
        """Activa/desactiva una regla"""
        for rule in self.user_rules:
            if rule["id"] == rule_id:
                rule["active"] = not rule["active"]
                return True
        return False
    
    def get_active_rules(self) -> List[Dict[str, Any]]:
        """Devuelve solo las reglas activas"""
        return [rule for rule in self.user_rules if rule.get("active", True)]
    
    def match_heuristic_rule(self, app: str, title: str) -> Tuple[Optional[str], Optional[str]]:
        """Busca coincidencias en las reglas del usuario y las predeterminadas"""
        app = app.lower() if isinstance(app, str) else ""
        title = title.lower() if isinstance(title, str) else ""
        combined_text = f"{app} {title}"
        
        # Primero busca en las reglas del usuario (mayor prioridad)
        for rule in self.get_active_rules():
            for keyword in rule["keywords"]:
                if keyword in combined_text:
                    return rule["subactivity"], rule["activity"]
        
        # Luego busca en las reglas predeterminadas
        from heuristic_rules import HEURISTIC_RULES
        for rule in HEURISTIC_RULES:
            for keyword in rule["keywords"]:
                if keyword in app or keyword in title:
                    return rule["subactivity"], rule["activity"]
        
        return None, None
    
    def export_rules(self) -> str:
        """Exporta las reglas como JSON string"""
        return json.dumps(self.user_rules, ensure_ascii=False, indent=2)
    
    def import_rules(self, json_string: str, replace: bool = False) -> bool:
        """Importa reglas desde un JSON string"""
        try:
            imported_rules = json.loads(json_string)
            
            # Validar estructura básica
            for rule in imported_rules:
                if not all(key in rule for key in ["keywords", "subactivity", "activity"]):
                    raise ValueError("Invalid rule format")
            
            if replace:
                # Reemplazar todas las reglas
                st.session_state.user_heuristic_rules = imported_rules
            else:
                # Asignar nuevos IDs para evitar conflictos
                max_id = max([rule["id"] for rule in self.user_rules], default=0)
                for i, rule in enumerate(imported_rules):
                    rule["id"] = max_id + i + 1
                    rule["active"] = rule.get("active", True)
                    rule["created_by"] = "imported"
                    if "created_at" not in rule:
                        rule["created_at"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                
                st.session_state.user_heuristic_rules.extend(imported_rules)
            
            # NUEVO: Sincronizar actividades/subactividades personalizadas
            self._sync_custom_categories()
            
            return True
        except Exception as e:
            st.error(f"Error importing rules: {str(e)}")
            return False
    
    def _sync_custom_categories(self):
        """Sincroniza actividades y subactividades desde las reglas al catálogo"""
        # Inicializar si no existe
        if "custom_categories" not in st.session_state:
            st.session_state.custom_categories = {
                "activities": set(),
                "subactivities": {}  # {activity: [subactivities]}
            }
        
        # Cargar actividades existentes de core_act.py
        try:
            from core_act import load_activities
            dicc_core, dicc_subact, _, _ = load_activities()
            
            existing_activities = set()
            for category in dicc_core.values():
                for act in category:
                    existing_activities.add(act['core_activity'])
        except:
            existing_activities = set()
        
        # Extraer categorías de las reglas
        for rule in self.user_rules:
            activity = rule.get("activity", "").strip()
            subactivity = rule.get("subactivity", "").strip()
            
            if not activity:
                continue
            
            # Si la actividad no existe en core_act, añadirla como personalizada
            if activity not in existing_activities:
                st.session_state.custom_categories["activities"].add(activity)
            
            # Añadir subactividad
            if activity not in st.session_state.custom_categories["subactivities"]:
                st.session_state.custom_categories["subactivities"][activity] = set()
            
            if subactivity:
                st.session_state.custom_categories["subactivities"][activity].add(subactivity)
    
    def get_all_activities(self):
        """Devuelve TODAS las actividades: predefinidas + personalizadas"""
        # Cargar de core_act.py
        try:
            from core_act import load_activities
            dicc_core, _, _, _ = load_activities()
            
            activities = []
            for category in dicc_core.values():
                for act in category:
                    activities.append(act['core_activity'])
        except:
            activities = []
        
        # Añadir personalizadas
        if "custom_categories" in st.session_state:
            activities.extend(list(st.session_state.custom_categories["activities"]))
        
        return sorted(set(activities))
    
    def get_all_subactivities(self, activity: str):
        """Devuelve TODAS las subactividades de una actividad: predefinidas + personalizadas"""
        # Cargar de core_act.py
        try:
            from core_act import load_activities
            _, dicc_subact, _, _ = load_activities()
            subactivities = dicc_subact.get(activity, [])
        except:
            subactivities = []
        
        # Añadir personalizadas
        if "custom_categories" in st.session_state:
            custom_subs = st.session_state.custom_categories["subactivities"].get(activity, set())
            subactivities.extend(list(custom_subs))
        
        return sorted(set(subactivities))
    
    def get_color_for_activity(self, activity: str) -> str:
        """Devuelve color para una actividad (predefinida o genera uno para personalizada)"""
        
        # Validar que activity sea un string válido PRIMERO
        if activity is None or (isinstance(activity, float) and pd.isna(activity)):
            return '#CCCCCC'  # Color gris por defecto para valores vacíos
        
        # Convertir a string por seguridad
        activity = str(activity)
        
        # Cargar colores predefinidos
        try:
            from core_act import load_activities
            _, _, _, dicc_core_color = load_activities()
            
            if activity in dicc_core_color:
                return dicc_core_color[activity]
        except:
            pass
        
        # Generar color para actividad personalizada
        if "custom_activity_colors" not in st.session_state:
            st.session_state.custom_activity_colors = {}
        
        if activity not in st.session_state.custom_activity_colors:
            # Generar color basado en hash del nombre
            import hashlib
            hash_value = int(hashlib.md5(activity.encode()).hexdigest()[:6], 16)
            
            # Convertir a color pastel
            r = (hash_value >> 16) & 0xFF
            g = (hash_value >> 8) & 0xFF
            b = hash_value & 0xFF
            
            # Hacer más claro (pastel)
            r = int((r + 255) / 2)
            g = int((g + 255) / 2)
            b = int((b + 255) / 2)
            
            color = f"#{r:02X}{g:02X}{b:02X}"
            st.session_state.custom_activity_colors[activity] = color
        
        return st.session_state.custom_activity_colors[activity]


def show_custom_heuristic_interface(heuristic_manager: CustomHeuristicManager):
    """Interface de Streamlit para gestionar reglas heurísticas"""
    
    # Pestañas para organizar funcionalidades
    tab1, tab2, tab3, tab4 = st.tabs(["➕ Add Rule", "📝 Manage Rules", "📊 Test Rules", "🔄 Import/Export"])
    
    with tab1:
        st.markdown("#### Create New Rule")
        
        with st.form("nueva_regla", clear_on_submit=True):
            col1, col2 = st.columns(2)
            
            with col1:
                keywords_input = st.text_area(
                    "Keywords (one per line)",
                    placeholder="Example:\nzoom\nmicrosoft teams\nmeeting",
                    help="Enter keywords that will identify this activity"
                )
                subactivity = st.text_input(
                    "Subactivity",
                    placeholder="e.g., Team meetings"
                )
            
            with col2:
                activity = st.text_input(
                    "Activity",
                    placeholder="e.g., Coordination"
                )
                description = st.text_area(
                    "Description (optional)",
                    placeholder="Describe when to apply this rule"
                )
            
            submitted = st.form_submit_button("Create Rule")
            
            if submitted:
                keywords_list = [kw.strip() for kw in keywords_input.split('\n') if kw.strip()]
                if keywords_list and subactivity and activity:
                    if heuristic_manager.add_rule(keywords_list, subactivity, activity, description):
                        st.success("✅ Rule added successfully! You can continue adding more rules.")
                        # NO HACER st.rerun() AQUÍ
                    else:
                        st.error("Error adding rule")
                else:
                    st.error("Please complete all required fields")
    
    with tab2:
        st.markdown("#### Existing Rules")
        
        if not heuristic_manager.user_rules:
            st.info("No custom rules yet. Create your first one!")
        else:
            for rule in heuristic_manager.user_rules:
                status_icon = '🟢' if rule.get('active', True) else '🔴'
                
                with st.container():
                    st.markdown(f"**{status_icon} {rule['subactivity']} → {rule['activity']}**")
                    
                    col1, col2, col3 = st.columns([3, 1, 1])
                    
                    with col1:
                        st.caption(f"**Keywords:** {', '.join(rule['keywords'])}")
                        if rule.get('description'):
                            st.caption(f"**Description:** {rule['description']}")
                        st.caption(f"**Status:** {'Active' if rule.get('active', True) else 'Inactive'}")
                    
                    with col2:
                        if st.button("🔄 Toggle", key=f"toggle_{rule['id']}"):
                            heuristic_manager.toggle_rule(rule['id'])
                            # st.rerun() se ejecuta automáticamente al hacer clic
                    
                    with col3:
                        if st.button("🗑️ Delete", key=f"delete_{rule['id']}"):
                            heuristic_manager.delete_rule(rule['id'])
                            st.success("Rule deleted")
                            # st.rerun() se ejecuta automáticamente al hacer clic
                    
                    st.divider()
    
    with tab3:
        st.markdown("#### Test Classification")
        
        col1, col2 = st.columns(2)
        with col1:
            test_app = st.text_input("Application name", placeholder="chrome.exe", key="test_app_input")
        with col2:
            test_title = st.text_input("Window title", placeholder="Zoom Meeting", key="test_title_input")
        
        if st.button("🧪 Test Classification"):
            if test_app or test_title:
                subact, act = heuristic_manager.match_heuristic_rule(test_app, test_title)
                if subact and act:
                    st.success(f"✅ **Result:** {subact} → {act}")
                else:
                    st.warning("⚠️ No matching rules found")
            else:
                st.error("Enter at least the app or title")
        
        # Mostrar estadísticas
        st.markdown("#### Rules Statistics")
        total_rules = len(heuristic_manager.user_rules)
        active_rules = len(heuristic_manager.get_active_rules())
        
        col1, col2, col3 = st.columns(3)
        col1.metric("Total rules", total_rules)
        col2.metric("Active rules", active_rules)
        col3.metric("Inactive rules", total_rules - active_rules)
    
    with tab4:
        st.markdown("#### Import/Export Rules")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**📤 Export current rules**")
            
            total_rules = len(heuristic_manager.user_rules)
            
            if total_rules > 0:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = f"my_heuristic_rules_{timestamp}.json"
                rules_json = heuristic_manager.export_rules()
                
                st.download_button(
                    label="💾 Download my rules",
                    data=rules_json,
                    file_name=filename,
                    mime="application/json",
                    help="Download a JSON file with all your rules"
                )
                
                st.info("💡 Save this file to restore your rules in future sessions")
            else:
                st.info("No rules to export")
        
        with col2:
            st.markdown("**📥 Import rules from file**")
            
            # Usar label vacío para que solo se vea el área de drop
            uploaded_file = st.file_uploader(
                label="import_file",  # Oculto visualmente
                type=["json"],
                help="Upload a JSON file with previously saved rules",
                key="import_rules_uploader",
                label_visibility="collapsed"  # ESTO OCULTA EL LABEL
            )
            
            if uploaded_file is not None:
                try:
                    content = uploaded_file.read().decode('utf-8')
                    rules_preview = json.loads(content)
                    
                    st.success(f"📄 Valid file with {len(rules_preview)} rules")
                    
                    col_a, col_b = st.columns(2)
                    
                    with col_a:
                        if st.button("🔄 Replace my rules", type="primary", key="replace_rules_btn"):
                            if heuristic_manager.import_rules(content, replace=True):
                                st.success("✅ Rules replaced")
                                # Aquí SÍ necesitamos rerun porque cambió todo
                                st.rerun()
                    
                    with col_b:
                        if st.button("➕ Add to existing", key="add_rules_btn"):
                            if heuristic_manager.import_rules(content, replace=False):
                                st.success("✅ Rules added")
                                # Aquí SÍ necesitamos rerun
                                st.rerun()
                
                except json.JSONDecodeError:
                    st.error("❌ File is not valid JSON")
                except Exception as e:
                    st.error(f"❌ Error: {str(e)}")
        
        # Reglas de ejemplo
        st.markdown("---")
        st.markdown("**🎯 Sample rules**")
        st.markdown("First time? Download these sample rules to get started:")
        
        sample_rules = [
            {
                "id": 1,
                "keywords": ["zoom", "teams", "meet", "reunion"],
                "subactivity": "Video conferences",
                "activity": "Communication",
                "description": "Virtual meetings",
                "active": True,
                "created_by": "sample"
            },
            {
                "id": 2,
                "keywords": ["gmail", "outlook", "email", "mail"],
                "subactivity": "Email management",
                "activity": "Communication",
                "description": "Reading and responding to emails",
                "active": True,
                "created_by": "sample"
            },
            {
                "id": 3,
                "keywords": ["code", "python", "vscode", "programming"],
                "subactivity": "Programming",
                "activity": "Development",
                "description": "Writing code",
                "active": True,
                "created_by": "sample"
            }
        ]
        
        sample_json = json.dumps(sample_rules, ensure_ascii=False, indent=2)
        
        st.download_button(
            label="📥 Download sample rules",
            data=sample_json,
            file_name="sample_heuristic_rules.json",
            mime="application/json"
        )

# Integración con la clasificación existente
def clasificar_por_heuristica_personalizada(app: str, title: str, heuristic_manager: CustomHeuristicManager) -> Tuple[Optional[str], Optional[str]]:
    """
    Función actualizada que usa el gestor de reglas personalizadas
    """
    return heuristic_manager.match_heuristic_rule(app, title)


# Ejemplo de uso en tu aplicación principal
def integrate_custom_heuristics():
    """Ejemplo de cómo integrar el sistema en tu app principal"""
    
    # Inicializar el gestor de reglas (solo una vez en tu aplicación)
    if "heuristic_manager" not in st.session_state:
        st.session_state.heuristic_manager = CustomHeuristicManager()
    
    heuristic_manager = st.session_state.heuristic_manager
    
    # Mostrar la interfaz de gestión de reglas
    show_custom_heuristic_interface(heuristic_manager)
    
    return heuristic_manager
