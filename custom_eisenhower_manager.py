# custom_eisenhower_manager.py

import json
import streamlit as st
from typing import List, Dict, Any, Optional
from datetime import datetime

class CustomEisenhowerManager:
    """Gestor de reglas de Eisenhower personalizables por el usuario"""
    
    QUADRANTS = [
        "I: Urgent & Important",
        "II: Not urgent but Important", 
        "III: Urgent but Not important",
        "IV: Not urgent & Not important"
    ]
    
    def __init__(self):
        if "user_eisenhower_rules" not in st.session_state:
            st.session_state.user_eisenhower_rules = []
    
    @property
    def user_rules(self) -> List[Dict[str, Any]]:
        return st.session_state.user_eisenhower_rules
    
    def add_rule(self, keywords: List[str], quadrant: str, description: str = "") -> bool:
        """Añade una regla para clasificar en Eisenhower"""
        new_id = max([rule["id"] for rule in self.user_rules], default=0) + 1
        
        new_rule = {
            "id": new_id,
            "keywords": [kw.strip().lower() for kw in keywords if kw.strip()],
            "quadrant": quadrant,
            "description": description.strip(),
            "active": True,
            "created_by": "user",
            "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }
        
        if not new_rule["keywords"] or not new_rule["quadrant"]:
            return False
        
        st.session_state.user_eisenhower_rules.append(new_rule)
        return True
    
    def delete_rule(self, rule_id: int) -> bool:
        st.session_state.user_eisenhower_rules = [
            rule for rule in self.user_rules if rule["id"] != rule_id
        ]
        return True
    
    def toggle_rule(self, rule_id: int) -> bool:
        for rule in self.user_rules:
            if rule["id"] == rule_id:
                rule["active"] = not rule["active"]
                return True
        return False
    
    def get_active_rules(self) -> List[Dict[str, Any]]:
        return [rule for rule in self.user_rules if rule.get("active", True)]
    
    def classify_eisenhower(self, subactivity: str) -> Optional[str]:
        """Clasifica una subactividad usando reglas personalizadas + predeterminadas"""
        if not subactivity or not isinstance(subactivity, str):
            return None
        
        subactivity_lower = subactivity.lower()
        
        # 1. PRIORIDAD: Reglas del usuario
        for rule in self.get_active_rules():
            for keyword in rule["keywords"]:
                if keyword in subactivity_lower:
                    return rule["quadrant"]
        
        # 2. FALLBACK: Reglas predeterminadas
        from heuristic_eisenhower import clasificar_eisenhower_por_heuristica
        return clasificar_eisenhower_por_heuristica(subactivity)
    
    def export_rules(self) -> str:
        return json.dumps(self.user_rules, ensure_ascii=False, indent=2)
    
    def import_rules(self, json_string: str, replace: bool = False) -> bool:
        try:
            imported_rules = json.loads(json_string)
            
            for rule in imported_rules:
                if not all(key in rule for key in ["keywords", "quadrant"]):
                    raise ValueError("Invalid rule format")
            
            if replace:
                st.session_state.user_eisenhower_rules = imported_rules
            else:
                max_id = max([rule["id"] for rule in self.user_rules], default=0)
                for i, rule in enumerate(imported_rules):
                    rule["id"] = max_id + i + 1
                    rule["active"] = rule.get("active", True)
                    rule["created_by"] = "imported"
                    if "created_at" not in rule:
                        rule["created_at"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                
                st.session_state.user_eisenhower_rules.extend(imported_rules)
            
            return True
        except Exception as e:
            st.error(f"Error importing rules: {str(e)}")
            return False
        
def show_eisenhower_rules_interface(eisenhower_manager: CustomEisenhowerManager):
    """Interfaz para gestionar reglas de Eisenhower"""
    
    tab1, tab2, tab3, tab4 = st.tabs(["➕ Add Rule", "📝 Manage Rules", "📊 Test", "🔄 Import/Export"])
    
    with tab1:
        st.markdown("#### Create New Eisenhower Rule")
        st.markdown("Define keywords that identify subactivities belonging to a specific quadrant.")
        
        with st.form("nueva_regla_eisenhower", clear_on_submit=True):
            keywords_input = st.text_area(
                "Keywords (one per line)",
                placeholder="Example:\nurgent\ndeadline\nemergency",
                help="Keywords found in subactivity names"
            )
            
            quadrant = st.selectbox(
                "Eisenhower Quadrant",
                options=CustomEisenhowerManager.QUADRANTS,
                help="Select the quadrant for activities matching these keywords"
            )
            
            description = st.text_area(
                "Description (optional)",
                placeholder="Explain when to apply this rule"
            )
            
            submitted = st.form_submit_button("Create Rule")
            
            if submitted:
                keywords_list = [kw.strip() for kw in keywords_input.split('\n') if kw.strip()]
                if keywords_list and quadrant:
                    if eisenhower_manager.add_rule(keywords_list, quadrant, description):
                        st.success("✅ Rule added successfully!")
                    else:
                        st.error("Error adding rule")
                else:
                    st.error("Please complete all required fields")
    
    with tab2:
        st.markdown("#### Existing Rules")
        
        if not eisenhower_manager.user_rules:
            st.info("No custom Eisenhower rules yet. Create your first one!")
        else:
            # Agrupar por cuadrante para mejor visualización
            for quadrant in CustomEisenhowerManager.QUADRANTS:
                rules_in_quadrant = [r for r in eisenhower_manager.user_rules if r["quadrant"] == quadrant]
                
                if rules_in_quadrant:
                    st.markdown(f"##### {quadrant}")
                    
                    for rule in rules_in_quadrant:
                        status_icon = '🟢' if rule.get('active', True) else '🔴'
                        
                        with st.container():
                            col1, col2, col3 = st.columns([3, 1, 1])
                            
                            with col1:
                                st.markdown(f"**{status_icon} Keywords:** {', '.join(rule['keywords'])}")
                                if rule.get('description'):
                                    st.caption(f"*{rule['description']}*")
                            
                            with col2:
                                if st.button("🔄", key=f"toggle_eisen_{rule['id']}"):
                                    eisenhower_manager.toggle_rule(rule['id'])
                            
                            with col3:
                                if st.button("🗑️", key=f"delete_eisen_{rule['id']}"):
                                    eisenhower_manager.delete_rule(rule['id'])
                                    st.success("Rule deleted")
                            
                            st.divider()
    
    with tab3:
        st.markdown("#### Test Classification")
        
        test_subactivity = st.text_input(
            "Subactivity to test",
            placeholder="e.g., Email management"
        )
        
        if st.button("🧪 Test Eisenhower Classification"):
            if test_subactivity:
                result = eisenhower_manager.classify_eisenhower(test_subactivity)
                if result:
                    st.success(f"✅ **Result:** {result}")
                else:
                    st.warning("⚠️ No matching rules found")
            else:
                st.error("Enter a subactivity to test")
        
        # Estadísticas
        st.markdown("#### Rules Statistics")
        total_rules = len(eisenhower_manager.user_rules)
        active_rules = len(eisenhower_manager.get_active_rules())
        
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Total rules", total_rules)
        col2.metric("Active", active_rules)
        col3.metric("Inactive", total_rules - active_rules)
        
        # Distribución por cuadrante
        quadrant_counts = {}
        for quadrant in CustomEisenhowerManager.QUADRANTS:
            count = len([r for r in eisenhower_manager.user_rules if r["quadrant"] == quadrant])
            quadrant_counts[quadrant] = count
        
        col4.metric("Quadrants used", sum(1 for c in quadrant_counts.values() if c > 0))
    
    with tab4:
        st.markdown("#### Import/Export Rules")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**📤 Export rules**")
            
            if len(eisenhower_manager.user_rules) > 0:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = f"eisenhower_rules_{timestamp}.json"
                rules_json = eisenhower_manager.export_rules()
                
                st.download_button(
                    label="💾 Download rules",
                    data=rules_json,
                    file_name=filename,
                    mime="application/json"
                )
            else:
                st.info("No rules to export")
        
        with col2:
            st.markdown("**📥 Import rules**")
            
            uploaded_file = st.file_uploader(
                "Select JSON file",
                type=["json"],
                key="import_eisenhower_uploader",
                label_visibility="collapsed"
            )
            
            if uploaded_file is not None:
                try:
                    content = uploaded_file.read().decode('utf-8')
                    rules_preview = json.loads(content)
                    
                    st.success(f"📄 Valid file with {len(rules_preview)} rules")
                    
                    col_a, col_b = st.columns(2)
                    
                    with col_a:
                        if st.button("🔄 Replace", type="primary", key="replace_eisen_btn"):
                            if eisenhower_manager.import_rules(content, replace=True):
                                st.success("✅ Rules replaced")
                                st.rerun()
                    
                    with col_b:
                        if st.button("➕ Add", key="add_eisen_btn"):
                            if eisenhower_manager.import_rules(content, replace=False):
                                st.success("✅ Rules added")
                                st.rerun()
                
                except json.JSONDecodeError:
                    st.error("❌ Invalid JSON file")
                except Exception as e:
                    st.error(f"❌ Error: {str(e)}")