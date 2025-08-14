import streamlit.components.v1 as components

def change_color(element_type, widget_label, font_color, background_color='transparent'):
    if element_type == 'select_box':
        query_selector = 'label'
    elif element_type == 'button':
        query_selector = 'button'
    else:
        return

    htmlstr = f"""
        <script>
            var elements = window.parent.document.querySelectorAll('{query_selector}');
            for (var i = 0; i < elements.length; ++i) {{
                if (elements[i].innerText == '{widget_label}') {{
                    elements[i].style.color ='{font_color}';
                    elements[i].style.background = '{background_color}'
                }}
            }}
        </script>
    """
    components.html(f"{htmlstr}", height=1, width=1)

