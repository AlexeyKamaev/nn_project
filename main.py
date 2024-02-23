import streamlit as st


st.markdown("""
    <style>
        section[data-testid="stSidebar"][aria-expanded="true"]{
            display: none;
        }
    </style>
    """, unsafe_allow_html=True)



st.title('Neural project')



st.write('choose your option')



st.page_link("pages/cell_dete.py", label="Cell detector", icon='🦠')
st.page_link("pages/streamlit_sport_model.py", label="Sport detector", icon='🏀')

st.header(f'''made by: Alexey Kamaev & Valery Fedorov''')