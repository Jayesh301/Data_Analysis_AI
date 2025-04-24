mkdir -p ~/.streamlit/

echo "\
[theme]\n\
primaryColor='#4B8BBE'\n\
backgroundColor='#FFFFFF'\n\
secondaryBackgroundColor='#F0F2F6'\n\
textColor='#262730'\n\
font='sans serif'\n\
\n\
[server]\n\
headless = true\n\
port = $PORT\n\
enableCORS = false\n\
" > ~/.streamlit/config.toml 