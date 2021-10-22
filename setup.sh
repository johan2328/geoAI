mkdir -p ~ / .streamlit /

echo "
[general] \ n
email =“ your-email@domain.com ”\ n
"> ~ / .streamlit / credentials.toml

echo "
[servidor] \ n
headless = true \ n
enableCORS = false \ n
port = $ PORT \ n
[tema] \ n
primaryColor =" # 08424f "\ n
backgroundColor =" # 08424f "\ n
secundariaBackgroundColor =" # 0d5a6a " \ n
textColor = " #ffffff " \ n
font = "sans serif" \ n
"> ~ / .streamlit / config.toml