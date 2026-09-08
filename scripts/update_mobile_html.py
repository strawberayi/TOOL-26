with open('index.html', 'r', encoding='utf-8') as f:
    html = f.read()

# 1. Update viewport meta to include viewport-fit=cover
html = html.replace(
    '<meta name="viewport" content="width=device-width, initial-scale=1.0, maximum-scale=1.0, user-scalable=no">',
    '<meta name="viewport" content="width=device-width, initial-scale=1.0, maximum-scale=1.0, user-scalable=no, viewport-fit=cover">'
)

print("Updated viewport tag")
with open('index.html', 'w', encoding='utf-8') as f:
    f.write(html)
