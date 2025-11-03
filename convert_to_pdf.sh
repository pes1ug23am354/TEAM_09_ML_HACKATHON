#!/bin/bash

# Script to convert Report.md to Report.pdf
# This script provides multiple methods to create the PDF

echo "=========================================="
echo "Report.md to Report.pdf Converter"
echo "=========================================="
echo ""

# Check if Report.md exists
if [ ! -f "Report.md" ]; then
    echo "❌ Error: Report.md not found!"
    exit 1
fi

echo "✅ Found Report.md"
echo ""
echo "Choose a method to convert to PDF:"
echo ""
echo "METHOD 1: Using Python markdown2 + weasyprint (Recommended)"
echo "  Install: pip3 install markdown2 weasyprint"
echo "  Command: python3 -c \"import markdown2; from weasyprint import HTML; html = markdown2.markdown(open('Report.md').read(), extras=['tables', 'fenced-code-blocks']); HTML(string=f'<html><head><style>body{{font-family: Arial; margin: 40px; line-height: 1.6;}} h1{{color: #1a5490;}} h2{{color: #2c5aa0;}} table{{border-collapse: collapse; width: 100%;}} th, td{{border: 1px solid #ddd; padding: 8px; text-align: left;}} th{{background-color: #1a5490; color: white;}}</style></head><body>{html}</body></html>').write_pdf('Report.pdf')\""
echo ""
echo "METHOD 2: Using pandoc (If installed)"
echo "  Install: brew install pandoc"
echo "  Command: pandoc Report.md -o Report.pdf --pdf-engine=pdflatex"
echo ""
echo "METHOD 3: Using grip (GitHub-style rendering)"
echo "  Install: pip3 install grip"
echo "  Command: grip Report.md --export Report.html"
echo "  Then: Open Report.html in browser and Print to PDF"
echo ""
echo "METHOD 4: Manual (Easiest, No Installation)"
echo "  1. Open Report.md in any text editor or VS Code"
echo "  2. Use VS Code Markdown Preview (Cmd+Shift+V)"
echo "  3. Right-click preview → 'Open in Browser'"
echo "  4. In browser: File → Print → Save as PDF"
echo ""
echo "METHOD 5: Using online converter"
echo "  1. Go to: https://www.markdowntopdf.com/"
echo "  2. Upload Report.md"
echo "  3. Download Report.pdf"
echo ""
echo "=========================================="
echo "Attempting automatic conversion..."
echo "=========================================="

# Try method 1 (Python)
if command -v python3 &> /dev/null; then
    echo "Trying Python method..."
    python3 << 'PYTHON_SCRIPT'
try:
    import markdown2
    from weasyprint import HTML
    
    with open('Report.md', 'r') as f:
        md_content = f.read()
    
    html_content = markdown2.markdown(md_content, extras=['tables', 'fenced-code-blocks', 'header-ids'])
    
    full_html = f'''
    <html>
    <head>
        <meta charset="utf-8">
        <style>
            body {{
                font-family: Arial, sans-serif;
                margin: 40px;
                line-height: 1.6;
                color: #333;
            }}
            h1 {{
                color: #1a5490;
                border-bottom: 3px solid #1a5490;
                padding-bottom: 10px;
                page-break-before: always;
            }}
            h1:first-of-type {{
                page-break-before: avoid;
            }}
            h2 {{
                color: #2c5aa0;
                margin-top: 30px;
            }}
            h3 {{
                color: #333;
                margin-top: 20px;
            }}
            table {{
                border-collapse: collapse;
                width: 100%;
                margin: 20px 0;
            }}
            th, td {{
                border: 1px solid #ddd;
                padding: 12px;
                text-align: left;
            }}
            th {{
                background-color: #1a5490;
                color: white;
                font-weight: bold;
            }}
            tr:nth-child(even) {{
                background-color: #f9f9f9;
            }}
            code {{
                background-color: #f5f5f5;
                padding: 2px 6px;
                border-radius: 3px;
                font-family: 'Courier New', monospace;
            }}
            pre {{
                background-color: #f5f5f5;
                padding: 15px;
                border-radius: 5px;
                overflow-x: auto;
            }}
            ul, ol {{
                margin-left: 20px;
            }}
            strong {{
                color: #1a5490;
            }}
            @page {{
                margin: 1in;
            }}
        </style>
    </head>
    <body>
        {html_content}
    </body>
    </html>
    '''
    
    HTML(string=full_html).write_pdf('Report.pdf')
    print("✅ SUCCESS! Report.pdf created using Python method")
    exit(0)
except ImportError as e:
    print(f"⚠️  Python libraries not installed: {e}")
    print("   Install with: pip3 install markdown2 weasyprint")
except Exception as e:
    print(f"⚠️  Python method failed: {e}")
PYTHON_SCRIPT

    if [ $? -eq 0 ]; then
        exit 0
    fi
fi

# Try method 2 (pandoc)
if command -v pandoc &> /dev/null; then
    echo "Trying pandoc method..."
    pandoc Report.md -o Report.pdf --pdf-engine=pdflatex 2>/dev/null
    if [ $? -eq 0 ]; then
        echo "✅ SUCCESS! Report.pdf created using pandoc"
        exit 0
    else
        echo "⚠️  Pandoc method failed (pdflatex may not be installed)"
    fi
fi

echo ""
echo "=========================================="
echo "❌ Automatic conversion failed"
echo "=========================================="
echo ""
echo "Please use one of the manual methods above."
echo "The easiest is METHOD 4 (Manual using browser)."
echo ""
echo "Report.md is ready and contains all the content!"
